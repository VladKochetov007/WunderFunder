"""Run the hash-pinned architecture GPU packet in preflight, fit, and grade stages.

This is a target-host orchestration entry point. It is intentionally not run on
the preparation Mac: preflight executes CUDA tests and training workers.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prepare_architecture_gpu_wave import (
    CAMPAIGN,
    REQUIRED_GPU_TESTS,
    PreparationError,
    canonical_bytes,
    sha256_file,
    source_lock,
    validate_data,
)


class WaveError(RuntimeError):
    """Raised when packet identity, execution, provenance, or grading fails."""


def atomic_json_write(path: Path, value: object) -> None:
    """Atomically replace a JSON status/report file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, label: str) -> dict[str, Any]:
    """Read one JSON object with a packet-specific diagnostic."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WaveError(f"cannot read {label} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise WaveError(f"{label} at {path} must be a JSON object")
    return value


def verify_data_identity(saved: dict[str, Any], current: dict[str, Any]) -> None:
    """Compare content and membership identities while ignoring file mtimes."""
    def stable(value: dict[str, Any]) -> dict[str, Any]:
        sources = {}
        for name, source in value["sources"].items():
            sources[name] = {key: item for key, item in source.items() if key != "mtime_ns"}
        return {**value, "sources": sources}

    if canonical_bytes(stable(saved)) != canonical_bytes(stable(current)):
        raise WaveError("current fold/data identity differs from the prepared manifest")


def verify_packet(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify packet manifests, source closure, split pin, data, and worker specs."""
    manifest_path = manifest_path.resolve()
    output_dir = manifest_path.parent
    manifest = read_json(manifest_path, "wave manifest")
    sidecar = read_json(output_dir / "manifest.sha256.json", "manifest SHA256 sidecar")
    if sidecar.get("manifest_sha256") != sha256_file(manifest_path):
        raise WaveError("wave manifest SHA256 does not match its sidecar")
    if manifest.get("schema") != "wnn.architecture-gpu-wave-preparation.v1":
        raise WaveError("unsupported architecture-wave manifest schema")
    if Path(manifest.get("output_dir", "")).resolve() != output_dir:
        raise WaveError("manifest output directory does not match its location")

    project_root = Path(manifest["project_root"]).resolve()
    data_root = Path(manifest["data_root"]).resolve()
    recipe_path = output_dir / "recipe.json"
    recipe = read_json(recipe_path, "frozen recipe")
    if sha256_file(recipe_path) != manifest.get("recipe_sha256"):
        raise WaveError("prepared recipe SHA256 mismatch")
    source_recipe = project_root / CAMPAIGN / "recipe.json"
    if sha256_file(source_recipe) != manifest.get("recipe_sha256"):
        raise WaveError("source checkout recipe differs from the prepared recipe")
    source_closure = read_json(Path(manifest["source_closure_path"]), "source closure")
    if source_closure.get("closure_sha256") != manifest.get("source_closure_sha256"):
        raise WaveError("prepared source-closure manifest hash mismatch")
    current_closure = source_lock(project_root, recipe)
    if current_closure.get("closure_sha256") != manifest.get("source_closure_sha256"):
        raise WaveError("source files changed after GPU preparation")

    data_path = Path(manifest["data_identity_path"])
    saved_data = read_json(data_path, "prepared data identity")
    if hashlib.sha256(canonical_bytes(saved_data)).hexdigest() != manifest.get(
        "data_identity_sha256"
    ):
        raise WaveError("prepared data-identity hash mismatch")
    split_path = Path(manifest["split_manifest"]).resolve()
    if split_path != (project_root / recipe["split"]["path"]).resolve():
        raise WaveError("prepared split path differs from the recipe pin")
    if sha256_file(split_path) != recipe["split"]["sha256"]:
        raise WaveError("frozen fold-0 split SHA256 differs from the recipe pin")
    portable = (
        None
        if manifest.get("portable_data_manifest") is None
        else Path(manifest["portable_data_manifest"]).resolve()
    )
    try:
        current_data = validate_data(project_root, data_root, split_path, portable)
    except (PreparationError, OSError, ValueError) as error:
        raise WaveError(f"target data validation failed: {error}") from error
    verify_data_identity(saved_data, current_data)

    queue_records = manifest.get("queues")
    if not isinstance(queue_records, list) or not queue_records:
        raise WaveError("manifest has no worker queue specs")
    if hashlib.sha256(canonical_bytes(queue_records)).hexdigest() != manifest.get(
        "spec_manifest_sha256"
    ):
        raise WaveError("worker-spec index SHA256 mismatch")
    for record in queue_records:
        queue = Path(record["queue"])
        run_id = record["run_id"]
        candidates = [
            queue / state / f"{run_id}.json"
            for state in ("pending", "running", "done", "failed")
        ]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise WaveError(
                f"expected exactly one pending/running/done/failed spec for {run_id}; "
                f"found {len(existing)}"
            )
        spec_path = existing[0]
        if sha256_file(spec_path) != record["spec_sha256"]:
            raise WaveError(f"worker spec hash changed for {run_id}")
        allowlist_path = Path(record["allowed_spec_manifest"])
        if sha256_file(allowlist_path) != record["allowed_spec_manifest_sha256"]:
            raise WaveError(f"worker allowlist hash changed for {run_id}")
        allowlist = read_json(allowlist_path, f"worker allowlist for {run_id}")
        if allowlist != {run_id: record["spec_sha256"]}:
            raise WaveError(f"worker allowlist does not admit exactly {run_id}")
    return manifest, recipe, current_data


def latest_worker_result(queue: Path, run_id: str) -> dict[str, Any] | None:
    """Return the latest worker result for a run, if present."""
    results = queue / "results.jsonl"
    if not results.is_file():
        return None
    last = None
    try:
        for line in results.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if isinstance(record, dict) and record.get("run_id") == run_id:
                last = record
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WaveError(f"cannot read worker results at {results}: {error}") from error
    return last


def worker_spec_path(record: dict[str, Any]) -> Path:
    """Find a worker spec after the queue moves it between state directories."""
    queue = Path(record["queue"])
    run_id = record["run_id"]
    found = [
        queue / state / f"{run_id}.json"
        for state in ("pending", "running", "done", "failed")
        if (queue / state / f"{run_id}.json").is_file()
    ]
    if len(found) != 1:
        raise WaveError(f"worker spec for {run_id} is missing or duplicated")
    return found[0]


def require_done(record: dict[str, Any]) -> None:
    """Require an exact admitted worker spec and its successful result record."""
    queue = Path(record["queue"])
    run_id = record["run_id"]
    done_path = queue / "done" / f"{run_id}.json"
    if not done_path.is_file() or sha256_file(done_path) != record["spec_sha256"]:
        raise WaveError(f"worker did not complete the admitted spec: {run_id}")
    result = latest_worker_result(queue, run_id)
    if (
        result is None
        or result.get("status") != "succeeded"
        or result.get("spec_sha256") != record["spec_sha256"]
    ):
        raise WaveError(f"done queue item lacks its successful exact-spec result: {run_id}")


def running_owner_is_alive(queue: Path, run_id: str) -> bool:
    """Fail closed unless a running queue item has a provably dead local owner."""
    heartbeat_path = queue / "worker_heartbeat.json"
    try:
        heartbeat = read_json(heartbeat_path, "queue worker heartbeat")
    except WaveError:
        return True
    host = heartbeat.get("host")
    pid = heartbeat.get("pid")
    if host != socket.gethostname() or not isinstance(pid, int) or pid <= 0:
        return True
    if heartbeat.get("current_run_id") not in (None, run_id):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def run_worker(record: dict[str, Any], python: Path) -> None:
    """Run exactly one admitted queue item or validate its prior completion."""
    queue = Path(record["queue"])
    run_id = record["run_id"]
    spec_path = worker_spec_path(record)
    if sha256_file(spec_path) != record["spec_sha256"]:
        raise WaveError(f"worker spec hash changed before launch: {run_id}")
    state = spec_path.parent.name
    result = latest_worker_result(queue, run_id)
    if state == "done":
        require_done(record)
        return
    if state == "failed":
        raise WaveError(f"queue item is failed and requires diagnosis: {run_id}")
    if state == "running":
        if running_owner_is_alive(queue, run_id):
            raise WaveError(f"queue item has a live or unverified owner; refusing duplicate worker: {run_id}")
        # The established worker owns stale-item recovery. It resumes only a
        # checkpointed step and fails closed if the saved state is incomplete.

    spec = read_json(spec_path, f"worker spec {run_id}")
    steps = spec.get("steps")
    if not isinstance(steps, list) or not steps or not isinstance(steps[0], dict):
        raise WaveError(f"worker spec has no first step environment: {run_id}")
    step_env = steps[0].get("env", {})
    if not isinstance(step_env, dict):
        raise WaveError(f"worker step environment is invalid: {run_id}")
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in step_env.items()})
    env["PYTHONUNBUFFERED"] = "1"
    log_path = queue / "logs" / f"{run_id}.launcher.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        str(python),
        "-u",
        str(Path(spec["steps"][0]["cwd"]) / "scripts/warm_queue_worker.py"),
        "--queue",
        str(queue),
        "--once",
        "--allowed-spec-manifest",
        str(record["allowed_spec_manifest"]),
    ]
    with log_path.open("ab") as log:
        log.write(
            f"\n[{datetime.now(UTC).isoformat()}] START {run_id}\n".encode()
        )
        completed = subprocess.run(
            argv,
            cwd=steps[0]["cwd"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(
            f"[{datetime.now(UTC).isoformat()}] EXIT {completed.returncode}\n".encode()
        )
    result = latest_worker_result(queue, run_id)
    if (
        completed.returncode != 0
        or result is None
        or result.get("status") != "succeeded"
        or result.get("spec_sha256") != record["spec_sha256"]
    ):
        status = None if result is None else result.get("status")
        raise WaveError(
            f"worker {run_id} failed (launcher_exit={completed.returncode}, result={status}); "
            f"inspect {log_path}"
        )
    done_path = queue / "done" / f"{run_id}.json"
    if not done_path.is_file():
        raise WaveError(f"worker {run_id} reported success without moving its spec to done")
    require_done(record)


def run_worker_pool(
    records: list[dict[str, Any]],
    arm_gpu: dict[str, str],
    gpu_ids: list[str],
    python: Path,
) -> None:
    """Run parallel queue workers with no more than one worker per physical GPU."""
    by_gpu: dict[str, list[dict[str, Any]]] = {gpu: [] for gpu in gpu_ids}
    for record in records:
        gpu = arm_gpu.get(record["arm"])
        if gpu is None or gpu not in by_gpu:
            raise WaveError(f"no unique physical GPU assignment for {record['run_id']}")
        by_gpu[gpu].append(record)

    def run_gpu_lane(lane: list[dict[str, Any]]) -> list[str]:
        failures = []
        for record in lane:
            try:
                run_worker(record, python)
            except WaveError as error:
                failures.append(str(error))
        return failures

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = [pool.submit(run_gpu_lane, by_gpu[gpu]) for gpu in gpu_ids]
        for future in concurrent.futures.as_completed(futures):
            failures.extend(future.result())
    if failures:
        raise WaveError("GPU worker stage had failures: " + " | ".join(failures))


def validate_provenance(
    run_ids: list[str],
    data_root: Path,
    recipe: dict[str, Any],
    *,
    preflight: bool,
    gpu_memory_by_arm: dict[str, int],
) -> dict[str, Any]:
    """Require matching fit membership, norm, order, and finite completed updates."""
    common = recipe["common"]
    reference: dict[str, Any] | None = None
    common_updates: tuple[int, int, int] | None = None
    results: dict[str, Any] = {}
    arms_by_run_id = {
        arm["preflight_run_id"] if preflight else arm["run_id"]: arm
        for arm in recipe["arms"]
    }
    expected_batches = (
        common["preflight_batches"]
        if preflight
        else common["full_epoch_batches"]
    )
    expected_attempts = expected_batches * common["optimizer_updates_per_batch"]
    for run_id in run_ids:
        arm = arms_by_run_id.get(run_id)
        if arm is None:
            raise WaveError(f"{run_id} is not a declared {'preflight' if preflight else 'fit'} arm")
        run_root = data_root / "artifacts/runs" / run_id
        provenance = read_json(run_root / "provenance.json", f"{run_id} provenance")
        if provenance.get("split_manifest_sha256") != recipe["split"]["sha256"]:
            raise WaveError(f"{run_id} fit used a different split manifest")
        args = provenance.get("args", {})
        expected_args = {
            "seed": common["seed"],
            "epochs": common["epochs"],
            "train_source": common["fit_source"],
            "batch": common["batch"],
            "chunk": common["chunk"],
            "precision": common["precision"],
            "batch_order": common["batch_order"],
            "lr": common["learning_rate"],
            "wd": common["weight_decay"],
            "optim": common["optimizer"],
            "beta2": common["beta2"],
            "sched": common["scheduler"],
            "clip": common["gradient_clip"],
            "loss": common["loss"],
            "loss_norm": common["loss_norm"],
            "dropout": common["dropout"],
            "device": "cuda",
            "checkpoint_policy": common["checkpoint_policy"],
            "checkpoint_epoch": common["checkpoint_epoch"],
            "no_eval_during_fit": True,
            "max_batches": common["preflight_batches"] if preflight else 0,
            "check_finite_updates": True,
        }
        expected_args.update(arm["model"])
        for key, expected in expected_args.items():
            if args.get(key) != expected:
                raise WaveError(f"{run_id} trainer argument {key}={args.get(key)!r}, expected {expected!r}")
        if provenance.get("fit_membership_identity") is None:
            raise WaveError(f"{run_id} lacks fit membership identity")
        if provenance.get("normalization_identity") is None:
            raise WaveError(f"{run_id} lacks normalization identity")
        updates = provenance.get("update_diagnostics")
        if not isinstance(updates, list) or len(updates) != 1:
            raise WaveError(f"{run_id} must contain exactly one completed epoch record")
        diagnostic = updates[0]
        if diagnostic.get("epoch_index") != 0 or diagnostic.get("normalization_identity") != provenance["normalization_identity"]:
            raise WaveError(f"{run_id} epoch diagnostic does not match its epoch-0 normalization")
        epoch = diagnostic.get("epoch")
        if not isinstance(epoch, dict):
            raise WaveError(f"{run_id} has no optimizer update counters")
        attempts = epoch.get("optimizer_attempts")
        completed = epoch.get("optimizer_completed_updates")
        skipped = epoch.get("amp_skipped_updates")
        batches = epoch.get("completed_batches")
        if not all(isinstance(value, int) for value in (attempts, completed, skipped, batches)):
            raise WaveError(f"{run_id} has incomplete optimizer or batch counts")
        if (
            batches != expected_batches
            or attempts != expected_attempts
            or completed != expected_attempts
            or skipped != 0
        ):
            raise WaveError(
                f"{run_id} update counts do not match the declared "
                f"{'one-batch preflight' if preflight else 'full epoch'} budget: "
                f"batches={batches}, attempts={attempts}, completed={completed}, skipped={skipped}"
            )
        grad_stats = diagnostic.get("grad_stats")
        if not isinstance(grad_stats, dict):
            raise WaveError(f"{run_id} lacks FP32 gradient-finiteness evidence")
        if grad_stats.get("steps") != attempts or grad_stats.get("nonfinite_steps") != 0:
            raise WaveError(f"{run_id} has incomplete or nonfinite pre-clip gradient observations")
        finite_evidence = diagnostic.get("finite_update_evidence")
        if not isinstance(finite_evidence, dict) or any(
            finite_evidence.get(key) is not True
            for key in (
                "loss_finite",
                "gradient_finite",
                "counts_consistent",
                "valid_for_throughput",
            )
        ):
            raise WaveError(f"{run_id} lacks successful finite-loss/update evidence")
        resource = diagnostic.get("resource_usage")
        expected_gpu_bytes = gpu_memory_by_arm.get(arm["arm"])
        if (
            not isinstance(resource, dict)
            or expected_gpu_bytes is None
            or resource.get("cuda_total_bytes") != expected_gpu_bytes
            or not all(
                isinstance(resource.get(key), int) and resource.get(key) > 0
                for key in (
                    "cuda_peak_allocated_bytes",
                    "cuda_peak_reserved_bytes",
                    "cuda_total_bytes",
                    "parent_peak_rss_bytes",
                )
            )
            or resource["cuda_peak_allocated_bytes"] > resource["cuda_peak_reserved_bytes"]
            or resource["cuda_peak_reserved_bytes"] > 0.8 * expected_gpu_bytes
        ):
            raise WaveError(
                f"{run_id} resource report is missing, mismatched, or exceeds the 80% GPU-memory gate"
            )
        order = diagnostic.get("sample_order_identity")
        expected_seeded_order = {
            "seed": common["seed"],
            "epoch": 0,
            "antithetic": False,
            "shared_dropped_tail": True,
        }
        order_rows = expected_batches * common["batch"]
        ordered_samples = order.get("ordered_sample_indices") if isinstance(order, dict) else None
        if (
            not isinstance(order, dict)
            or order.get("seeded_order") != expected_seeded_order
            or order.get("batch_size") != common["batch"]
            or order.get("drop_last") is not True
            or not isinstance(ordered_samples, dict)
            or ordered_samples.get("count") != order_rows
            or not isinstance(ordered_samples.get("ordered_int64_le_sha256"), str)
            or len(ordered_samples["ordered_int64_le_sha256"]) != 64
        ):
            raise WaveError(f"{run_id} does not prove the shared seeded epoch-0 sample order")
        comparable_order = {
            "seeded_order": order["seeded_order"],
            "batch_size": order["batch_size"],
            "drop_last": order["drop_last"],
            "ordered_sample_indices": ordered_samples,
        }
        identity = {
            "fit_membership_identity": provenance["fit_membership_identity"],
            "normalization_identity": provenance["normalization_identity"],
            "sample_order_identity": comparable_order,
            "optimizer_counts": (attempts, completed, batches),
            "grad_stats": grad_stats,
            "resource_usage": resource,
        }
        if reference is None:
            reference = identity
            common_updates = (attempts, completed, batches)
        else:
            for key in ("fit_membership_identity", "normalization_identity", "sample_order_identity"):
                if identity[key] != reference[key]:
                    raise WaveError(f"{run_id} {key} differs from the common reference")
            if identity["optimizer_counts"] != common_updates:
                raise WaveError(f"{run_id} optimizer attempts/completions differ across arms")
            if grad_stats["steps"] != reference["grad_stats"]["steps"]:
                raise WaveError(f"{run_id} finite-gradient observation count differs across arms")
        results[run_id] = identity
    return results


def gpu_memory_limits_by_arm(
    manifest: dict[str, Any], recipe: dict[str, Any]
) -> dict[str, int]:
    """Bind each arm to the CUDA capacity probed for its selected physical GPU."""
    gpu_ids = manifest.get("gpu_ids")
    devices = manifest.get("gpu_devices")
    if not isinstance(gpu_ids, list) or not gpu_ids or not isinstance(devices, list):
        raise WaveError("manifest does not bind GPU ids to CUDA memory probes")
    if len(gpu_ids) != len(devices):
        raise WaveError("manifest GPU id and device inventories differ in length")
    capacities = []
    for device in devices:
        capacity = device.get("cuda_probe", {}).get("memory_bytes") if isinstance(device, dict) else None
        if not isinstance(capacity, int) or capacity <= 0:
            raise WaveError("manifest has an invalid CUDA device memory capacity")
        capacities.append(capacity)
    return {
        arm["arm"]: capacities[index % len(capacities)]
        for index, arm in enumerate(recipe["arms"])
    }


def validate_gpu_junit(junit_path: Path, expected_counts: dict[str, Any] | None = None) -> dict[str, int]:
    """Require executed, unskipped cases from each declared GPU test module."""
    try:
        root = ET.parse(junit_path).getroot()
        cases = root.findall(".//testcase")
    except (OSError, ET.ParseError) as error:
        raise WaveError(f"cannot validate CUDA JUnit report: {error}") from error
    skipped = sum(case.find("skipped") is not None for case in cases)
    failures = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    executed_modules = "\n".join(
        f"{case.attrib.get('classname', '')} {case.attrib.get('name', '')}"
        for case in cases
    )
    missing = [
        Path(test).stem
        for test in REQUIRED_GPU_TESTS
        if Path(test).stem not in executed_modules
    ]
    counts = {
        "test_cases": len(cases),
        "skipped": skipped,
        "failures": failures,
        "errors": errors,
    }
    if not cases or skipped or failures or errors or missing:
        raise WaveError(
            f"CUDA tests did not all execute cleanly: cases={len(cases)} skipped={skipped} "
            f"failures={failures} errors={errors} missing_modules={missing}"
        )
    if expected_counts is not None and any(
        expected_counts.get(key) != value for key, value in counts.items()
    ):
        raise WaveError("CUDA JUnit results differ from the saved preflight report")
    return counts


def verify_exports(arm_ids: list[str], data_root: Path, recipe: dict[str, Any], split_path: Path) -> None:
    """Require fixed epoch-0 checkpoints and full-valid feature-only exports."""
    import numpy as np

    with np.load(data_root / "cache/valid_targets.npz", allow_pickle=False) as targets:
        expected_groups = targets["row_group"].astype("<i8", copy=False).tolist()
    if len(expected_groups) != recipe["split"]["full_valid_groups"]:
        raise WaveError("current target cache differs from the frozen full-valid group count")
    for run_id in arm_ids:
        run_root = data_root / "artifacts/runs" / run_id
        provenance = read_json(run_root / "provenance.json", f"{run_id} provenance")
        checkpoint = run_root / "model_ep0.pt"
        checkpoint_sha = sha256_file(checkpoint)
        if provenance.get("checkpoint_sha256", {}).get("model_ep0.pt") != checkpoint_sha:
            raise WaveError(f"{run_id} epoch-0 checkpoint hash is not bound in provenance")
        sidecar_path = data_root / "cache" / f"pred_{run_id}_valid.json"
        sidecar = read_json(sidecar_path, f"{run_id} full-valid prediction sidecar")
        if (
            sidecar.get("run") != run_id
            or sidecar.get("split") != "valid"
            or sidecar.get("epoch") != 0
            or sidecar.get("checkpoint_sha256") != checkpoint_sha
            or sidecar.get("campaign_split_manifest_sha256") != sha256_file(split_path)
            or sidecar.get("campaign_export_scope") != "full-valid-feature-only"
            or sidecar.get("feature_only") is not True
            or sidecar.get("inference_precision") != "fp32"
            or sidecar.get("row_groups") != expected_groups
        ):
            raise WaveError(f"{run_id} prediction export does not prove the full 1873-group FP32 panel")
        prediction = data_root / "cache" / f"pred_{run_id}_valid.npy"
        if sidecar.get("prediction_sha256") != sha256_file(prediction):
            raise WaveError(f"{run_id} prediction sidecar hash does not match its array")
        array = np.load(prediction, mmap_mode="r", allow_pickle=False)
        if array.shape != (
            recipe["split"]["full_valid_groups"],
            recipe["common"]["sequence_rows"],
            2,
        ):
            raise WaveError(f"{run_id} prediction array has an unexpected full-valid shape")


def verify_grade_record(record: dict[str, Any]) -> None:
    """Require worker receipt success and the declared ledger event after grading."""
    queue = Path(record["queue"])
    result = latest_worker_result(queue, record["run_id"])
    if result is None or result.get("status") != "succeeded":
        raise WaveError(f"grade queue did not succeed: {record['run_id']}")
    spec = read_json(queue / "done" / f"{record['run_id']}.json", "completed grade spec")
    comparison = spec.get("comparison")
    if not isinstance(comparison, dict):
        return
    receipt = read_json(Path(comparison["receipt_path"]), "comparison receipt")
    candidate_binding = receipt.get("candidate")
    if (
        receipt.get("status") != "valid"
        or receipt.get("experiment_id") != comparison.get("experiment_id")
        or not isinstance(candidate_binding, dict)
        or candidate_binding.get("identity") != comparison.get("candidate")
    ):
        raise WaveError(f"grade receipt is not valid for {record['run_id']}")
    ledger_path = Path(comparison["ledger_path"])
    try:
        ledger_rows = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WaveError(f"comparison ledger is missing or invalid for {record['run_id']}: {error}") from error
    if not any(
        row.get("experiment_id") == comparison.get("experiment_id")
        and row.get("status") == "valid"
        and row.get("candidate") == comparison.get("candidate")
        for row in ledger_rows
        if isinstance(row, dict)
    ):
        raise WaveError(f"comparison ledger lacks the exact valid event for {record['run_id']}")
    result_receipt = result.get("comparison_receipt")
    if not isinstance(result_receipt, dict) or result_receipt.get("status") != "valid":
        raise WaveError(f"queue worker did not validate a comparison receipt for {record['run_id']}")


def run_preflight(
    manifest: dict[str, Any], recipe: dict[str, Any], python: Path
) -> None:
    """Run strict CUDA test suite first, then one real-data batch per arm."""
    project_root = Path(manifest["project_root"])
    output_dir = Path(manifest["output_dir"])
    if tuple(recipe["target_gpu_preflight"]["required_tests"]) != REQUIRED_GPU_TESTS:
        raise WaveError("recipe GPU-test list differs from the runner's required CUDA test set")
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    junit_path = reports / f"gpu_tests_{stamp}.xml"
    log_path = reports / f"gpu_tests_{stamp}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = manifest["gpu_ids"][0]
    env["WNN_PROJECT_ROOT"] = manifest["data_root"]
    env["PYTHONPATH"] = str(project_root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    if manifest.get("portable_data_manifest"):
        env["WNN_PORTABLE_DATA_MANIFEST"] = manifest["portable_data_manifest"]
    else:
        env.pop("WNN_PORTABLE_DATA_MANIFEST", None)
    argv = [
        str(python),
        "-m",
        "pytest",
        "-q",
        "--junitxml",
        str(junit_path),
        *[str(project_root / test) for test in REQUIRED_GPU_TESTS],
    ]
    try:
        with log_path.open("wb") as log:
            completed = subprocess.run(
                argv,
                cwd=project_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=3600,
            )
    except subprocess.TimeoutExpired as error:
        raise WaveError(f"CUDA GPU tests exceeded the 3600-second limit; inspect {log_path}") from error
    if completed.returncode != 0 or not junit_path.is_file():
        raise WaveError(
            f"CUDA GPU tests failed (exit={completed.returncode}); inspect {log_path}"
        )
    test_counts = validate_gpu_junit(junit_path)
    preflight_records = [
        record for record in manifest["queues"] if record["stage"] == "preflight"
    ]
    if len(preflight_records) != len(recipe["arms"]):
        raise WaveError("prepared packet does not contain exactly one preflight queue per arm")
    assignment = {
        arm["arm"]: manifest["gpu_ids"][index % len(manifest["gpu_ids"])]
        for index, arm in enumerate(recipe["arms"])
    }
    run_worker_pool(preflight_records, assignment, manifest["gpu_ids"], python)
    for record in preflight_records:
        require_done(record)
    preflight_runs = [arm["preflight_run_id"] for arm in recipe["arms"]]
    validate_provenance(
        preflight_runs,
        Path(manifest["data_root"]),
        recipe,
        preflight=True,
        gpu_memory_by_arm=gpu_memory_limits_by_arm(manifest, recipe),
    )
    atomic_json_write(
        reports / "preflight_passed.json",
        {
            "status": "passed",
            "created_utc": datetime.now(UTC).isoformat(),
            "gpu_tests": {
                "junit_xml": str(junit_path),
                "stdout_stderr": str(log_path),
                **test_counts,
                "junit_sha256": sha256_file(junit_path),
            },
            "preflight_runs": preflight_runs,
            "source_closure_sha256": manifest["source_closure_sha256"],
            "recipe_sha256": manifest["recipe_sha256"],
            "split_manifest_sha256": manifest["split_manifest_sha256"],
        },
    )


def run_fit(manifest: dict[str, Any], recipe: dict[str, Any], python: Path) -> None:
    """Fit and export all four arms, then require matched provenance and full panels."""
    fit_records = [record for record in manifest["queues"] if record["stage"] == "fit"]
    assignment = {
        arm["arm"]: manifest["gpu_ids"][index % len(manifest["gpu_ids"])]
        for index, arm in enumerate(recipe["arms"])
    }
    run_worker_pool(fit_records, assignment, manifest["gpu_ids"], python)
    data_root = Path(manifest["data_root"])
    run_ids = [arm["run_id"] for arm in recipe["arms"]]
    provenance = validate_provenance(
        run_ids,
        data_root,
        recipe,
        preflight=False,
        gpu_memory_by_arm=gpu_memory_limits_by_arm(manifest, recipe),
    )
    verify_exports(
        run_ids, data_root, recipe, Path(manifest["split_manifest"]).resolve()
    )
    atomic_json_write(
        Path(manifest["output_dir"]) / "reports" / "fit_passed.json",
        {
            "status": "passed",
            "created_utc": datetime.now(UTC).isoformat(),
            "run_ids": run_ids,
            "matched_provenance": provenance,
            "full_valid_group_count": recipe["split"]["full_valid_groups"],
            "scored_group_count": recipe["split"]["evaluation_groups"],
            "source_closure_sha256": manifest["source_closure_sha256"],
            "recipe_sha256": manifest["recipe_sha256"],
            "split_manifest_sha256": manifest["split_manifest_sha256"],
        },
    )


def run_grade(manifest: dict[str, Any], recipe: dict[str, Any], python: Path) -> None:
    """Grade only after every fit/export and the standalone GRU reference exist."""
    data_root = Path(manifest["data_root"])
    run_ids = [arm["run_id"] for arm in recipe["arms"]]
    fit_records = [record for record in manifest["queues"] if record["stage"] == "fit"]
    if len(fit_records) != len(recipe["arms"]):
        raise WaveError("prepared packet does not contain exactly one fit queue per arm")
    for record in fit_records:
        require_done(record)
    validate_provenance(
        run_ids,
        data_root,
        recipe,
        preflight=False,
        gpu_memory_by_arm=gpu_memory_limits_by_arm(manifest, recipe),
    )
    verify_exports(
        run_ids, data_root, recipe, Path(manifest["split_manifest"]).resolve()
    )
    records = manifest["queues"]
    reference = next(record for record in records if record["stage"] == "grade_reference")
    run_worker(reference, python)
    grade_dir = Path(manifest["output_dir"]) / "grades"
    baseline_path = grade_dir / f"{recipe['comparison']['reference_run_id']}_standalone_grade.json"
    baseline = read_json(baseline_path, "standalone GRU grade")
    if (
        baseline.get("candidate") != f"{recipe['comparison']['reference_run_id']}@ep0"
        or baseline.get("split_manifest_sha256") != recipe["split"]["sha256"]
        or baseline.get("evaluation_groups") != recipe["split"]["evaluation_groups"]
    ):
        raise WaveError("standalone GRU grade does not match the fixed fold-0 contract")
    for record in records:
        if record["stage"] not in {"grade_candidate", "grade_mix"}:
            continue
        run_worker(record, python)
        verify_grade_record(record)
    atomic_json_write(
        Path(manifest["output_dir"]) / "reports" / "grade_passed.json",
        {
            "status": "passed",
            "created_utc": datetime.now(UTC).isoformat(),
            "reference_grade": str(baseline_path),
            "comparison_count": sum(
                record["stage"] in {"grade_candidate", "grade_mix"} for record in records
            ),
            "ledger": str(grade_dir / "comparison_ledger.jsonl"),
        },
    )


def verify_preflight_passed(manifest: dict[str, Any], recipe: dict[str, Any]) -> None:
    """Revalidate CUDA-test receipts and all four real-data preflight runs."""
    output_dir = Path(manifest["output_dir"])
    report = read_json(output_dir / "reports/preflight_passed.json", "GPU preflight report")
    expected_runs = [arm["preflight_run_id"] for arm in recipe["arms"]]
    if (
        report.get("status") != "passed"
        or report.get("preflight_runs") != expected_runs
        or report.get("source_closure_sha256") != manifest["source_closure_sha256"]
        or report.get("recipe_sha256") != manifest["recipe_sha256"]
        or report.get("split_manifest_sha256") != manifest["split_manifest_sha256"]
    ):
        raise WaveError("GPU preflight report is stale or does not match this frozen packet")
    test_report = report.get("gpu_tests")
    if not isinstance(test_report, dict):
        raise WaveError("GPU preflight report lacks its required CUDA test evidence")
    junit_path = Path(test_report.get("junit_xml", ""))
    if (
        not junit_path.is_file()
        or sha256_file(junit_path) != test_report.get("junit_sha256")
    ):
        raise WaveError("CUDA JUnit file is missing or changed since preflight")
    try:
        validate_gpu_junit(junit_path, test_report)
    except WaveError as error:
        raise WaveError(f"CUDA JUnit evidence no longer satisfies the preflight gate: {error}") from error
    records = [record for record in manifest["queues"] if record["stage"] == "preflight"]
    if len(records) != len(recipe["arms"]):
        raise WaveError("GPU preflight queue does not contain exactly four architecture probes")
    for record in records:
        require_done(record)
    validate_provenance(
        expected_runs,
        Path(manifest["data_root"]),
        recipe,
        preflight=True,
        gpu_memory_by_arm=gpu_memory_limits_by_arm(manifest, recipe),
    )


def verify_fit_passed(manifest: dict[str, Any], recipe: dict[str, Any]) -> None:
    """Revalidate the four full-fit queue records before any grading begins."""
    output_dir = Path(manifest["output_dir"])
    report = read_json(output_dir / "reports/fit_passed.json", "fit completion report")
    expected_runs = [arm["run_id"] for arm in recipe["arms"]]
    if (
        report.get("status") != "passed"
        or report.get("run_ids") != expected_runs
        or report.get("full_valid_group_count") != recipe["split"]["full_valid_groups"]
        or report.get("scored_group_count") != recipe["split"]["evaluation_groups"]
        or report.get("source_closure_sha256") != manifest["source_closure_sha256"]
        or report.get("recipe_sha256") != manifest["recipe_sha256"]
        or report.get("split_manifest_sha256") != manifest["split_manifest_sha256"]
    ):
        raise WaveError("fit report is stale or does not match all four declared fits")
    records = [record for record in manifest["queues"] if record["stage"] == "fit"]
    if len(records) != len(recipe["arms"]):
        raise WaveError("fit queue does not contain exactly four architecture arms")
    for record in records:
        require_done(record)
    validate_provenance(
        expected_runs,
        Path(manifest["data_root"]),
        recipe,
        preflight=False,
        gpu_memory_by_arm=gpu_memory_limits_by_arm(manifest, recipe),
    )
    verify_exports(
        expected_runs,
        Path(manifest["data_root"]),
        recipe,
        Path(manifest["split_manifest"]),
    )


def parse_args() -> argparse.Namespace:
    """Parse the target-GPU stage to run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "fit", "grade", "all"), required=True)
    return parser.parse_args()


def main() -> int:
    """Run requested stages fail-closed and persist a compact stage report."""
    args = parse_args()
    try:
        manifest, recipe, _ = verify_packet(args.manifest)
        if sys.platform != "linux":
            raise WaveError("this runner is only for the prepared Linux GPU host")
        python = Path(manifest["python"])
        if not python.exists():
            raise WaveError(f"prepared target Python executable is missing: {python}")
        requested = ("preflight", "fit", "grade") if args.stage == "all" else (args.stage,)
        for stage in requested:
            try:
                if stage == "preflight":
                    run_preflight(manifest, recipe, python)
                elif stage == "fit":
                    verify_preflight_passed(manifest, recipe)
                    run_fit(manifest, recipe, python)
                else:
                    verify_fit_passed(manifest, recipe)
                    run_grade(manifest, recipe, python)
            except (WaveError, OSError, ValueError) as error:
                report = {
                    "status": "failed",
                    "stage": stage,
                    "created_utc": datetime.now(UTC).isoformat(),
                    "error": str(error),
                }
                atomic_json_write(
                    Path(manifest["output_dir"]) / "reports" / f"{stage}_failed.json", report
                )
                raise
        print(json.dumps({"status": "passed", "stages": requested}, indent=2))
        return 0
    except (WaveError, PreparationError, OSError, ValueError, KeyError) as error:
        print(
            json.dumps(
                {
                    "status": "HOLD",
                    "error": str(error),
                    "gpu_fitting_started": False,
                    "submission_or_billing_action": False,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
