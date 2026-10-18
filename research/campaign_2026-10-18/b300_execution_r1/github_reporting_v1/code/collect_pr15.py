"""Read-only, independently verified PR15 campaign progress collector."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

EXPERIMENT_ID, EXPECTED_CELLS, EXPECTED_PER_ARM = "PR15", 120, 30
QUEUE_SHA = "8b3bc9b636ce192c6da9ababace1c5a1a59e58ebcbbe3394fe7f2ae940704eb9"
PARENT_SHA = "950f5178652ad81ae62220a124dc0253cf5f80f9b1dfef7250f39aa1d0c9f1c8"
PARENT_FROZEN_SHA = "c18f1877b8579c03d2a0905534f5ecbd13ed56a54306ba66d63b47d892b66510"
SOURCE_COMMIT = "e238fc5c82ac8c4591f3ca3ef87c37b2b73b929e"
SOURCE_SHA = "873662aded939afd64c17c80b684011de023c04a31be5b2c74e0b9d97a18efc0"
ARMS = {
    "baseline": "all",
    "moments": "moments_only",
    "dictionary": "dictionary_only",
    "combined": "none",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _validators(packet: Path) -> tuple[ModuleType, ModuleType]:
    tag = hashlib.sha256(str(packet.resolve()).encode()).hexdigest()[:10]
    ident = _module(packet / "checkpoint_identity.py", f"_pr15_ident_{tag}")
    previous = sys.modules.get("checkpoint_identity")
    sys.modules["checkpoint_identity"] = ident
    try:
        val = _module(packet / "validation.py", f"_pr15_val_{tag}")
        fit = _module(packet / "fit_validation.py", f"_pr15_fit_{tag}")
    finally:
        if previous is None:
            sys.modules.pop("checkpoint_identity", None)
        else:
            sys.modules["checkpoint_identity"] = previous
    return val, fit


def _packet(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    runtime = root / "runtime"
    packet = runtime / "pr15-host-v2"
    manifest_path = packet / "QUEUE_EXECUTION_MANIFEST.json"
    if _sha(manifest_path) != QUEUE_SHA:
        raise ValueError("host-v2 execution manifest is not pinned")
    manifest = _json(manifest_path)
    for relative, expected in manifest["files"].items():
        path = (packet / relative).resolve(strict=True)
        if not path.is_relative_to(runtime.resolve(strict=True)) or _sha(path) != expected:
            raise ValueError(f"host-v2 manifest member differs: {relative}")
    original = runtime / "pr15-host-v1"
    original_manifest_path = original / "EXECUTION_MANIFEST.json"
    if _sha(original_manifest_path) != PARENT_SHA:
        raise ValueError("original host packet differs from pin")
    original_manifest = _json(original_manifest_path)
    if original_manifest.get("files", {}).get("FROZEN.json") != PARENT_FROZEN_SHA:
        raise ValueError("original host packet does not pin the expected frozen profile")
    frozen_path = original / "FROZEN.json"
    if _sha(frozen_path) != PARENT_FROZEN_SHA:
        raise ValueError("original host frozen profile differs from pin")
    frozen = _json(frozen_path)
    for relative, expected in frozen["files"].items():
        path = (original / relative).resolve(strict=True)
        if not path.is_relative_to(runtime.resolve(strict=True)) or _sha(path) != expected:
            raise ValueError(f"original host frozen member differs: {relative}")
    if _sha(packet / "source_manifest.json") != SOURCE_SHA:
        raise ValueError("PR15 source manifest differs from pin")
    return packet, runtime / "pr15-source-e238-v1", manifest


def _panel(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected = {
        (arm, fold, seed)
        for arm in ARMS
        for fold in range(4)
        for seed in range(1001, 1009 if fold < 2 else 1008)
    }
    by_id, seen = {}, set()
    for job in jobs:
        arm, fold, seed, name = job.get("arm"), job.get("fold"), job.get("seed"), job.get("id")
        cell = (arm, fold, seed)
        if (
            cell not in expected
            or cell in seen
            or job.get("control") != ARMS.get(arm)
            or name != f"pr15_b300_v1_{arm}_f{fold}_s{seed}"
        ):
            raise ValueError("panel has a duplicate or drifted scientific cell")
        seen.add(cell)
        by_id[name] = job
    if len(jobs) != 120 or seen != expected or len(by_id) != 120:
        raise ValueError("panel is not the fixed 120-cell CP30 matrix")
    return by_id


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _progress(
    rows: list[dict[str, Any]], ids: set[str]
) -> tuple[int, int, list[str], set[str], set[str]]:
    started, ended, errors = set(), set(), []
    for row in rows:
        kind, job = row.get("event"), row.get("job")
        name = job.get("id") if isinstance(job, dict) else row.get("process")
        if kind == "fit_started":
            if name not in ids or name in started:
                errors.append(f"unexpected/duplicate start: {name}")
            else:
                started.add(name)
        elif kind in {"process_finished", "process_stopped"}:
            if name not in started or name in ended:
                errors.append(f"unexpected/duplicate terminal event: {name}")
            else:
                ended.add(name)
            if kind == "process_stopped" or row.get("status") != "completed":
                errors.append(f"process failure {name}: {row.get('error', row.get('reason'))}")
        elif kind == "admission_failed":
            errors.append(f"admission failure {name}: {row.get('error')}")
    return len(started - ended), len(ids - started), errors, started, ended


def _paths(
    root: Path, work: Path, endpoint: Path, score: dict[str, Any]
) -> tuple[list[str], list[str]]:
    receipts = [work / f"{phase}_receipt.json" for phase in ("fit", "export", "score")]
    evidence = [endpoint, work / "worker_request.json", work / "worker_receipt.json"]
    if endpoint.parent.parent.name == "pr15-delivery-recovery-v2":
        evidence += [
            endpoint.parent / "FIT_RECOVERY.json",
            work / "envelope/fit/train_cell_resource_receipt.json",
            work / "logs/fit.log",
        ]
    endpoint_record = _json(endpoint)
    phases = [endpoint_record.get("export"), endpoint_record.get("score_phase")]
    full = endpoint_record.get("full_fit_contract", {})
    if isinstance(full, dict):
        phases.extend(full.get("phase_receipts", {}).values())
    for phase in phases:
        if isinstance(phase, dict):
            for key in ("log", "resource_receipt"):
                if isinstance(phase.get(key), str):
                    evidence.append(Path(phase[key]))
    for receipt in receipts:
        if receipt.is_file():
            evidence.append(receipt)
            phase = _json(receipt).get("phase", {})
            if isinstance(phase, dict) and isinstance(phase.get("log"), str):
                log = Path(phase["log"])
                evidence.append(log if log.is_absolute() else work / log)
    run = work / "artifacts/runs" / work.name
    evidence += [
        run / "history.json",
        run / "provenance.json",
        Path(score["score_path"]),
        Path(score["prediction_metadata_path"]),
    ]
    weights = [run / name for name in ("model_ep2.pt", "model.pt", "training_state.pt")]

    def safe(items: list[Path]) -> list[str]:
        result = set()
        for item in items:
            try:
                path = item.resolve(strict=True)
            except OSError:
                continue
            if path.is_file() and path.is_relative_to(root) and path.suffix != ".npy":
                result.add(str(path))
        return sorted(result)

    return safe(evidence), safe(weights)


def _phase(
    work: Path,
    phase: dict[str, Any],
    expected_command: list[str],
    actor: str | None,
    uuid: str,
    cpus: list[int] | None,
    fitv: ModuleType,
) -> None:
    if (
        phase.get("command") != expected_command
        or phase.get("exit_code") != 0
        or isinstance(phase.get("elapsed_seconds"), bool)
        or not isinstance(phase.get("elapsed_seconds"), (int, float))
        or not math.isfinite(phase["elapsed_seconds"])
        or phase["elapsed_seconds"] < 0
    ):
        raise ValueError("recovered export/score command or phase timing differs")
    checked = fitv._phase_receipt(work, phase, actor=actor, gpu_uuid=uuid)
    if actor is not None:
        resource = _json(Path(checked["resource_receipt"]))
        if resource.get("cpu_affinity") != cpus:
            raise ValueError("recovered export CPU affinity differs from its worker")


def _row(
    job: dict[str, Any],
    endpoint_path: Path,
    full: dict[str, Any],
    score: dict[str, Any],
    digests: dict[str, str | None],
    paths: tuple[list[str], list[str]],
    old: bool,
) -> dict[str, Any]:
    checkpoint = full.get("checkpoint", full)
    metric = score.get("score", {})
    if not isinstance(metric, dict) or isinstance(metric.get("wp"), bool):
        raise ValueError("verified score has no official WP")
    wp = float(metric["wp"])
    if not math.isfinite(wp):
        raise ValueError("official WP is non-finite")
    return {
        "id": job["id"],
        "arm": job["arm"],
        "fold": job["fold"],
        "seed": job["seed"],
        "wp": wp,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA,
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "checkpoint_payload_sha256": checkpoint["checkpoint_identity"].get("payload_sha256"),
        "prediction_sha256": score["prediction_sha256"],
        "score_sha256": score["score_sha256"],
        "prediction_metadata_sha256": score["prediction_metadata_sha256"],
        "receipt_sha256": _sha(endpoint_path),
        "fit_recovery_sha256": digests.get("fit_recovery_sha256"),
        "fit_receipt_sha256": None if old else digests.get("fit_receipt_sha256"),
        "fit_timing_seconds": None if old else full.get("fit_timing_seconds"),
        "fit_timing_note": "original fit timing absent; not reconstructed"
        if old
        else "verified worker receipts",
        "receipt_digests": digests,
        "evidence_paths": paths[0],
        "weights_paths": paths[1],
    }


def _old_cell(
    root: Path,
    packet: Path,
    job: dict[str, Any],
    endpoint_path: Path,
    source: Path,
    val: ModuleType,
    fitv: ModuleType,
    contract: dict[str, Any],
) -> dict[str, Any]:
    endpoint, recovery = _json(endpoint_path), _json(endpoint_path.parent / "FIT_RECOVERY.json")
    recovery_path = endpoint_path.parent / "FIT_RECOVERY.json"
    work = root / "state/pr15-execution-v1/work" / job["id"]
    failure, resource_path = (
        work / "worker_receipt.json",
        work / "envelope/fit/train_cell_resource_receipt.json",
    )
    if (
        endpoint.get("schema") != "tabdata-pr15-recovered-endpoint.v1"
        or endpoint.get("status") != "exported_and_graded"
        or endpoint.get("job") != job
        or endpoint.get("training_not_repeated") is not True
        or endpoint.get("fit_recovery_sha256") != _sha(recovery_path)
    ):
        raise ValueError("recovered endpoint identity/status differs")
    if (
        recovery.get("schema") != "tabdata-pr15-fit-recovery.v1"
        or recovery.get("job") != job
        or recovery.get("training_not_repeated") is not True
        or recovery.get("full_payload_bitwise_identical") is not True
        or recovery.get("missing_original_phase_timing_not_reconstructed") is not True
        or recovery.get("original_failure_receipt_sha256") != _sha(failure)
        or recovery.get("original_resource_receipt_sha256") != _sha(resource_path)
    ):
        raise ValueError("original failure/resource/recovery provenance differs")
    worker, resource = _json(failure), _json(resource_path)
    if any(
        worker.get(key) != value
        for key, value in {
            "status": "failed",
            "name": job["id"],
            "fold": job["fold"],
            "seed": job["seed"],
            "control": job["control"],
        }.items()
    ):
        raise ValueError("failed original worker receipt does not bind the cell")
    if any(
        resource.get(key) != value
        for key, value in {
            "status": "actor_returned",
            "visible_devices": job["gpu_uuid"],
            "cpu_affinity": job["cpus"],
            "allocator_budget_bytes": 6 * 1024**3,
        }.items()
    ):
        raise ValueError("original resource receipt differs")
    split_path = root / f"splits/fold{job['fold']}.json"
    split, split_hash = _json(split_path), _sha(split_path)
    fit = fitv._check_checkpoint(
        work, job["id"], job["fold"], job["seed"], job["control"], split_hash, source, contract
    )
    score = val.validate_prediction_and_score(work, job["id"], split, split_hash)
    independent = fitv._check_export_and_score(
        work,
        job["id"],
        split_hash,
        fit["checkpoint_sha256"],
        split["evaluation"]["valid"],
        fit["optimizer_completed_updates"],
        {"score": score},
    )
    if score != endpoint.get("score") or independent != endpoint.get("independent_endpoint_check"):
        raise ValueError("recovered checkpoint, prediction, or official score differs")
    python = str(root / ".venv/bin/python")
    original_packet = root / "runtime/pr15-host-v1"
    _phase(
        work,
        endpoint.get("export", {}),
        [
            python,
            "-u",
            str(original_packet / "gpu_entrypoint.py"),
            str(original_packet / "export_cell.py"),
            "--request",
            str(work / "worker_request.json"),
        ],
        "export_cell.py",
        job["gpu_uuid"],
        job["cpus"],
        fitv,
    )
    _phase(
        work,
        endpoint.get("score_phase", {}),
        [
            python,
            "-u",
            str(work / "scripts/score_run.py"),
            "--run",
            job["id"],
            "--split",
            str(root / f"splits/fold{job['fold']}.json"),
            "--epoch",
            "2",
        ],
        None,
        job["gpu_uuid"],
        None,
        fitv,
    )
    hashes = recovery.get("artifact_sha256", {})
    run = work / "artifacts/runs" / job["id"]
    expected_artifacts = {
        "history.json": run / "history.json",
        "model.pt": run / "model.pt",
        "model_ep2.pt": run / "model_ep2.pt",
        "provenance.json": run / "provenance.json",
        "training_state.pt": run / "training_state.pt",
    }
    if (
        not isinstance(hashes, dict)
        or set(hashes) != set(expected_artifacts)
        or any(
            not isinstance(hashes.get(name), str) or hashes[name] != _sha(path)
            for name, path in expected_artifacts.items()
        )
    ):
        raise ValueError("recovery map differs from the five immutable training artifacts")
    digests = {
        "endpoint_sha256": _sha(endpoint_path),
        "fit_recovery_sha256": _sha(recovery_path),
        "original_failure_receipt_sha256": _sha(failure),
        "original_resource_receipt_sha256": _sha(resource_path),
        "request_sha256": _sha(work / "worker_request.json"),
        "split_manifest_sha256": split_hash,
        "export_receipt_sha256": None,
        "score_receipt_sha256": None,
        "fit_receipt_sha256": None,
    }
    return _row(
        job, endpoint_path, fit, score, digests, _paths(root, work, endpoint_path, score), old=True
    )


def _new_cell(
    root: Path, job: dict[str, Any], endpoint_path: Path, val: ModuleType, fitv: ModuleType
) -> dict[str, Any]:
    endpoint, work = _json(endpoint_path), root / "state/pr15-execution-v2/work" / job["id"]
    worker_path = work / "worker_receipt.json"
    worker = _json(worker_path)
    required = {
        "schema": "pr15-host-worker-receipt.v1",
        "status": "fit_export_grade_completed",
        "mode": "fit",
        "name": job["id"],
        "fold": job["fold"],
        "seed": job["seed"],
        "control": job["control"],
        "cpu_affinity": job["cpus"],
    }
    if (
        endpoint.get("job") != job
        or any(worker.get(k) != v for k, v in required.items())
        or worker.get("gpu", {}).get("uuid") != job["gpu_uuid"]
        or worker.get("source", {}).get("source_commit") != SOURCE_COMMIT
        or worker.get("source", {}).get("manifest_sha256") != SOURCE_SHA
        or endpoint.get("worker_receipt_sha256") != _sha(worker_path)
    ):
        raise ValueError("successor endpoint/worker identity differs")
    split_path = root / f"splits/fold{job['fold']}.json"
    split_hash = _sha(split_path)
    full = fitv.validate_full_fit(
        work, job["id"], job["fold"], job["seed"], job["control"], split_hash
    )
    fit = val.validate_fit_artifacts(work, job["id"])
    score = val.validate_prediction_and_score(work, job["id"], _json(split_path), split_hash)
    if (
        full != endpoint.get("full_fit_contract")
        or fit != endpoint.get("fit")
        or score != endpoint.get("score")
    ):
        raise ValueError("successor endpoint differs from independently verified artifacts")
    fit_phase = _json(work / "fit_receipt.json").get("phase", {})
    elapsed = fit_phase.get("elapsed_seconds") if isinstance(fit_phase, dict) else None
    if (
        isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and math.isfinite(elapsed)
    ):
        full = dict(full, fit_timing_seconds=float(elapsed))
    digests = {
        "endpoint_sha256": _sha(endpoint_path),
        "worker_receipt_sha256": _sha(worker_path),
        "request_sha256": _sha(work / "worker_request.json"),
        "split_manifest_sha256": split_hash,
        "fit_receipt_sha256": _sha(work / "fit_receipt.json"),
        "export_receipt_sha256": _sha(work / "export_receipt.json"),
        "score_receipt_sha256": _sha(work / "score_receipt.json"),
    }
    return _row(
        job,
        endpoint_path,
        full,
        score,
        digests,
        _paths(root, work, endpoint_path, score),
        old=False,
    )


def _collect(root: Path) -> dict[str, Any]:
    root, completed, errors, details = Path(root).resolve(strict=True), [], [], {}
    active = pending = 0
    source_info = {"commit": SOURCE_COMMIT, "manifest_sha256": SOURCE_SHA}
    try:
        packet, source, manifest = _packet(root)
        old, new = root / "state/pr15-execution-v1", root / "state/pr15-execution-v2"
        old_panel_path = old / "PANEL.json"
        if _sha(old_panel_path) != manifest["original_panel_sha256"]:
            raise ValueError("original panel differs from successor manifest")
        panel = _json(old_panel_path)
        jobs = _panel(panel["jobs"])
        new_panel = _json(new / "PANEL.json")
        if new_panel.get("jobs") != panel["jobs"]:
            raise ValueError("successor panel changes original cells")
        old_events = _events(old / "events.jsonl")
        old_starts = [row["job"] for row in old_events if row.get("event") == "fit_started"]
        started_old = {job["id"] for job in old_starts}
        successor = new_panel.get("successor_jobs", [])
        by_id = {job.get("id"): job for job in successor}
        if (
            len(started_old) != 16
            or len(old_starts) != 16
            or not started_old.issubset(jobs)
            or any(job != jobs.get(job.get("id")) for job in old_starts)
            or len(by_id) != 104
            or set(by_id) != set(jobs) - started_old
            or new_panel.get("already_completed_original_fits") != 16
            or any(by_id[name] != jobs[name] for name in by_id)
        ):
            raise ValueError("original/successor split is not the frozen 16+104 roster")
        val, fitv = _validators(packet)
        contract = fitv._check_contract()
        source_info.update(val.verify_source_tree(source, packet / "source_manifest.json"))
        for name in sorted(started_old):
            path = root / "state/pr15-delivery-recovery-v2" / name / "ENDPOINT.json"
            try:
                completed.append(
                    _old_cell(root, packet, jobs[name], path, source, val, fitv, contract)
                )
            except Exception as error:
                errors.append(f"{name}: {type(error).__name__}: {error}")
        active, pending, event_errors, started_new, ended_new = _progress(
            _events(new / "events.jsonl"), set(by_id)
        )
        errors.extend(event_errors)
        for name, job in by_id.items():
            path = new / "endpoints" / f"{name}.json"
            if path.is_file():
                if name not in started_new:
                    errors.append(f"{name}: endpoint exists without a recorded fit start")
                    continue
                try:
                    completed.append(_new_cell(root, job, path, val, fitv))
                except Exception as error:
                    errors.append(f"{name}: {type(error).__name__}: {error}")
            elif name in ended_new:
                errors.append(f"{name}: fit process ended without a verified endpoint")
        details = {
            "queue_manifest_sha256": QUEUE_SHA,
            "started_original": len(started_old),
            "started_successor": len(started_new),
            "verified_original": sum(x["id"] in started_old for x in completed),
            "verified_successor": sum(x["id"] in by_id for x in completed),
        }
    except Exception as error:
        errors.append(f"collection: {type(error).__name__}: {error}")
    completed.sort(key=lambda x: (x["arm"], x["fold"], x["seed"]))
    counts = {arm: sum(row["arm"] == arm for row in completed) for arm in ARMS}
    terminal = (
        len(completed) == 120
        and set(counts.values()) == {30}
        and not active
        and not pending
        and not errors
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "expected_cells": 120,
        "expected_per_arm": 30,
        "completed": completed,
        "completed_per_arm": counts,
        "active": active,
        "pending": pending,
        "errors": errors,
        "terminal": terminal,
        "source": source_info,
        "details": details,
        "contract_note": "Recovered original fits retain the failed worker receipt and actor-returned envelope; "
        "their missing fit receipt/timing is not reconstructed. New fits are independently revalidated.",
    }


def collect(root: Path) -> dict[str, Any]:
    """Read and validate progress without writes, jobs, CUDA initialization, or GitHub calls."""
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return _collect(root)
    finally:
        sys.dont_write_bytecode = bytecode
