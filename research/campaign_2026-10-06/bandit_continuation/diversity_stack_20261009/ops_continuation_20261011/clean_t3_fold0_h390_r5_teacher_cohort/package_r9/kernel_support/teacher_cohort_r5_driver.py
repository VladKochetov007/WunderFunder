"""Measure host state, advise verified input cache, then run unchanged fit admission."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PACKET_RELATIVE = (
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/"
    "ops_continuation_20261011/clean_t3_fold0_h390_r3_teacher_cohort"
)
PROFILE_NAMES = [f"profile_clean_t3_f0_muonall_h384_e3_s{seed}_paired_t4_r5" for seed in (3, 4)]
FIT_NAMES = [f"fit_clean_t3_f0_muonall_h384_e3_s{seed}_r5" for seed in (3, 4, 5)]
EXPORT_NAMES = [
    f"export_clean_t3_f0_muonall_h384_e3_s{seed}_r5_{kind}"
    for seed in (3, 4, 5)
    for kind in ("fit_train", "fit_valid", "full_valid")
]
FINALIZATION_RESERVE_SECONDS = 300
KERNEL_WALL_CEILING_SECONDS = 14_400
MAX_PROFILE_RAM_BYTES = 6_000_000_000
MAX_COMBINED_PROFILE_RAM_BYTES = 12_000_000_000
MAX_PROFILE_GPU_BYTES = 5_000_000_000


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load bundled helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_tree(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return False


def parse_marker(log: Path, marker: str) -> dict:
    matches = [
        line[len(marker) :].strip()
        for line in log.read_text(errors="replace").splitlines()
        if line.startswith(marker)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {marker.strip()} line in {log.name}")
    value = json.loads(matches[0])
    if not isinstance(value, dict) or not finite_tree(value):
        raise RuntimeError(f"non-finite or malformed {marker.strip()} record")
    return value


def load_bootstrap_helpers(root: Path):
    return load_module("r3_kaggle_kernel_helpers", root / "scripts/kaggle_kernel.py")


def load_runner(root: Path):
    sys.path.insert(0, str(root / "scripts"))
    return load_module("r3_kaggle_gpu_runner", root / "scripts/kaggle_gpu_runner.py")


def validate_stage(root: Path, output: Path, names: list[str]) -> dict[str, dict]:
    summary = read_json(output / "summary.json")
    expected = set(names)
    rows = summary.get("finished", [])
    if summary.get("incident") is not None or summary.get("not_started"):
        raise RuntimeError("stage summary has an incident or unstarted workers")
    if {row.get("name") for row in rows} != expected:
        raise RuntimeError("stage summary does not contain exactly the requested workers")
    results = {}
    for name in names:
        row = next(item for item in rows if item.get("name") == name)
        result = read_json(output / name / "result.json")
        if (
            row.get("status") != "complete"
            or row.get("returncode") != 0
            or result.get("status") != "complete"
            or result.get("returncode") != 0
            or result.get("artifacts", {}).get("missing")
        ):
            raise RuntimeError(f"worker did not complete cleanly: {name}")
        declared = {item["path"] for item in result.get("artifacts", {}).get("files", [])}
        spec = read_json(root / "kaggle_specs" / f"{name}.json")
        if not set(spec.get("artifacts", [])).issubset(declared):
            raise RuntimeError(f"worker did not produce all declared artifacts: {name}")
        results[name] = result
    return results


def profile_guard(root: Path, output: Path, preflight: dict) -> dict:
    results = validate_stage(root, output, PROFILE_NAMES)
    devices = []
    workers = []
    for name in PROFILE_NAMES:
        result = results[name]
        device = result.get("device", {})
        if "T4" not in device.get("name", "") or not device.get("uuid"):
            raise RuntimeError(f"paired profile did not use a measured T4: {name}")
        devices.append(device["uuid"])
        ram = result.get("peak_ram_bytes")
        gpu = result.get("peak_gpu_bytes")
        if not isinstance(ram, int) or ram > MAX_PROFILE_RAM_BYTES:
            raise RuntimeError(f"profile RAM guard failed: {name}")
        if not isinstance(gpu, int) or gpu > MAX_PROFILE_GPU_BYTES:
            raise RuntimeError(f"profile GPU-memory guard failed: {name}")
        log = output / name / "worker.log"
        counts = parse_marker(log, "PROFILE_UPDATE_COUNTS ")
        epoch = counts.get("epoch", {})
        attempts = epoch.get("optimizer_attempts")
        completed = epoch.get("optimizer_completed_updates")
        skipped = epoch.get("amp_skipped_updates")
        batches = epoch.get("completed_batches")
        if (
            batches != 10
            or not isinstance(attempts, int)
            or not isinstance(completed, int)
            or not isinstance(skipped, int)
            or attempts <= 0
            or completed <= 0
            or completed + skipped != attempts
        ):
            raise RuntimeError(f"profile update counters failed the sealed guard: {name}")
        grad = parse_marker(log, "GRAD_STATS ")
        if grad.get("steps") != attempts or grad.get("nonfinite_steps", attempts + 1) > skipped:
            raise RuntimeError(
                f"gradient summary is inconsistent with skipped-update counts: {name}"
            )
        profile = parse_marker(log, "PROFILE ")
        if not isinstance(profile.get("step_wall"), (int, float)) or profile["step_wall"] <= 0:
            raise RuntimeError(f"profile batch timing is missing or nonpositive: {name}")
        workers.append(
            {
                "name": name,
                "device_uuid": device["uuid"],
                "peak_ram_bytes": ram,
                "peak_gpu_bytes": gpu,
                "completed_batches": batches,
                "optimizer_attempts": attempts,
                "optimizer_completed_updates": completed,
                "amp_skipped_updates": skipped,
                "gradient_nonfinite_steps": grad["nonfinite_steps"],
                "profile": profile,
            }
        )
    if len(set(devices)) != 2:
        raise RuntimeError("paired profiles were not placed on two distinct physical T4 UUIDs")
    expected_devices = {row["uuid"] for row in preflight["devices"]}
    if set(devices) != expected_devices:
        raise RuntimeError("paired profile devices differ from the preflight inventory")
    combined_ram = sum(row["peak_ram_bytes"] for row in workers)
    if combined_ram > MAX_COMBINED_PROFILE_RAM_BYTES:
        raise RuntimeError("combined per-worker peak RAM exceeded the paired-profile guard")
    host = preflight["host"]
    budget = min(
        int(host["capacity_bytes"] * 0.8),
        host["available_bytes"] - int(host["capacity_bytes"] * 0.2),
    )
    if budget <= 0 or combined_ram > budget:
        raise RuntimeError("observed paired RAM peaks exceed the initial host operational budget")
    receipt = {
        "schema": "wnn.kaggle.paired_t4_profile_guard_receipt.r1",
        "status": "pass",
        "runner_concurrency": 2,
        "profile_workers": workers,
        "distinct_t4_uuids": sorted(set(devices)),
        "sum_worker_peak_ram_bytes": combined_ram,
        "initial_host_budget_bytes": budget,
        "host_inventory": host,
        "device_inventory": preflight["devices"],
        "fit_release": "still requires embedded root conditional admission; this receipt alone cannot authorize fits",
    }
    return receipt


R5_PACKET_RELATIVE = "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_fold0_h390_r5_teacher_cohort"
R5_GATE_RELATIVE = f"{R5_PACKET_RELATIVE}/CONDITIONAL_FIT_ADMISSION_R5.json"
R5_ROOT_RELATIVE = f"{R5_PACKET_RELATIVE}/ROOT_CONDITIONAL_RELEASE_R5.json"
R5_BINDINGS_RELATIVE = f"{R5_PACKET_RELATIVE}/R5_PRESEAL_BINDINGS_R1.json"
R5_TARGET = {
    "dataset": "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5",
    "kernel_id": "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r5-runner",
}
R5_ALLOWED_DELTA = [
    "embed_exact_root_conditional_release_receipt_r5",
    "authorize_only_the_existing_profile_gated_teacher_fit_branch",
    "rebind_bootstrap_constants_for_r5_slug_archive_gate_and_release_receipt",
    "add_host_and_cgroup_memory_snapshots_around_the_existing_fit_admission",
    "advise_dontneed_only_for_hash_verified_declared_raw_and_own_cache_inputs_after_profiles_exit",
    "rebind_worker_attempt_names_and_package_identity_without_changing_science_or_resource_limits",
]


def append_resource_event(output_root: Path, event: dict) -> None:
    path = output_root / "r5_resource_handoff.json"
    document = (
        read_json(path)
        if path.is_file()
        else {
            "schema": "wnn.kaggle.teacher_resource_handoff.r5",
            "status": "running",
            "events": [],
            "science_recipe_changed": False,
            "fit_worker_ram_reservation_bytes": 8_000_000_000,
            "admission_function": "unchanged kaggle_gpu_runner.admit_specs",
        }
    )
    document["events"].append(event)
    write_json(path, document)


def memory_snapshot(label: str, runner, *, cgroup_root: Path = Path("/sys/fs/cgroup")) -> dict:
    import psutil

    host = runner.host_inventory()
    devices = runner.gpu_inventory()
    vm = psutil.virtual_memory()
    cgroup: dict[str, object] = {"version": "v2"}
    for filename in ("memory.current", "memory.max", "memory.high", "memory.peak"):
        path = cgroup_root / filename
        if path.is_file():
            raw = path.read_text().strip()
            cgroup[filename] = raw if raw == "max" else int(raw)
    for filename in ("memory.stat", "memory.events"):
        path = cgroup_root / filename
        if path.is_file():
            fields: dict[str, int] = {}
            for line in path.read_text().splitlines():
                key, raw = line.split(maxsplit=1)
                if raw.strip().isdigit():
                    fields[key] = int(raw)
            cgroup[filename] = fields
    budget = min(
        int(host["capacity_bytes"] * 0.8),
        int(host["available_bytes"]) - int(host["capacity_bytes"] * 0.2),
    )
    return {
        "label": label,
        "captured_utc": datetime.now(UTC).isoformat(),
        "captured_monotonic": time.monotonic(),
        "runner_host_inventory": host,
        "psutil_virtual_memory": {
            "total_bytes": int(vm.total),
            "available_bytes": int(vm.available),
            "used_bytes": int(vm.used),
            "percent": float(vm.percent),
        },
        "cgroup": cgroup,
        "gpu_inventory": devices,
        "runner_operational_host_budget_bytes": budget,
    }


def verify_r5_authority(root: Path) -> dict:
    gate_path = root / R5_GATE_RELATIVE
    release_path = root / R5_ROOT_RELATIVE
    bindings_path = root / R5_BINDINGS_RELATIVE
    gate_digest = sha256_file(gate_path)
    if gate_digest != os.environ.get("WNN_R5_GATE_SHA256"):
        raise PermissionError("R5 conditional gate is not bound by the bootstrap")
    gate = read_json(gate_path)
    if gate.get("status") != "admitted" or gate.get("release_authorized") is not True:
        raise PermissionError("R5 root release is pending")
    if gate.get("root_release_receipt_path") != R5_ROOT_RELATIVE:
        raise PermissionError("R5 gate names an unexpected root receipt")
    release_digest = sha256_file(release_path)
    if release_digest != os.environ.get("WNN_R5_ROOT_RELEASE_SHA256") or release_digest != gate.get(
        "root_release_receipt_sha256"
    ):
        raise PermissionError("R5 root receipt bytes are not bound")
    release = read_json(release_path)
    bindings = read_json(bindings_path)
    if (
        release.get("schema") != "wnn.root_conditional_teacher_cohort_release.r5"
        or release.get("status") != "admitted"
        or release.get("authorized") is not True
        or release.get("binding", {}).get("preseal_r5") != bindings
        or release.get("target") != R5_TARGET
        or release.get("allowed_delta") != R5_ALLOWED_DELTA
    ):
        raise PermissionError("R5 root receipt does not bind the frozen packet")
    quota = release.get("binding", {}).get("quota_budget", {})
    for key, expected in (
        ("wall_ceiling_seconds", 14400),
        ("two_device_gpu_seconds_upper_bound", 28800),
        ("reserved_gpu_seconds", 28800),
    ):
        if type(quota.get(key)) is not int or quota[key] != expected:
            raise PermissionError(f"R5 root quota has invalid {key}")
    if quota.get("paid_spending_authorized") is not False:
        raise PermissionError("R5 root receipt cannot authorize paid spending")
    for key in ("campaign_headroom_gpu_seconds", "provider_headroom_gpu_seconds"):
        value = quota.get(key)
        if type(value) is int:
            valid = value >= 28800
        elif type(value) is float:
            valid = math.isfinite(value) and value >= 28800
        else:
            valid = False
        if not valid:
            raise PermissionError(f"R5 root quota has invalid {key}")
    return {"gate_sha256": gate_digest, "root_release_receipt_sha256": release_digest}


def input_stat_snapshot(root: Path, contract: dict, *, label: str) -> dict:
    """Record mounted-file identities without rereading immutable input contents."""
    input_root = Path("/kaggle/input").resolve(strict=True)
    rows = []
    for item in contract["allowed_paths"]:
        name = item["path"]
        resolved = (root / name).resolve(strict=True)
        if not resolved.is_relative_to(input_root):
            raise RuntimeError(f"declared input is not mounted under /kaggle/input: {name}")
        metadata = resolved.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != item["size"]:
            raise RuntimeError(f"declared input type/size changed: {name}")
        rows.append(
            {
                "path": name,
                "expected_size_bytes": int(item["size"]),
                "expected_sha256": item["sha256"],
                "resolved_path": str(resolved),
                "device": int(metadata.st_dev),
                "inode": int(metadata.st_ino),
                "size_bytes": int(metadata.st_size),
                "mtime_ns": int(metadata.st_mtime_ns),
                "ctime_ns": int(metadata.st_ctime_ns),
                "mode": int(metadata.st_mode),
            }
        )
    return {"schema": "wnn.kaggle.teacher_input_stat_snapshot.r5", "label": label, "files": rows}


def _profile_input_receipts(profile_output: Path, contract: dict) -> list[dict]:
    """Reuse this attempt's actual runner SHA receipts and trainer Parquet hashes."""
    raw_names = {"datasets/train.parquet", "datasets/valid.parquet"}
    expected = {item["path"]: item for item in contract["allowed_paths"]}
    evidence = []
    for seed in (3, 4):
        name = f"profile_clean_t3_f0_muonall_h384_e3_s{seed}_paired_t4_r5"
        identity_path = profile_output / f"{name}.input_identity.json"
        worker_log = profile_output / name / "worker.log"
        identity = read_json(identity_path)
        if identity.get("status") != "verified_internal_deferred_external":
            raise RuntimeError(f"profile input identity did not pass: {name}")
        observations = {row.get("path"): row for row in identity.get("observations", [])}
        for path, item in expected.items():
            row = observations.get(path)
            if not isinstance(row, dict):
                raise RuntimeError(f"profile receipt omits declared input: {name}:{path}")
            if row.get("expected") != {"size": item["size"], "sha256": item["sha256"]}:
                raise RuntimeError(
                    f"profile expected identity differs from R5 binding: {name}:{path}"
                )
            observed = row.get("observed", {})
            if path in raw_names:
                if (
                    row.get("status") != "deferred_to_worker_preflight"
                    or observed.get("size") != item["size"]
                    or observed.get("sha256") is not None
                ):
                    raise RuntimeError(
                        f"raw parquet runner receipt differs from contract: {name}:{path}"
                    )
            elif (
                row.get("status") != "match"
                or observed.get("size") != item["size"]
                or observed.get("sha256") != item["sha256"]
            ):
                raise RuntimeError(f"cache input SHA receipt differs from contract: {name}:{path}")
        completed = []
        for line_number, line in enumerate(worker_log.read_text(errors="replace").splitlines(), 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "portable_sha256_complete":
                completed.append((line_number, row))
        for path in sorted(raw_names):
            item = expected[path]
            matches = [
                (line_number, row)
                for line_number, row in completed
                if row.get("file") == Path(path).name
                and row.get("bytes_total") == item["size"]
                and row.get("bytes_read") == item["size"]
                and row.get("sha256") == item["sha256"]
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"trainer did not verify the exact raw parquet identity: {name}:{path}"
                )
            line_number, _ = matches[0]
            evidence.append(
                {
                    "profile": name,
                    "path": path,
                    "producer": "wnn.portability.verify_portable_source",
                    "worker_log_sha256": sha256_file(worker_log),
                    "line_number": line_number,
                    "bytes_verified": int(item["size"]),
                    "sha256": item["sha256"],
                }
            )
        evidence.append(
            {
                "profile": name,
                "path": "cache_inputs",
                "producer": "scripts/kaggle_gpu_runner.py input_identity.json",
                "identity_receipt_sha256": sha256_file(identity_path),
                "verified_cache_paths": sorted(set(expected) - raw_names),
            }
        )
    return evidence


def release_verified_input_cache(
    root: Path, profile_output: Path, output_root: Path, pre_profile_stats: dict
) -> dict:
    """Use existing R3 producer receipts and unchanged mount identity before cache advice."""
    binding = read_json(root / PACKET_RELATIVE / "INPUT_BINDING_R3.json")
    descriptors = [
        row
        for row in binding["external_inputs"]
        if row["path"].startswith("cache/")
        or row["path"] in {"datasets/train.parquet", "datasets/valid.parquet"}
    ]
    contract = read_json(root / R5_PACKET_RELATIVE / "R5_RESOURCE_HANDOFF_CONTRACT_R1.json")
    expected = {row["path"]: (row["size"], row["sha256"]) for row in contract["allowed_paths"]}
    observed = {row["path"]: (row["size"], row["sha256"]) for row in descriptors}
    if observed != expected:
        raise RuntimeError("R5 input allowlist differs from the sealed contract")
    if pre_profile_stats.get("schema") != "wnn.kaggle.teacher_input_stat_snapshot.r5":
        raise RuntimeError("R5 pre-profile input identity snapshot is missing")
    before_rows = {row["path"]: row for row in pre_profile_stats.get("files", [])}
    if set(before_rows) != set(expected):
        raise RuntimeError("R5 pre-profile input snapshot does not exactly cover the allowlist")
    profile_evidence = _profile_input_receipts(profile_output, contract)
    current = input_stat_snapshot(root, contract, label="before_cache_advice")
    current_rows = {row["path"]: row for row in current["files"]}
    stat_fields = ("resolved_path", "device", "inode", "size_bytes", "mtime_ns", "ctime_ns", "mode")
    for name in sorted(expected):
        if any(
            before_rows[name].get(field) != current_rows[name].get(field) for field in stat_fields
        ):
            raise RuntimeError(
                f"mounted input identity changed since pre-profile verification: {name}"
            )
    result = {
        "schema": "wnn.kaggle.teacher_cache_advice_receipt.r5",
        "status": "running",
        "api": "POSIX_FADV_DONTNEED",
        "identity_method": "bound R3 input manifest + current-attempt Kaggle runner/trainer receipts + unchanged mounted device/inode/size/mtime/ctime",
        "content_rehashed_for_advice": False,
        "advice_semantics": "best effort; a returned call does not prove bytes were released",
        "pre_profile_stat_snapshot": pre_profile_stats,
        "pre_advice_stat_snapshot": current,
        "existing_verified_receipts": profile_evidence,
        "files": [],
    }
    receipt_path = output_root / "r5_cache_advice.json"
    write_json(receipt_path, result)
    advice = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if not callable(advice) or dontneed is None:
        result["status"] = "unsupported"
        result["post_advice_stat_snapshot"] = input_stat_snapshot(
            root, contract, label="after_cache_advice"
        )
        write_json(receipt_path, result)
        return result
    for item in sorted(descriptors, key=lambda row: row["path"]):
        name = item["path"]
        path = root / name
        record = {
            "path": name,
            "expected_size_bytes": int(item["size"]),
            "expected_sha256": item["sha256"],
            "range": {"offset_bytes": 0, "length_bytes": 0, "meaning": "whole file to EOF"},
            "status": "verifying_existing_receipts_and_stat",
        }
        result["files"].append(record)
        write_json(receipt_path, result)
        fd = None
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(Path("/kaggle/input").resolve(strict=True)):
                raise RuntimeError(f"declared input is not mounted under /kaggle/input: {name}")
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
            metadata = os.fstat(fd)
            old = before_rows[name]
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size != item["size"]
                or str(resolved) != old["resolved_path"]
                or metadata.st_dev != old["device"]
                or metadata.st_ino != old["inode"]
                or metadata.st_mtime_ns != old["mtime_ns"]
                or metadata.st_ctime_ns != old["ctime_ns"]
            ):
                raise RuntimeError(f"declared input metadata changed: {name}")
            record.update(
                {
                    "identity_verified": True,
                    "identity_verification": "current-attempt producer receipt plus unchanged mounted file identity",
                    "device": int(metadata.st_dev),
                    "inode": int(metadata.st_ino),
                    "mtime_ns": int(metadata.st_mtime_ns),
                    "ctime_ns": int(metadata.st_ctime_ns),
                }
            )
        except Exception as error:
            record.update(
                {
                    "status": "identity_failure",
                    "error_type": type(error).__name__,
                    "error": str(error)[:500],
                }
            )
            result["status"] = "identity_failure"
            write_json(receipt_path, result)
            if fd is not None:
                os.close(fd)
            raise
        try:
            advice(fd, 0, 0, dontneed)
            record["status"] = "advice_returned"
            record["advice_returned"] = True
        except OSError as error:
            record.update(
                {"status": "advice_error", "errno": error.errno, "error": str(error)[:500]}
            )
            result["status"] = "partial_advice_error"
        finally:
            os.close(fd)
            write_json(receipt_path, result)
    result["attempted_files"] = len(result["files"])
    result["advice_returned_files"] = sum(
        row.get("advice_returned") is True for row in result["files"]
    )
    result["attempted_bytes"] = sum(row["expected_size_bytes"] for row in result["files"])
    after = input_stat_snapshot(root, contract, label="after_cache_advice")
    after_rows = {row["path"]: row for row in after["files"]}
    for name in sorted(expected):
        if any(
            current_rows[name].get(field) != after_rows[name].get(field) for field in stat_fields
        ):
            result["status"] = "input_identity_changed_after_advice"
            result["post_advice_stat_snapshot"] = after
            write_json(receipt_path, result)
            raise RuntimeError(f"mounted input metadata changed during cache advice: {name}")
    result["post_advice_stat_snapshot"] = after
    if result["status"] == "running":
        result["status"] = "complete"
    write_json(receipt_path, result)
    return result


def record_fit_admission(output_root: Path, runner, fit_specs: list[dict]) -> int:
    before = memory_snapshot("pre_fit_admission", runner)
    append_resource_event(output_root, {"event": "pre_fit_admission", "snapshot": before})
    record: dict[str, object] = {
        "schema": "wnn.kaggle.teacher_fit_admission_r5",
        "admission_function": "unchanged kaggle_gpu_runner.admit_specs",
        "status": "running",
        "fit_worker_ram_reservations_bytes": [spec["limits"]["ram_bytes"] for spec in fit_specs],
        "snapshot": before,
    }
    output = output_root / "r5_fit_admission.json"
    write_json(output, record)
    try:
        host, devices = runner.host_inventory(), runner.gpu_inventory()
        record["admission_host_inventory"] = host
        record["admission_device_inventory"] = devices
        write_json(output, record)
        admitted = runner.admit_specs(fit_specs, host, devices)
    except Exception as error:
        record.update(
            {"status": "refused", "error_type": type(error).__name__, "error": str(error)[:1000]}
        )
        record["after_snapshot"] = memory_snapshot("after_fit_admission_refusal", runner)
        write_json(output, record)
        append_resource_event(output_root, {"event": "fit_admission_refused", "receipt": record})
        raise
    record.update(
        {
            "status": "admitted_two" if admitted == 2 else "refused_concurrency",
            "concurrency_admitted": admitted,
        }
    )
    record["after_snapshot"] = memory_snapshot("after_fit_admission", runner)
    write_json(output, record)
    append_resource_event(output_root, {"event": "fit_admission_result", "receipt": record})
    return admitted


def ensure_pair_admission(root: Path, profile_names: list[str], output_root: Path) -> dict:
    runner = load_runner(root)
    helpers = load_bootstrap_helpers(root)
    host, devices = runner.host_inventory(), runner.gpu_inventory()
    if len(devices) != 2 or any("T4" not in d.get("name", "") for d in devices):
        raise RuntimeError("R3 paired profile gate requires exactly two visible Tesla T4 devices")
    specs = [runner.read_spec(root / "kaggle_specs" / f"{name}.json") for name in profile_names]
    admitted = runner.admit_specs(specs, host, devices)
    if admitted != 2:
        raise RuntimeError(
            "runner host/CPU admission would serialize the paired profile; refusing to launch"
        )
    preflight = {"host": host, "devices": devices, "concurrency_admitted": admitted}
    write_json(output_root / "profile_preflight_inventory.json", preflight)
    append_resource_event(
        output_root,
        {
            "event": "before_paired_profiles",
            "snapshot": memory_snapshot("before_paired_profiles", runner),
        },
    )
    return {"runner": runner, "helpers": helpers, "specs": specs, "preflight": preflight}


def run_stage(
    root: Path, output_root: Path, stage: str, names: list[str], deadline: float, helpers
) -> dict:
    output = output_root / stage
    command = [
        sys.executable,
        str(root / "scripts/kaggle_gpu_runner.py"),
        "--root",
        str(root),
        "--output",
        str(output),
    ]
    runtime_specs = []
    for name in names:
        source = root / "kaggle_specs" / f"{name}.json"
        runtime = Path("/kaggle/working/runtime_specs") / source.name
        helpers.write_runtime_spec(source, runtime, deadline)
        runtime_specs.append(runtime)
        command.extend(["--spec", str(runtime)])
    remaining = helpers.remaining_seconds(deadline)
    if remaining <= FINALIZATION_RESERVE_SECONDS:
        raise TimeoutError(f"insufficient kernel wall budget to begin {stage}")
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=root, env=os.environ.copy(), start_new_session=True)
    try:
        returncode = process.wait(timeout=max(1, int(remaining - FINALIZATION_RESERVE_SECONDS)))
    except subprocess.TimeoutExpired as error:
        helpers.terminate_process_tree(process)
        raise TimeoutError(f"{stage} exceeded remaining bounded kernel wall time") from error
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    result = {
        "name": stage,
        "status": "complete",
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started,
        "workers": names,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    packet = root / PACKET_RELATIVE
    verify_r5_authority(root)
    gate = read_json(packet / "CONDITIONAL_FIT_ADMISSION_R3.json")
    if gate.get("status") != "admitted" or gate.get("release_authorized") is not True:
        raise PermissionError("R3 root conditional admission is pending")
    release_sha = gate.get("root_release_receipt_sha256")
    if (
        not isinstance(release_sha, str)
        or len(release_sha) != 64
        or any(character not in "0123456789abcdef" for character in release_sha)
    ):
        raise PermissionError("R3 root release receipt is not bound")
    if gate.get("profile_reference_sha256") != sha256_file(packet / "PROFILE_REFERENCE_R3.json"):
        raise PermissionError("R3 root gate does not bind the actual single-T4 profile reference")
    if gate.get("profile_guard_contract_sha256") != sha256_file(
        packet / "PROFILE_GUARD_CONTRACT_R3.json"
    ):
        raise PermissionError("R3 root gate does not bind the paired-T4 profile guard")
    deadline_value = os.environ.get("WNN_KERNEL_DEADLINE_MONOTONIC")
    if deadline_value is None:
        raise RuntimeError("Kaggle bootstrap did not pass its monotonic deadline")
    deadline = float(deadline_value)
    output_root = Path("/kaggle/working/results")
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline_path = output_root / "teacher_cohort_pipeline.json"
    pipeline = {
        "schema": "wnn.kaggle.teacher_cohort_pipeline.r5",
        "status": "running",
        "stages": [],
    }
    write_json(pipeline_path, pipeline)
    try:
        pair = ensure_pair_admission(root, PROFILE_NAMES, output_root)
        pipeline["paired_profile_preflight"] = pair["preflight"]
        resource_contract = read_json(
            root / R5_PACKET_RELATIVE / "R5_RESOURCE_HANDOFF_CONTRACT_R1.json"
        )
        pre_profile_stats = input_stat_snapshot(
            root, resource_contract, label="before_paired_profiles"
        )
        write_json(output_root / "r5_input_pre_profile_stat.json", pre_profile_stats)
        pipeline["stages"].append(
            run_stage(
                root, output_root, "paired_profiles", PROFILE_NAMES, deadline, pair["helpers"]
            )
        )
        profile_output = output_root / "paired_profiles"
        guard_receipt = profile_guard(root, profile_output, pair["preflight"])
        append_resource_event(
            output_root,
            {
                "event": "after_paired_profiles_before_cache_advice",
                "snapshot": memory_snapshot(
                    "after_paired_profiles_before_cache_advice", pair["runner"]
                ),
            },
        )
        write_json(output_root / "paired_profile_guard_receipt.json", guard_receipt)
        pipeline["paired_profile_guard"] = guard_receipt
        pipeline["status"] = "profile_guard_passed_pending_fit_stage"
        write_json(pipeline_path, pipeline)

        # The embedded root receipt authorizes a conditional continuation. This
        # exact branch runs only after the paired profiles pass every guard.
        if gate.get("profile_guard_contract_sha256") != sha256_file(
            packet / "PROFILE_GUARD_CONTRACT_R3.json"
        ):
            raise PermissionError("root gate does not bind the frozen paired-profile contract")
        runner = pair["runner"]
        fit_specs = [runner.read_spec(root / "kaggle_specs" / f"{name}.json") for name in FIT_NAMES]
        cache_receipt = release_verified_input_cache(
            root, profile_output, output_root, pre_profile_stats
        )
        append_resource_event(
            output_root,
            {
                "event": "after_cache_advice_before_fit_admission",
                "cache_advice_status": cache_receipt["status"],
                "snapshot": memory_snapshot("after_cache_advice_before_fit_admission", runner),
            },
        )
        if record_fit_admission(output_root, runner, fit_specs) != 2:
            raise RuntimeError(
                "fit admission returned concurrency other than two; no full fits were started"
            )
        helpers = pair["helpers"]
        pipeline["stages"].append(
            run_stage(root, output_root, "teacher_fits", FIT_NAMES, deadline, helpers)
        )
        write_json(pipeline_path, pipeline)
        validate_stage(root, output_root / "teacher_fits", FIT_NAMES)
        pipeline["stages"].append(
            run_stage(root, output_root, "teacher_exports", EXPORT_NAMES, deadline, helpers)
        )
        pipeline["status"] = "complete"
        pipeline["completed_elapsed_seconds"] = (
            KERNEL_WALL_CEILING_SECONDS - helpers.remaining_seconds(deadline)
        )
        write_json(pipeline_path, pipeline)
    except BaseException as error:
        pipeline["status"] = "blocked_or_failed"
        pipeline["error_type"] = type(error).__name__
        pipeline["error"] = str(error)[:1000]
        write_json(pipeline_path, pipeline)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
