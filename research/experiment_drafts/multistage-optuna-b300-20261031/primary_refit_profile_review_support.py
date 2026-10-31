"""Pinned primitives for the Stage 4 primary-refit profile reviewer."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

GIB = 1 << 30
UUID_RE = re.compile(r"^(?:GPU-)?([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$", re.I)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected regular non-symlink file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def uuid(value: object) -> str | None:
    match = UUID_RE.fullmatch(value) if isinstance(value, str) else None
    return match.group(1).lower() if match else None


def process_tuple(value: object) -> bool:
    row = mapping(value)
    return (
        type(row.get("pid")) is int and row["pid"] > 0
        and type(row.get("pgid")) is int and row["pgid"] > 0
        and isinstance(row.get("birth"), (str, int)) and str(row["birth"]) not in ("", "0")
    )


def rank0_ok(path: Path) -> bool:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return False
    if len(rows) != 50:
        return False
    for step, row in enumerate(rows, 1):
        if not isinstance(row, dict) or row.get("rank") != 0 or row.get("step") != step:
            return False
        for key in ("total_loss", "label_loss", "gradient_norm_preclip", "learning_rate"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return False
        if any(type(row.get(key)) is not int or row[key] <= 0 for key in ("valid_rows", "real_rows", "forwarded_rows")):
            return False
        if row["forwarded_rows"] < row["real_rows"] or not isinstance(row.get("label_weight"), (int, float)) or row["label_weight"] <= 0:
            return False
    return True


def argv_value(argv: object, flag: str) -> str | None:
    if not isinstance(argv, list):
        return None
    try:
        index = argv.index(flag)
        return str(argv[index + 1])
    except (ValueError, IndexError):
        return None


def verify_source(root: Path, expected_files: object, expected_sha: str) -> bool:
    if not isinstance(expected_files, dict) or canonical(expected_files) != expected_sha:
        return False
    actual: dict[str, str] = {}
    for path in (root / "src").rglob("*.py"):
        if path.is_symlink() or not path.is_file():
            return False
        actual[path.relative_to(root).as_posix()] = sha(path)
    entry = root / "scripts/train_sequence.py"
    if entry.is_symlink() or not entry.is_file():
        return False
    actual[entry.relative_to(root).as_posix()] = sha(entry)
    return actual == expected_files and canonical(actual) == expected_sha


def check_memory(
    policy: dict[str, Any], limits: dict[str, Any], inventory_path: Path,
    inventory: dict[str, Any], attempt: dict[str, Any], worker: dict[str, Any],
    profile: dict[str, Any], runtime: dict[str, Any], actual_env: dict[str, Any],
) -> bool:
    gpu, nvml = mapping(attempt.get("device")), mapping(attempt.get("nvml_pre_profile"))
    rows, devices = profile.get("memory_by_rank"), runtime.get("devices")
    memory = rows[0] if isinstance(rows, list) and len(rows) == 1 else {}
    native = devices[0] if isinstance(devices, list) and len(devices) == 1 else {}
    records = inventory.get("devices")
    device = next((row for row in records if isinstance(row, dict) and row.get("gpu_index") == gpu.get("gpu_index")), {}) if isinstance(records, list) else {}
    identity = mapping(mapping(policy.get("device_identity")).get("memory_identity"))
    torch_total, nvml_total = identity.get("torch_total_memory_bytes"), identity.get("nvml_total_memory_bytes")
    return (
        sha(inventory_path) == mapping(policy.get("device_identity")).get("inventory_sha256")
        and inventory.get("schema") == "wnn.b300-device-inventory.v1"
        and inventory.get("status") == "complete"
        and all(device.get(key) == gpu.get(key) for key in ("gpu_index", "uuid", "name", "total_memory_bytes"))
        and device.get("profile_allowed") is True
        and "B300" in str(gpu.get("name", "")).upper()
        and uuid(gpu.get("uuid")) is not None
        and uuid(gpu.get("uuid")) == uuid(mapping(native).get("uuid")) == uuid(nvml.get("uuid"))
        and uuid(actual_env.get("CUDA_VISIBLE_DEVICES")) == uuid(gpu.get("uuid"))
        and worker.get("device") == gpu
        and nvml.get("status") == "observed" and nvml.get("total_memory_bytes") == nvml_total
        and worker.get("nvml_pre_profile") == nvml and attempt.get("nvml_pre_profile") == nvml
        and type(nvml.get("free_memory_bytes")) is int and nvml["free_memory_bytes"] >= limits.get("min_nvml_free_bytes", 0)
        and mapping(native).get("total_memory_bytes") == torch_total
        and profile.get("device_total_memory_bytes") == [torch_total]
        and type(memory.get("peak_allocated_bytes")) is int and 0 < memory["peak_allocated_bytes"] <= limits.get("max_allocated_bytes", 0)
        and type(memory.get("peak_reserved_bytes")) is int and memory["peak_allocated_bytes"] <= memory["peak_reserved_bytes"] <= limits.get("max_reserved_bytes", 0)
        and type(memory.get("host_peak_rss_bytes")) is int and 0 < memory["host_peak_rss_bytes"] <= limits.get("max_host_rss_bytes", 0)
    )


def check_owned_worker(
    worker: dict[str, Any], attempt: dict[str, Any], request: dict[str, Any],
    request_sha: str, command: list[str], expected_threads: dict[str, Any],
    environment_template: dict[str, Any], data_root: Path,
) -> bool:
    env = mapping(worker.get("child_environment"))
    owner, child = worker.get("owner_process"), worker.get("child_process")
    command_sha = hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()
    return (
        worker.get("schema") == "wnn.primary-refit-profile-worker.v1"
        and worker.get("status") == "completed" and worker.get("exit_code") == 0
        and worker.get("fit_id") == request.get("fit_id") and worker.get("request_sha256") == request_sha
        and worker.get("command") == command and worker.get("command_sha256") == command_sha
        and process_tuple(owner) and process_tuple(child)
        and attempt.get("owner_process") == owner and attempt.get("child_process") == child
        and worker.get("started_at_epoch") == attempt.get("started_at_epoch")
        and worker.get("finished_at_epoch") == attempt.get("finished_at_epoch")
        and env.get("OMP_NUM_THREADS") == env.get("MKL_NUM_THREADS") == env.get("OPENBLAS_NUM_THREADS") == "4"
        and env.get("CUBLAS_EMULATE_SINGLE_PRECISION") == "0"
        and {key: env.get(key) for key in expected_threads} == expected_threads
        and env.get("WORLD_SIZE", "1") == "1" and env.get("LOCAL_WORLD_SIZE", "1") == "1"
        and all(env.get(key) == value for key, value in environment_template.items() if key != "CUDA_VISIBLE_DEVICES")
        and environment_template.get("CUDA_VISIBLE_DEVICES") == "<independently reviewed GPU UUID>"
        and env.get("WNN_PROJECT_ROOT") == str(data_root)
    )


def check_budget_deadline(
    policy: dict[str, Any], limits: dict[str, Any], attempt: dict[str, Any],
    budget: dict[str, Any], deadline: dict[str, Any], request_sha: str,
    profile_path: Path, profile_sha: str, worker_path: Path, worker_sha: str,
    budget_path: Path, deadline_path: Path, inventory_path: Path, inventory_sha: str,
    rank_path: Path,
) -> bool:
    bounds = mapping(policy.get("campaign_bounds"))
    profiles, fits = budget.get("profile_attempts"), budget.get("fit_attempts")
    return (
        attempt.get("schema") == "wnn.primary-refit-profile-attempt.v1"
        and attempt.get("status") == "completed" and attempt.get("exit_code") == 0
        and attempt.get("request_sha256") == request_sha
        and attempt.get("profile_path") == str(profile_path) and attempt.get("profile_sha256") == profile_sha
        and attempt.get("worker_receipt_path") == str(worker_path) and attempt.get("worker_receipt_sha256") == worker_sha
        and attempt.get("budget_snapshot_path") == str(budget_path) and attempt.get("budget_snapshot_sha256") == sha(budget_path)
        and attempt.get("device_inventory_path") == str(inventory_path) and attempt.get("device_inventory_sha256") == inventory_sha
        and attempt.get("deadline_receipt_path") == str(deadline_path) and attempt.get("deadline_receipt_sha256") == sha(deadline_path)
        and attempt.get("rank0_log_path") == str(rank_path) and attempt.get("rank0_log_sha256") == sha(rank_path)
        and deadline.get("schema") == "wnn.multistage-deadline.v1"
        and deadline.get("campaign_wall_seconds") == bounds.get("campaign_wall_seconds")
        and deadline.get("collection_reserve_seconds") == bounds.get("collection_reserve_seconds")
        and deadline.get("reserve_start_epoch") == bounds.get("reserve_start_epoch")
        and deadline.get("deadline_epoch") == bounds.get("hard_deadline_epoch")
        and type(attempt.get("actual_wall_seconds")) in (int, float)
        and 0 < attempt["actual_wall_seconds"] <= limits.get("max_wall_seconds", 0)
        and abs(attempt["finished_at_epoch"] - attempt["started_at_epoch"] - attempt["actual_wall_seconds"]) <= 1.0
        and attempt.get("finished_at_epoch", bounds.get("hard_deadline_epoch")) <= deadline.get("reserve_start_epoch")
        and budget.get("schema") == "wnn.multistage-budget.v1"
        and budget.get("max_fit_attempts") == bounds.get("max_fit_attempts")
        and budget.get("max_profile_attempts") == bounds.get("max_profile_attempts")
        and isinstance(fits, list) and len(fits) <= bounds.get("max_fit_attempts")
        and isinstance(profiles, list) and len(profiles) <= bounds.get("max_profile_attempts")
        and attempt.get("profile_attempt_count_at_completion") == len(profiles)
        and attempt.get("fit_attempt_count_at_completion") == len(fits)
        and sum(mapping(row).get("stage") == 4 for row in fits) <= bounds.get("stage4_fit_cap_initial_wave")
        and sum(mapping(row).get("stage") == 4 for row in profiles) <= bounds.get("stage4_profile_cap_initial_wave")
        and any(mapping(row).get("fit_id") == attempt.get("fit_id") and mapping(row).get("stage") == 4 for row in profiles)
    )
