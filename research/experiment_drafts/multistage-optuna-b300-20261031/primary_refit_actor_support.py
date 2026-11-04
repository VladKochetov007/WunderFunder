"""Bounded file, source, device, and budget checks for the primary-refit actor."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from primary_refit_profile_review_support import uuid as normalize_uuid

SOURCE4_SHA256 = "c520c67ca8d1fb49fc0c452a4606f5ad130ed51e7fc689fdaf9daea2b429dd81"
PRIMARY_SHA256 = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
AUXILIARY_SHA256 = "26b448bad7f60612cea482f4abaf5f184e5a724228eef227d7e78a8155190075"
POLICY_SHA256 = "36f99c3944e4e414528fe215fc6ae671226c5c9bbee6d826e429ce257ccdf44e"
DEADLINE_SHA256 = "eb7cd396e2c5d8fb4bf0c5f9ae9cac483dba614dc0c7a45a3bc61d0edcaf0fab"
REVIEWER_SHA256 = "ae883b8cc61b882f2401b0478db7407da9bdc03546f1644997c242264b9d1ad2"
REVIEW_SUPPORT_SHA256 = "c33bfca9c0b84c6958c4df9e3514d6a04f15e7838c904363146a5d324cb7b656"
INVENTORY_SHA256 = "feffa074b132b3bee01c4854ee4753ec4f555743137a572e7fcb3edf106ae405"
FIT_WALL_SECONDS = 7_200
MAX_PROFILE_ATTEMPTS = 56
MAX_FIT_ATTEMPTS = 48
STAGE4_INITIAL_CAP = 4


def sha256(path: Path) -> str:
    """Hash a file in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash JSON-compatible data using the canonical WNN encoding."""
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path, label: str, limit: int = 32_000_000) -> dict[str, Any]:
    """Read one bounded regular JSON object without following a final symlink."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"{label} must be a bounded regular non-symlink file")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def write_new(path: Path, value: object) -> None:
    """Write one immutable JSON record, refusing to replace an earlier artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_source4(source_root: Path, expected: Mapping[str, object]) -> None:
    """Require the exact frozen Source4 Python inventory used by prepared requests."""
    actual: dict[str, str] = {}
    for path in sorted((source_root / "src").rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Source4 inventory contains an unsafe Python path")
        actual[path.relative_to(source_root).as_posix()] = sha256(path)
    entry = source_root / "scripts/train_sequence.py"
    if entry.is_symlink() or not entry.is_file():
        raise ValueError("Source4 trainer entrypoint is missing or unsafe")
    actual[entry.relative_to(source_root).as_posix()] = sha256(entry)
    if actual != expected.get("files") or canonical_sha256(actual) != SOURCE4_SHA256:
        raise ValueError("Source4 files differ from the frozen profile/refit identity")


def load_device_inventory(
    runtime_root: Path, admission: Mapping[str, object], explicit: Path | None
) -> tuple[Path, dict[str, Any]]:
    """Locate the root-issued B300 inventory and require its frozen byte digest."""
    profile = admission.get("profile_policy")
    if not isinstance(profile, Mapping):
        raise ValueError("ADMISSION_R2 lacks a profile policy")
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for key in ("device_inventory_path", "inventory_path"):
        value = profile.get(key)
        if isinstance(value, str):
            candidates.append(Path(value))
    candidates.extend(
        runtime_root / "receipts" / name
        for name in (
            "B300_DEVICE_INVENTORY_ACTUAL_R2.json",
            "B300_DEVICE_INVENTORY_R2.json",
            "device-inventory-r2.json",
            "device_inventory.json",
        )
    )
    inventory_path = next(
        (path for path in candidates if path.is_file() and not path.is_symlink()), None
    )
    if inventory_path is None:
        raise FileNotFoundError(
            "pass --device-inventory or provide the path referenced by ADMISSION_R2"
        )
    inventory_path = inventory_path.resolve(strict=True)
    inventory = read_json(inventory_path, "B300 device inventory")
    if (
        sha256(inventory_path) != INVENTORY_SHA256
        or profile.get("device_inventory_sha256", INVENTORY_SHA256) != INVENTORY_SHA256
        or inventory.get("schema") != "wnn.b300-device-inventory.v1"
        or inventory.get("status") != "complete"
        or not isinstance(inventory.get("devices"), list)
    ):
        raise ValueError("device inventory differs from the frozen actual B300 inventory")
    return inventory_path, inventory


def resolve_device(
    admission: Mapping[str, object], inventory: Mapping[str, object], gpu_key: str
) -> dict[str, object]:
    """Join one admitted GPU key to its root-inventoried physical identity."""
    profile = admission.get("profile_policy")
    if gpu_key not in {"0", "1", "2", "3"}:
        raise ValueError("initial-wave primary refits are restricted to GPU keys 0 through 3")
    devices = profile.get("devices") if isinstance(profile, Mapping) else None
    if not isinstance(devices, Mapping):
        raise ValueError("ADMISSION_R2 has no device mapping")
    spec = next((value for key, value in devices.items() if str(key) == gpu_key), None)
    if not isinstance(spec, Mapping):
        raise ValueError(f"GPU key {gpu_key} is not admitted by ADMISSION_R2")
    device_rows = inventory.get("devices")
    if not isinstance(device_rows, list):
        raise ValueError("root B300 inventory has no device list")
    uuid = normalize_uuid(spec.get("uuid"))
    matches = [
        row
        for row in device_rows
        if isinstance(row, dict) and normalize_uuid(row.get("uuid")) == uuid
    ]
    if uuid is None or len(matches) != 1:
        raise ValueError("ADMISSION_R2 GPU UUID is absent or duplicated in its inventory")
    device = dict(matches[0])
    if (
        device.get("profile_allowed") is not True
        or "B300" not in str(device.get("name", "")).upper()
        or device.get("gpu_index") != spec.get("gpu_index")
        or device.get("total_memory_bytes") != spec.get("total_memory_bytes")
        or normalize_uuid(spec.get("cuda_visible_devices")) != uuid
    ):
        raise ValueError("admitted GPU identity differs from the root device inventory")
    device["cuda_visible_devices"] = str(spec["cuda_visible_devices"])
    return device


def profile_environment(
    request: Mapping[str, object], device: Mapping[str, object]
) -> dict[str, str]:
    """Materialize the exact four-thread strict-FP32 profile environment."""
    profile = request.get("profile")
    template = profile.get("environment_template") if isinstance(profile, Mapping) else None
    if not isinstance(template, Mapping):
        raise ValueError("prepared profile request has no environment template")
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in template.items()})
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(device["cuda_visible_devices"]),
            "CUBLAS_EMULATE_SINGLE_PRECISION": "0",
        }
    )
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(key, None)
    required = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    if any(env.get(key) != value for key, value in required.items()):
        raise ValueError("prepared profile thread limits differ from the frozen four-thread policy")
    return env


def reserve_profile_attempt(
    run_dir: Path,
    fit_id: str,
    trial_number: int,
    gpu_key: str,
    device: Mapping[str, object],
    deadline: Mapping[str, object],
) -> dict[str, object]:
    """Atomically consume one of the four Stage4 profile slots and global cap56."""
    from wnn.multistage_search import atomic_write_json, file_lock
    from wnn.multistage_search_budget import utc_now

    with file_lock(run_dir / "budget.lock"):
        current = read_json(run_dir / "budget.json", "fit/profile budget")
        profiles, fits = current.get("profile_attempts"), current.get("fit_attempts")
        reserve_start = deadline.get("reserve_start_epoch")
        now = time.time()
        if (
            current.get("max_profile_attempts") != MAX_PROFILE_ATTEMPTS
            or current.get("max_fit_attempts") != MAX_FIT_ATTEMPTS
            or not isinstance(profiles, list)
            or not isinstance(fits, list)
            or len(profiles) >= MAX_PROFILE_ATTEMPTS
            or len(fits) >= MAX_FIT_ATTEMPTS
            or type(reserve_start) not in (int, float)
            or now + 180 > reserve_start
        ):
            raise RuntimeError("global profile cap or campaign collection reserve blocks profiling")
        stage4_profiles = [
            row for row in profiles if isinstance(row, dict) and row.get("stage") == 4
        ]
        if len(stage4_profiles) >= STAGE4_INITIAL_CAP:
            raise RuntimeError("four initial-wave Stage4 profile attempts are already consumed")
        if any(row.get("fit_id") == fit_id for row in stage4_profiles):
            raise ValueError("this prepared fit already has a Stage4 profile attempt")
        if any(
            row.get("status") in {"reserved", "running"} and row.get("gpu_key") == gpu_key
            for row in stage4_profiles
        ):
            raise RuntimeError("another live Stage4 profile owns this GPU key")
        profile_id = f"stage4-primary-refit-{fit_id}"
        if any(isinstance(row, dict) and row.get("profile_id") == profile_id for row in profiles):
            raise ValueError("Stage4 profile attempt id already exists")
        record = {
            "profile_id": profile_id,
            "fit_id": fit_id,
            "stage": 4,
            "trial_number": trial_number,
            "gpu_key": gpu_key,
            "device": dict(device),
            "status": "reserved",
            "reserved_at_utc": utc_now(),
            "reserved_at_epoch": now,
            "wall_limit_seconds": 180,
            "profile_attempt_count_at_reservation": len(profiles) + 1,
            "fit_attempt_count_at_reservation": len(fits),
            "max_profile_attempts": MAX_PROFILE_ATTEMPTS,
            "max_fit_attempts": MAX_FIT_ATTEMPTS,
        }
        profiles.append(record)
        atomic_write_json(run_dir / "budget.json", current)
        return record


def finish_profile_attempt(run_dir: Path, profile_id: str, status: str, exit_code: int) -> None:
    """Persist the terminal result for one consumed Stage4 profile reservation."""
    from wnn.multistage_search import atomic_write_json, file_lock
    from wnn.multistage_search_budget import utc_now

    with file_lock(run_dir / "budget.lock"):
        budget = read_json(run_dir / "budget.json", "fit/profile budget")
        attempts = budget.get("profile_attempts")
        rows = (
            [
                row
                for row in attempts
                if isinstance(row, dict) and row.get("profile_id") == profile_id
            ]
            if isinstance(attempts, list)
            else []
        )
        if len(rows) != 1 or rows[0].get("status") != "reserved":
            raise ValueError("profile reservation is missing, duplicated, or already terminal")
        rows[0].update({"status": status, "exit_code": exit_code, "finished_at_utc": utc_now()})
        atomic_write_json(run_dir / "budget.json", budget)


def snapshot_budget(run_dir: Path, destination: Path) -> str:
    """Copy the exact current global ledger into a fresh durable profile receipt."""
    source = run_dir / "budget.json"
    payload = source.read_bytes()
    with destination.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def reserve_stage4_fit(
    run_dir: Path, attempt_id: str, trial_number: int, gpu_key: str
) -> tuple[dict[str, object], int]:
    """Consume one Stage 4 fit under the same lock as the global ledger."""
    from wnn.multistage_search import atomic_write_json, file_lock
    from wnn.multistage_search_budget import utc_now

    with file_lock(run_dir / "budget.lock"):
        budget = read_json(run_dir / "budget.json", "campaign fit budget")
        deadline = read_json(run_dir / "campaign_deadline.json", "campaign deadline")
        fits, profiles = budget.get("fit_attempts"), budget.get("profile_attempts")
        caps = budget.get("stage_fit_caps")
        reserve_start = deadline.get("reserve_start_epoch")
        if (
            budget.get("max_fit_attempts") != MAX_FIT_ATTEMPTS
            or budget.get("max_profile_attempts") != MAX_PROFILE_ATTEMPTS
            or not isinstance(fits, list)
            or not isinstance(profiles, list)
            or len(fits) >= MAX_FIT_ATTEMPTS
            or len(profiles) > MAX_PROFILE_ATTEMPTS
            or not isinstance(caps, Mapping)
            or type(caps.get("4")) is not int
            or sha256(run_dir / "campaign_deadline.json") != DEADLINE_SHA256
            or type(reserve_start) not in (int, float)
        ):
            raise RuntimeError(
                "global fit/profile ledger or frozen deadline blocks Stage 4 fitting"
            )
        stage_fits = [row for row in fits if isinstance(row, dict) and row.get("stage") == 4]
        if len(stage_fits) >= min(STAGE4_INITIAL_CAP, caps["4"]):
            raise RuntimeError("initial-wave Stage 4 fit cap is exhausted")
        if any(isinstance(row, dict) and row.get("attempt_id") == attempt_id for row in fits):
            raise ValueError("scientific-fit attempt id already exists")
        if any(
            row.get("gpu_key") == gpu_key and row.get("status") in {"reserved", "running"}
            for row in stage_fits
        ):
            raise RuntimeError("another live Stage 4 fit owns this GPU key")
        now = time.time()
        wall = min(FIT_WALL_SECONDS, int(reserve_start - now))
        if wall <= 0:
            raise RuntimeError("scientific-fit admission is closed for collection")
        record = {
            "attempt_id": attempt_id,
            "stage": 4,
            "trial_number": trial_number,
            "gpu_key": gpu_key,
            "status": "reserved",
            "reserved_at_utc": utc_now(),
            "reserved_at_epoch": now,
            "wall_limit_seconds": wall,
        }
        fits.append(record)
        atomic_write_json(run_dir / "budget.json", budget)
        return record, wall
