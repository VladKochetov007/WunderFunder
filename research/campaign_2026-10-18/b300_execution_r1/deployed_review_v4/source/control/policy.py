"""Frozen wave, runtime, profile, and resource gates for the tabdata L4 cohort."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ACTOR_SHA256 = "e6a155ee13a7e907bb7fca2c1ebed07879e8466422015546e20360af9b404220"
WAVE_SHA256 = "003f14169ad8867f2a9d7e86172ce3442edd2c600f0cf25b9fa7d6d31ac6884c"
EXPECTED_JOBS = 8
EXPECTED_CPUS = 16
EXPECTED_RAM = 192 * 1024**3
EXPECTED_PIDS = 512
EXPECTED_TORCH = "2.14.0+cu130"
EXPECTED_CUDA = "13.0"
GPU_FREE_MIN = 8 * 1024
ALLOCATOR_BYTES = 6 * 1024**3
JOB_RSS_LIMIT = 24 * 1024**3
DISK_FREE_MIN = 64 * 1024**3
PROFILE_TIMEOUT_SECONDS = 1200
FIT_SECONDS = 23 * 3600
WAVE_SECONDS = 24 * 3600


def sha256_file(path: Path) -> str:
    """Hash one file without loading its contents into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def control_bundle_digest(control_dir: Path) -> tuple[str, dict[str, str]]:
    """Hash every flat Python module in the frozen controller bundle."""
    files: dict[str, str] = {}
    for path in sorted(control_dir.glob("*.py"), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"control bundle entry is not a regular file: {path.name}")
        files[path.name] = sha256_file(path)
    if not files:
        raise ValueError("control bundle has no top-level Python modules")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), files


def validate_control_manifest(path: Path, control_dir: Path) -> str:
    """Verify root's manifest against the exact local set of controller modules."""
    if path.is_symlink() or control_dir.is_symlink():
        raise ValueError("control manifest and module directory cannot be symlinks")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "tabdata-control-bundle.v1":
        raise ValueError("control bundle manifest schema is not recognized")
    digest, actual_files = control_bundle_digest(control_dir)
    declared_files = manifest.get("files")
    if not isinstance(declared_files, dict) or declared_files != actual_files:
        raise ValueError("control bundle manifest files differ from local controller modules")
    canonical = json.dumps(declared_files, sort_keys=True, separators=(",", ":")).encode()
    declared_digest = hashlib.sha256(canonical).hexdigest()
    if manifest.get("sha256") != declared_digest or declared_digest != digest:
        raise ValueError("control bundle manifest digest does not match local controller modules")
    return digest


def validate_host_admission(
    value: dict[str, Any], wave_sha256: str, actor_sha256: str, control_sha256: str
) -> dict[str, Any]:
    """Require the root host watcher’s verified exact-container admission receipt."""
    expected: dict[str, Any] = {
        "verified": True,
        "host_cgroup_verified": True,
        "cpu_count": 16,
        "memory_max_bytes": 192 * 1024**3,
        "memory_swap_max_bytes": 0,
        "pids_max": EXPECTED_PIDS,
        "wave_plan_sha256": wave_sha256,
        "actor_bundle_sha256": actor_sha256,
        "control_bundle_sha256": control_sha256,
    }
    for key, expected_value in expected.items():
        actual = value.get(key)
        if type(expected_value) is bool:
            valid = actual is expected_value
        elif type(expected_value) is int:
            valid = type(actual) is int and actual == expected_value
        else:
            valid = actual == expected_value
        if not valid:
            raise ValueError(f"host admission field {key} differs from the authorized envelope")
    container_id = value.get("docker_container_id")
    if (
        not isinstance(container_id, str)
        or len(container_id) != 64
        or any(character not in "0123456789abcdef" for character in container_id)
    ):
        raise ValueError("host admission must identify one lowercase 64-hex Docker container ID")
    return value


def load_wave(
    path: Path,
    expected_sha256: str | None = WAVE_SHA256,
    expected_actor_sha256: str | None = ACTOR_SHA256,
) -> tuple[dict[str, Any], str]:
    """Load and verify the externally frozen wave plan and its eight-cell schedule."""
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("WAVE_PLAN.json bytes differ from the controller pin")
    wave = json.loads(payload)
    actor_sha = wave.get("actor_bundle_identity", {}).get("sha256")
    if actor_sha != ACTOR_SHA256 or (
        expected_actor_sha256 is not None and actor_sha != expected_actor_sha256
    ):
        raise ValueError("wave actor bundle differs from the controller pin")
    validate_schedule(wave)
    return wave, digest


def validate_authority(
    path: Path, wave_sha256: str, actor_sha256: str, control_sha256: str
) -> dict[str, Any]:
    """Require root's explicit conditional execution pin before any mutating step."""
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "wave_plan_sha256": wave_sha256,
        "actor_bundle_sha256": actor_sha256,
        "control_bundle_sha256": control_sha256,
    }
    if value.get("conditional_execute") is not True:
        raise ValueError("control authority does not enable conditional execution")
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("control authority pins do not match the frozen wave and actor bundle")
    return value


def validate_schedule(wave: dict[str, Any]) -> None:
    """Reject duplicate GPUs/CPUs or any cell set outside the frozen 4x2 wave."""
    jobs = wave.get("jobs")
    if (
        wave.get("execute_enabled") is not False
        or wave.get("root") != "/data2/lc-epanov_s/tabdata"
        or wave.get("cpu_limit_total") != EXPECTED_CPUS
        or wave.get("ram_limit_gib_total") != 192
        or wave.get("gpus") != EXPECTED_JOBS
        or wave.get("co_location_authorized") is not True
        or wave.get("allocator_gib_per_fit") != 6
        or wave.get("wall_hours_after_first_profile") != 24
        or not isinstance(jobs, list)
        or len(jobs) != EXPECTED_JOBS
    ):
        raise ValueError("wave fields differ from the finite authorized envelope")
    expected = {(arm, fold) for arm in ("wide", "deep") for fold in range(4)}
    actual = {(job.get("arm"), job.get("fold")) for job in jobs}
    uuids = [job.get("gpu_uuid") for job in jobs]
    cpus = [cpu for job in jobs for cpu in job.get("cpu_ids_initial", [])]
    if actual != expected or any(job.get("seed") != 1002 for job in jobs):
        raise ValueError("wave cells do not equal the fixed wide/deep four-fold seed-1002 set")
    if len(uuids) != EXPECTED_JOBS or len(set(uuids)) != EXPECTED_JOBS:
        raise ValueError("wave assigns a duplicate or missing GPU UUID")
    if len(cpus) != EXPECTED_CPUS or set(cpus) != set(range(EXPECTED_CPUS)):
        raise ValueError("wave CPU affinities must be disjoint pairs covering CPUs 0 through 15")
    for job in jobs:
        if len(job.get("cpu_ids_initial", [])) != 2:
            raise ValueError("every cell must have exactly two initial CPU cores")
        argv = job.get("argv", [])
        if argv.count("--mode") != 1 or argv[-1] != "fit":
            raise ValueError("wave actor argv must contain exactly one frozen fit mode")


def parse_gpu_query(output: str) -> dict[str, dict[str, int | str]]:
    """Parse nvidia-smi's UUID,name,free-MiB,total-MiB CSV rows."""
    result: dict[str, dict[str, int | str]] = {}
    for line in output.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 4:
            raise ValueError(f"malformed nvidia-smi row: {line!r}")
        uuid, name = fields[:2]
        free_mib, total_mib = int(fields[2]), int(fields[3])
        if uuid in result:
            raise ValueError(f"nvidia-smi returned duplicate GPU UUID {uuid}")
        result[uuid] = {"name": name, "free_mib": free_mib, "total_mib": total_mib}
    return result


def validate_gpu_inventory(
    inventory: dict[str, dict[str, int | str]],
    jobs: list[dict[str, Any]],
    free_mib: int = GPU_FREE_MIN,
) -> None:
    """Require exactly the eight assigned L4s and minimum free memory on each."""
    expected = {job["gpu_uuid"] for job in jobs}
    if set(inventory) != expected:
        raise ValueError("visible nvidia-smi UUID inventory differs from the eight wave GPUs")
    for uuid in expected:
        record = inventory[uuid]
        if "B300" not in str(record["name"]) or int(record["free_mib"]) < free_mib:
            raise ValueError(f"GPU {uuid} is not an L4 with at least {free_mib} MiB free")


def validate_profile_log(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Parse and enforce the required single finite actual-shape profile evidence pair."""
    records: dict[str, list[dict[str, Any]]] = {"PROFILE": [], "PROFILE_UPDATE_EVIDENCE": []}
    for line in text.splitlines():
        for name in records:
            prefix = name + " "
            if line.startswith(prefix):
                value = json.loads(line[len(prefix) :])
                if not isinstance(value, dict):
                    raise ValueError(f"{name} record must contain a JSON object")
                records[name].append(value)
    if any(len(values) != 1 for values in records.values()):
        raise ValueError(
            "profile log must contain exactly one PROFILE and one update-evidence record"
        )
    timings, evidence = records["PROFILE"][0], records["PROFILE_UPDATE_EVIDENCE"][0]
    for key, value in timings.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"PROFILE timing {key} must be finite numeric data")
        if value < 0:
            raise ValueError(f"PROFILE timing {key} cannot be negative")
    if not {"step_wall", "seq_per_s"}.issubset(timings):
        raise ValueError("PROFILE record lacks step-wall and throughput timings")
    for key in ("step_wall", "seq_per_s"):
        if timings[key] <= 0:
            raise ValueError(f"PROFILE timing {key} must be positive")
    expected = {
        "optimizer_attempts": 40,
        "optimizer_completed_updates": 40,
        "amp_skipped_updates": 0,
        "grad_scaler_enabled": False,
        "loss_checks": 40,
        "loss_finite": True,
        "gradient_finite": True,
        "gradient_stats_steps": 40,
        "gradient_stats_nonfinite_steps": 0,
        "counts_consistent": True,
        "valid_for_throughput": True,
    }
    integer_fields = {
        "optimizer_attempts",
        "optimizer_completed_updates",
        "amp_skipped_updates",
        "loss_checks",
        "gradient_stats_steps",
        "gradient_stats_nonfinite_steps",
    }
    for key in integer_fields:
        if type(evidence.get(key)) is not int or evidence[key] != expected[key]:
            raise ValueError(f"profile update evidence field {key} differs from the exact gate")
    for key in set(expected) - integer_fields:
        if evidence.get(key) is not expected[key]:
            raise ValueError(f"profile update evidence field {key} differs from the exact gate")
    return timings, evidence


def validate_memory_budget(
    memory_max: int,
    memory_current: int,
    wide_peak: int,
    deep_peak: int,
    headroom: int = DISK_FREE_MIN,
) -> int:
    """Require measured eight-cell memory plus reserve to fit the container cap."""
    if memory_max != EXPECTED_RAM:
        raise ValueError("container memory.max is not the authorized 192 GiB")
    if min(memory_current, wide_peak, deep_peak) < 0:
        raise ValueError("memory measurements must be nonnegative")
    if max(wide_peak, deep_peak) > JOB_RSS_LIMIT:
        raise ValueError("an actual-shape profile exceeded the per-job 24 GiB RSS gate")
    projected = memory_current + 4 * wide_peak + 4 * deep_peak + headroom
    if projected > memory_max:
        raise ValueError("measured profile peaks plus headroom exceed the eight-fit memory budget")
    return projected
