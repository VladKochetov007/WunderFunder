"""Fail-closed TPU fit headroom checks for the native R4 packet."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

HEADROOM_POLICY_SCHEMA = "wnn.loss-pair-tpu.fit-headroom-policy.v1"
HEADROOM_GATE_SCHEMA = "wnn.loss-pair-tpu.fit-headroom-gate.v1"
EXPECTED_TPU_DEVICE_COUNT = 8
REQUIRED_POLICY_BYTES = (
    "minimum_device_free_bytes",
    "minimum_host_available_bytes",
    "minimum_cgroup_headroom_bytes",
    "minimum_disk_free_bytes",
)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_policy(policy: Any) -> tuple[dict[str, int] | None, list[str]]:
    """Require root-provided positive byte reserves; do not infer local limits."""
    if not isinstance(policy, dict) or policy.get("schema") != HEADROOM_POLICY_SCHEMA:
        return None, ["missing_or_invalid_headroom_policy"]
    invalid = [name for name in REQUIRED_POLICY_BYTES if not _positive_int(policy.get(name))]
    if invalid:
        return None, [f"invalid_or_missing_policy:{name}" for name in invalid]
    return {name: policy[name] for name in REQUIRED_POLICY_BYTES}, []


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _host_available_bytes(meminfo_path: Path = Path("/proc/meminfo")) -> int | None:
    try:
        for line in meminfo_path.read_text().splitlines():
            key, raw = line.split(":", 1)
            if key == "MemAvailable":
                value, unit = raw.strip().split()[:2]
                return int(value) * (1024 if unit == "kB" else 1)
    except (OSError, ValueError):
        return None
    return None


def _cgroup_memory(root: Path = Path("/sys/fs/cgroup")) -> dict[str, int | str | None]:
    """Read cgroup-v2 or v1 current use and effective finite memory ceiling."""
    v2_current = _read_int(root / "memory.current")
    if v2_current is not None:
        limits = [
            value
            for value in (_read_int(root / "memory.high"), _read_int(root / "memory.max"))
            if value is not None
        ]
        ceiling = min(limits) if limits else None
        return {
            "version": "v2",
            "current_bytes": v2_current,
            "ceiling_bytes": ceiling,
            "headroom_bytes": max(0, ceiling - v2_current) if ceiling is not None else None,
        }
    v1_root = root / "memory"
    current = _read_int(v1_root / "memory.usage_in_bytes")
    ceiling = _read_int(v1_root / "memory.limit_in_bytes")
    if current is None:
        return {"version": None, "current_bytes": None, "ceiling_bytes": None, "headroom_bytes": None}
    # v1 reports a very large sentinel instead of "max" for an unlimited limit.
    if ceiling is not None and ceiling >= (1 << 60):
        ceiling = None
    return {
        "version": "v1",
        "current_bytes": current,
        "ceiling_bytes": ceiling,
        "headroom_bytes": max(0, ceiling - current) if ceiling is not None else None,
    }


def sample_live_resources(output_root: Path) -> dict[str, Any]:
    """Capture host, cgroup, and output-filesystem headroom at the call site."""
    return {
        "host_available_bytes": _host_available_bytes(),
        "cgroup": _cgroup_memory(),
        "disk_free_bytes": shutil.disk_usage(output_root).free,
    }


def evaluate_live_resources(policy: Any, live: Any) -> dict[str, Any]:
    """Check root-reserved live host/cgroup/disk margin for the entire fit phase."""
    normalized, errors = validate_policy(policy)
    observed = live if isinstance(live, dict) else {}
    cgroup = observed.get("cgroup") if isinstance(observed.get("cgroup"), dict) else {}
    checks = {
        "host_available_bytes": (observed.get("host_available_bytes"), "minimum_host_available_bytes"),
        "cgroup_headroom_bytes": (cgroup.get("headroom_bytes"), "minimum_cgroup_headroom_bytes"),
        "disk_free_bytes": (observed.get("disk_free_bytes"), "minimum_disk_free_bytes"),
    }
    outcomes: dict[str, dict[str, Any]] = {}
    for name, (actual, reserve_name) in checks.items():
        reserve = normalized.get(reserve_name) if normalized else None
        passed = _positive_int(actual) and reserve is not None and actual >= reserve
        outcomes[name] = {
            "observed_bytes": actual if _positive_int(actual) else None,
            "required_bytes": reserve,
            "passed": passed,
        }
        if not passed:
            errors.append(f"insufficient_or_missing_live_{name}")
    return {
        "schema": HEADROOM_GATE_SCHEMA,
        "passed": not errors,
        "errors": errors,
        "live_checks": outcomes,
        "cgroup": cgroup,
    }


def evaluate_actual_device_inventory(
    platform: Any, accelerator_count: Any, inventory: Any
) -> dict[str, Any]:
    """Require the actual eight-device TPU v5e profile inventory."""
    errors: list[str] = []
    rows = inventory if isinstance(inventory, list) else []
    if platform != "tpu":
        errors.append("actual_profile_platform_is_not_tpu")
    if accelerator_count != EXPECTED_TPU_DEVICE_COUNT:
        errors.append("actual_accelerator_count_differs_from_tpu_v5e8")
    if len(rows) != EXPECTED_TPU_DEVICE_COUNT:
        errors.append("actual_device_inventory_is_incomplete")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or row.get("platform") != "tpu"
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or not isinstance(row.get("kind"), str)
            or not row["kind"]
        ):
            errors.append(f"actual_device_inventory_entry_{index}_is_incomplete")
    return {
        "passed": not errors,
        "expected_device_count": EXPECTED_TPU_DEVICE_COUNT,
        "observed_device_count": len(rows),
        "errors": errors,
    }


def evaluate_profile_headroom(
    policy: Any,
    profile: Any,
    live: Any,
    expected_device_count: Any,
) -> dict[str, Any]:
    """Compare observed full-shape device peaks with capacity and live reserves."""
    normalized, errors = validate_policy(policy)
    expected = expected_device_count if _positive_int(expected_device_count) else None
    if expected is None:
        errors.append("missing_or_invalid_actual_device_count")
    rows_by_loss = profile if isinstance(profile, dict) else {}
    device_checks: dict[str, Any] = {}
    for loss in ("wmse", "huber"):
        row = rows_by_loss.get(loss)
        if not isinstance(row, dict) or row.get("shape") != [32, 2000, 211] or row.get("finite") is not True:
            errors.append(f"missing_or_invalid_{loss}_full_shape_profile")
        memory = row.get("memory_after_full_shape") if isinstance(row, dict) else None
        stats = memory.get("device_memory_stats") if isinstance(memory, dict) else None
        if not isinstance(stats, list) or expected is None or len(stats) != expected:
            errors.append(f"missing_or_incomplete_{loss}_device_memory_stats")
            device_checks[loss] = {"passed": False, "devices": []}
            continue
        devices: list[dict[str, Any]] = []
        for index, item in enumerate(stats):
            if not isinstance(item, dict):
                errors.append(f"missing_{loss}_device_{index}_memory_stats")
                devices.append({"device_index": index, "passed": False})
                continue
            capacity = item.get("bytes_limit")
            current = item.get("bytes_in_use")
            peak = item.get("peak_bytes_in_use")
            usage_values = [value for value in (current, peak) if _nonnegative_int(value)]
            if not _positive_int(capacity) or not usage_values:
                errors.append(f"incomplete_{loss}_device_{index}_capacity_or_usage")
                devices.append({"device_index": index, "passed": False})
                continue
            peak_usage = max(usage_values)
            free = max(0, capacity - peak_usage)
            required = normalized.get("minimum_device_free_bytes") if normalized else None
            passed = required is not None and free >= required
            if not passed:
                errors.append(f"insufficient_{loss}_device_{index}_headroom")
            devices.append(
                {
                    "device_index": index,
                    "bytes_limit": capacity,
                    "bytes_in_use": current if _nonnegative_int(current) else None,
                    "peak_bytes_in_use": peak if _nonnegative_int(peak) else None,
                    "conservative_peak_bytes": peak_usage,
                    "free_after_peak_bytes": free,
                    "required_free_bytes": required,
                    "passed": passed,
                }
            )
        device_checks[loss] = {"passed": all(row.get("passed") for row in devices), "devices": devices}
    live_check = evaluate_live_resources(policy, live)
    errors.extend(live_check["errors"])
    return {
        "schema": HEADROOM_GATE_SCHEMA,
        "passed": not errors,
        "errors": errors,
        "expected_device_count": expected,
        "device_checks": device_checks,
        "live_check": live_check,
    }
