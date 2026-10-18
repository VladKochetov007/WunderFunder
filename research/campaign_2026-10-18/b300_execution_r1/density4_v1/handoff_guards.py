from __future__ import annotations

import math

GIB = 1024**3
MEMORY_LIMIT = 192 * GIB
MEMORY_ADMISSION_CEILING = 180 * GIB
WORKER_RSS_LIMIT = 24 * GIB
DISK_FLOOR = 8 * GIB
PID_LIMIT = 512
OOM_KEYS = ("high", "max", "oom", "oom_group_kill", "oom_kill")


def validate_aggregate_resources(
    receipt: dict[str, object], baseline_oom: dict[str, int]
) -> None:
    """Fail closed unless combined old/new/R01 resource evidence is admissible."""
    if receipt.get("scope") != "b300_campaign_aggregate":
        raise ValueError("resource receipt must cover old, successor and R01 scopes")
    if receipt.get("ancestor_limits_verified") is not True:
        raise ValueError("shared ancestor cgroup limits were not verified")
    numeric_limits = {
        "memory_max_bytes": MEMORY_LIMIT,
        "memory_swap_max_bytes": 0,
        "pids_max": PID_LIMIT,
    }
    for key, expected in numeric_limits.items():
        value = receipt.get(key)
        if type(value) is not int or value != expected:
            raise ValueError(f"aggregate {key} must equal {expected}")
    current = receipt.get("memory_current_bytes")
    swap_current = receipt.get("memory_swap_current_bytes")
    tasks = receipt.get("pids_current")
    disk_free = receipt.get("disk_free_bytes")
    for key, value in (
        ("memory_current_bytes", current),
        ("memory_swap_current_bytes", swap_current),
        ("pids_current", tasks),
        ("disk_free_bytes", disk_free),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"aggregate {key} must be a nonnegative integer")
    if current >= MEMORY_ADMISSION_CEILING:
        raise ValueError("aggregate memory is at or above the 180 GiB stop-admission gate")
    if swap_current != 0 or tasks >= PID_LIMIT:
        raise ValueError("aggregate swap or task count violates the execution envelope")
    if disk_free < DISK_FLOOR:
        raise ValueError("less than 8 GiB of physical disk space remains")
    events = receipt.get("oom_events")
    if not isinstance(events, dict):
        raise ValueError("aggregate OOM counters are missing")
    for key in OOM_KEYS:
        value, baseline = events.get(key), baseline_oom.get(key)
        if type(value) is not int or type(baseline) is not int or value != baseline:
            raise ValueError(f"OOM counter changed during the observed interval: {key}")
    rss = receipt.get("worker_group_rss_bytes")
    if not isinstance(rss, dict) or not rss:
        raise ValueError("worker process-group RSS evidence is required")
    for group_id, value in rss.items():
        if not isinstance(group_id, str) or type(value) is not int or value < 0:
            raise ValueError("worker RSS records must bind names to nonnegative byte counts")
        if value > WORKER_RSS_LIMIT:
            raise ValueError(f"worker group exceeds the 24 GiB RSS cap: {group_id}")


def validate_absolute_deadlines(
    *,
    old_unit_started_unix: float,
    current_unix: float,
    job_started_unix: dict[str, float],
    campaign_cap_seconds: int = 120 * 3600,
    fit_cap_seconds: int = 12 * 3600,
) -> tuple[float, dict[str, float]]:
    """Carry original wall deadlines forward instead of granting a fresh budget."""
    values = (old_unit_started_unix, current_unix)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("deadline timestamps must be finite positive Unix times")
    if campaign_cap_seconds <= 0 or fit_cap_seconds <= 0:
        raise ValueError("campaign and fit caps must be positive")
    campaign_deadline = old_unit_started_unix + campaign_cap_seconds
    if current_unix >= campaign_deadline:
        raise TimeoutError("the original campaign deadline has expired")
    deadlines: dict[str, float] = {}
    for job_id, started in job_started_unix.items():
        if not math.isfinite(started) or started <= 0:
            raise ValueError(f"invalid original fit start time for {job_id}")
        deadline = started + fit_cap_seconds
        if current_unix >= deadline:
            raise TimeoutError(f"the original per-fit deadline has expired: {job_id}")
        deadlines[job_id] = min(deadline, campaign_deadline)
    return campaign_deadline, deadlines


def can_admit_new_fit(
    *,
    now_unix: float,
    campaign_deadline_unix: float,
    conservative_fit_seconds: float,
    fit_cap_seconds: float = 12 * 3600,
) -> bool:
    """Do not admit a fit unless its measured forecast and 12-hour cap fit."""
    values = (now_unix, campaign_deadline_unix, conservative_fit_seconds, fit_cap_seconds)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("fit admission timestamps and budgets must be positive")
    return campaign_deadline_unix - now_unix >= max(
        conservative_fit_seconds, fit_cap_seconds
    )
