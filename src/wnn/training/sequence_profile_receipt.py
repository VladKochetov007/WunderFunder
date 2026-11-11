from __future__ import annotations

import json
import math
from pathlib import Path

from wnn.training.sequence_provenance import file_sha256


def validate_profile_receipt(path: Path, identity: dict[str, object]) -> str:
    """Require an approved profile receipt matching identities and reload policy."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("profile receipt must contain a JSON object")
    if value.get("receipt_schema") != "wnn.sequence-profile.v1":
        raise ValueError("unsupported engineering profile receipt")
    if value.get("status") != "profile_complete" or value.get("approved") is not True:
        raise ValueError("full fitting requires an explicitly approved completed profile receipt")
    if value.get("identity") != identity:
        raise ValueError("profile receipt config/data/source/runtime identity does not match")
    for name in ("updates", "forwarded_rows", "real_rows", "valid_rows"):
        if not _positive_integer(value.get(name)):
            raise ValueError(f"profile receipt {name} must be a positive integer")
    epochs = value.get("epochs_traversed")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 0:
        raise ValueError("profile completed-epoch count must be a nonnegative integer")
    if value["forwarded_rows"] < value["real_rows"] or value["valid_rows"] > value["real_rows"]:
        raise ValueError("profile work counts are internally inconsistent")
    if not _finite_nonnegative(value.get("elapsed_seconds")) or value["elapsed_seconds"] <= 0:
        raise ValueError("profile receipt elapsed time must be finite and positive")
    if value.get("scientific_checkpoint_written") is not False:
        raise ValueError("engineering profiles must not write a scientific checkpoint")
    if identity.get("profile_reload_required") is True:
        _validate_reload_receipt(value.get("engineering_reload"), identity)
    _validate_profile_summary(value, identity)
    return file_sha256(path)


def _validate_reload_receipt(reload: object, identity: dict[str, object]) -> None:
    """Require exact per-rank model, optimizer, scheduler, and RNG reload evidence."""
    ranks = reload.get("by_rank") if isinstance(reload, dict) else None
    if (
        not isinstance(reload, dict)
        or reload.get("status") != "passed"
        or reload.get("scientific_resume_schema_written") is not False
        or not isinstance(ranks, list)
        or len(ranks) != identity.get("world_size")
        or any(
            not isinstance(rank, dict)
            or rank.get("status") != "passed"
            or rank.get("model_optimizer_scheduler_rng_equal") is not True
            or rank.get("scientific_resume_schema_written") is not False
            for rank in ranks
        )
    ):
        raise ValueError(
            "profile receipt is missing the required per-rank model/optimizer/RNG reload"
        )


def _validate_profile_summary(value: dict[str, object], identity: dict[str, object]) -> None:
    """Validate losses, model size, resource bounds, and optional device bindings."""
    errors = value.get("errors")
    if errors != [] or value.get("error_count") != 0:
        raise ValueError("profile receipt must have no runtime errors")
    losses = value.get("losses")
    if not isinstance(losses, dict) or set(losses) != {"label", "distillation", "hint", "total"}:
        raise ValueError("profile receipt is missing its objective summary")
    if not all(_finite_nonnegative(loss) for loss in losses.values()):
        raise ValueError("profile losses must be finite and nonnegative")
    if value.get("model_parameters") != identity.get("model_parameters"):
        raise ValueError("profile model parameter count does not match this fit")
    memory = value.get("memory_by_rank")
    world_size = identity.get("world_size")
    if not isinstance(world_size, int) or isinstance(world_size, bool):
        raise ValueError("run identity has an invalid world size")
    if not isinstance(memory, list) or len(memory) != world_size:
        raise ValueError("profile receipt must record memory metrics for every rank")
    if [item.get("rank") for item in memory if isinstance(item, dict)] != list(range(world_size)):
        raise ValueError("profile resource records are missing or misordered by rank")
    memory_fields = (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "device_total_memory_bytes",
        "host_peak_rss_bytes",
    )
    for item in memory:
        if not isinstance(item, dict) or not all(
            _positive_integer(item.get(field)) for field in memory_fields
        ):
            raise ValueError("profile resource values must be positive integer byte counts")
        if item["peak_allocated_bytes"] > item["peak_reserved_bytes"]:
            raise ValueError("profile allocated memory cannot exceed reserved memory")
        if item["peak_reserved_bytes"] > item["device_total_memory_bytes"]:
            raise ValueError("profile reserved memory cannot exceed total device memory")
    runtime = identity.get("runtime")
    if isinstance(runtime, dict) and "devices" in runtime:
        _validate_device_memory(runtime["devices"], memory, world_size)
    if value.get("peak_memory_bytes") != max(item["peak_allocated_bytes"] for item in memory):
        raise ValueError("profile aggregate allocated memory does not match per-rank values")
    if value.get("peak_reserved_bytes") != max(item["peak_reserved_bytes"] for item in memory):
        raise ValueError("profile aggregate reserved memory does not match per-rank values")


def _validate_device_memory(devices: object, memory: list[object], world_size: int) -> None:
    """Bind per-rank memory receipts to physical runtime device identities."""
    if not isinstance(devices, list) or len(devices) != world_size:
        raise ValueError("run identity must bind one runtime device per rank")
    for rank, (device, item) in enumerate(zip(devices, memory, strict=True)):
        if (
            not isinstance(device, dict)
            or not isinstance(item, dict)
            or device.get("rank") != rank
            or device.get("total_memory_bytes") != item.get("device_total_memory_bytes")
        ):
            raise ValueError("profile device memory does not match the runtime identity")


def _positive_integer(value: object) -> bool:
    """Accept only positive Python integer counters, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_nonnegative(value: object) -> bool:
    """Accept finite nonnegative numeric receipt values, excluding booleans."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )
