"""Validate the reviewer block attached to a canonical profile receipt."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from bilinear_entmax.bindings import (
    FULL_SHAPE,
    PACKET,
    SPLIT_SHA256,
    canonical,
    load_cell,
    read_object,
    source_identity,
)


def _positive_integer(value: object, label: str) -> int:
    """Require a positive integer receipt counter, excluding booleans."""
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_positive(value: object, label: str) -> float:
    """Require a finite positive JSON numeric value."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _profile_shape(config: dict[str, Any]) -> dict[str, object]:
    """Describe the actual frozen model and batch shape in a profile config."""
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("profile config lacks its model shape")
    return {
        "architecture": model.get("architecture"),
        "hidden": model.get("hidden"),
        "layers": model.get("layers"),
        "heads": model.get("heads"),
        "batch_size": config.get("batch_size"),
        "chunk_rows": config.get("chunk_rows"),
        "context_rows": model.get("context_rows"),
        "required_world_size": config.get("required_world_size"),
        "required_device": config.get("required_device"),
        "precision_mode": config.get("precision_mode"),
    }


def validate_profile(
    profile_path: Path,
    cell: dict[str, Any],
    runtime_root: Path,
    max_host_ram_gib: float,
    max_vram_gib_per_rank: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify profile, immutable run metadata, source, cell, reload and resource evidence."""
    profile = read_object(profile_path)
    metadata_path = profile_path.parent / "run_metadata.json"
    metadata = read_object(metadata_path)
    identity = profile.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("profile receipt lacks its canonical run identity")
    source = source_identity(runtime_root)
    config = metadata.get("config")
    metadata_source = metadata.get("source")
    provenance = metadata.get("provenance")
    if not all(isinstance(value, dict) for value in (config, metadata_source, provenance)):
        raise ValueError("immutable run metadata is incomplete")
    if (
        profile.get("receipt_schema") != "wnn.sequence-profile.v1"
        or profile.get("status") != "profile_complete"
        or profile.get("approved") is not False
        or profile.get("updates") != 50
        or profile.get("scientific_checkpoint_written") is not False
        or profile.get("errors") != []
        or profile.get("error_count") != 0
        or identity.get("bilinear_entmax_cell") != cell
        or identity.get("world_size") != 2
        or identity.get("source_sha256") != source["sha256"]
        or identity.get("source_files") != source["files"]
        or metadata_source.get("sha256") != source["sha256"]
        or metadata_source.get("files") != source["files"]
    ):
        raise ValueError("profile completion, approval, cell, world, or source identity is invalid")
    profile_data = identity.get("data")
    if (
        identity.get("config_sha256") != canonical(config)
        or not isinstance(profile_data, dict)
        or identity.get("data_sha256") != canonical(profile_data)
        or provenance.get("data_sha256") != identity.get("data_sha256")
        or provenance.get("data") != profile_data
        or provenance.get("runtime") != identity.get("runtime")
        or provenance.get("world_size") != 2
    ):
        raise ValueError("profile run metadata does not match its immutable run identity")
    if not isinstance(config, dict) or not isinstance(profile_data, dict):
        raise ValueError("profile config or fitting-data identity is malformed")
    if _profile_shape(config) != FULL_SHAPE:
        raise ValueError("profile was not measured at the frozen full training shape")
    from wnn.training.sequence_config import load_config

    cell_descriptor, native_path = load_cell(PACKET / "configs" / f"{cell['id']}.json")
    base_config = {key: value for key, value in config.items() if key != "bilinear_entmax"}
    if cell_descriptor != cell or base_config != load_config(native_path):
        raise ValueError("profile normalized recipe differs from the immutable PR75 cell recipe")
    extension = config.get("bilinear_entmax")
    if (
        not isinstance(extension, dict)
        or extension.get("schema") != "bilinear-entmax.model.v1"
        or extension.get("cell") != cell
        or identity.get("config_sha256") != canonical(config)
        or identity.get("data", {}).get("bilinear_entmax_preprocessing_sha256")
        != extension.get("preprocessing_sha256")
        or identity.get("data", {}).get("split_sha256") != SPLIT_SHA256
        or not isinstance(identity.get("data", {}).get("fit_members"), list)
        or len(identity.get("data", {}).get("fit_members", [])) != 10_325
        or config.get("precision_mode") != "bf16_bf16x9"
        or config.get("epochs") != 4
        or config.get("seed") != cell["seed"]
        or config.get("required_world_size") != 2
        or config.get("required_device") != "B300"
    ):
        raise ValueError("profile config is not the declared BF16x9 B300 fixed-E4 cell")
    runtime = identity.get("runtime")
    devices = runtime.get("devices") if isinstance(runtime, dict) else None
    if (
        not isinstance(devices, list)
        or len(devices) != 2
        or any(
            not isinstance(item, dict) or "B300" not in str(item.get("name", ""))
            for item in devices
        )
    ):
        raise ValueError("profile runtime identity does not show two B300 ranks")

    _finite_positive(profile.get("elapsed_seconds"), "elapsed_seconds")
    forwarded = _positive_integer(profile.get("forwarded_rows"), "forwarded_rows")
    real = _positive_integer(profile.get("real_rows"), "real_rows")
    valid = _positive_integer(profile.get("valid_rows"), "valid_rows")
    if forwarded < real or valid > real:
        raise ValueError("profile work counters are internally inconsistent")
    epochs = profile.get("epochs_traversed")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 0:
        raise ValueError("profile epoch count must be a nonnegative integer")
    losses = profile.get("losses")
    if not isinstance(losses, dict) or set(losses) != {"label", "distillation", "hint", "total"}:
        raise ValueError("profile objective receipt is incomplete")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in losses.values()
    ):
        raise ValueError("profile losses must be finite and nonnegative")
    parameters = profile.get("model_parameters")
    if not isinstance(parameters, dict) or parameters != identity.get("model_parameters"):
        raise ValueError("profile model parameter count differs from its run identity")
    if provenance.get("model_parameters") != parameters:
        raise ValueError("run metadata model parameter count differs from profile")

    memory = profile.get("memory_by_rank")
    if not isinstance(memory, list) or len(memory) != 2:
        raise ValueError("profile must contain resource receipts for both ranks")
    host_bytes = 0
    allocated_by_rank: list[int] = []
    reserved_by_rank: list[int] = []
    device_memory_by_rank: list[int] = []
    host_memory_by_rank: list[int] = []
    for rank, item in enumerate(memory):
        if not isinstance(item, dict) or item.get("rank") != rank:
            raise ValueError("profile rank resource records are missing or misordered")
        allocated = _positive_integer(item.get("peak_allocated_bytes"), "peak_allocated_bytes")
        reserved = _positive_integer(item.get("peak_reserved_bytes"), "peak_reserved_bytes")
        total = _positive_integer(
            item.get("device_total_memory_bytes"), "device_total_memory_bytes"
        )
        host = _positive_integer(item.get("host_peak_rss_bytes"), "host_peak_rss_bytes")
        host_bytes += host
        allocated_by_rank.append(allocated)
        reserved_by_rank.append(reserved)
        device_memory_by_rank.append(total)
        host_memory_by_rank.append(host)
        if allocated > reserved or reserved > total:
            raise ValueError("profile GPU memory counters are internally inconsistent")
        if reserved > max_vram_gib_per_rank * (1024**3):
            raise ValueError(f"rank {rank} reserved VRAM exceeds the approved ceiling")
        if devices[rank].get("total_memory_bytes") != total:
            raise ValueError("profile GPU total memory differs from runtime identity")
    if host_bytes > max_host_ram_gib * (1024**3):
        raise ValueError("conservative sum of per-rank host RSS exceeds the approved ceiling")
    if (
        profile.get("peak_memory_bytes") != max(allocated_by_rank)
        or profile.get("peak_reserved_bytes") != max(reserved_by_rank)
        or profile.get("device_total_memory_bytes") != device_memory_by_rank
        or profile.get("host_peak_rss_bytes") != host_memory_by_rank
    ):
        raise ValueError("profile aggregate memory values differ from per-rank records")

    reload = profile.get("engineering_reload")
    by_rank = reload.get("by_rank") if isinstance(reload, dict) else None
    if (
        not isinstance(reload, dict)
        or reload.get("status") != "passed"
        or reload.get("scientific_resume_schema_written") is not False
        or not isinstance(by_rank, list)
        or len(by_rank) != 2
    ):
        raise ValueError("profile lacks a passed two-rank engineering reload receipt")
    for rank, item in enumerate(by_rank):
        if (
            not isinstance(item, dict)
            or item.get("status") != "passed"
            or item.get("rank") != rank
            or item.get("model_optimizer_scheduler_rng_equal") is not True
            or item.get("temporary_engineering_state_removed") is not True
            or item.get("scientific_resume_schema_written") is not False
        ):
            raise ValueError("engineering reload did not pass for both ranks")
    return profile, identity


def validate_approval(
    receipt: dict[str, Any], identity: dict[str, object], cell: dict[str, Any]
) -> None:
    """Bind a human approval to the untouched profile bytes, fit identity, and cell."""
    approval = receipt.get("approval")
    receipt_identity = receipt.get("identity")
    if not isinstance(approval, dict) or not isinstance(receipt_identity, dict):
        raise ValueError("approved profile lacks its human review or canonical identity")
    fields = {
        "schema",
        "reviewer",
        "note",
        "profile_sha256",
        "source_sha256",
        "cell",
        "approved_at_utc",
        "max_host_ram_gib",
        "max_vram_gib_per_rank",
    }
    original = {key: value for key, value in receipt.items() if key != "approval"}
    original["approved"] = False
    encoded = (
        json.dumps(original, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    expected_sha256 = hashlib.sha256(encoded).hexdigest()
    for field in ("max_host_ram_gib", "max_vram_gib_per_rank"):
        value = approval.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"profile approval {field} must be finite and positive")
    if (
        set(approval) != fields
        or approval.get("schema") != "bilinear-entmax.profile-approval.v1"
        or not isinstance(approval.get("reviewer"), str)
        or not approval["reviewer"].strip()
        or not isinstance(approval.get("note"), str)
        or not approval["note"].strip()
        or not isinstance(approval.get("approved_at_utc"), str)
        or not approval["approved_at_utc"].strip()
        or approval.get("profile_sha256") != expected_sha256
        or approval.get("source_sha256") != identity.get("source_sha256")
        or approval.get("cell") != cell
        or receipt_identity != identity
    ):
        raise ValueError("profile approval does not bind the original profile and fit identity")
