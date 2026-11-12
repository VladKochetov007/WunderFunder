"""Review and admit one complete, source-matched 50-update engineering profile."""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Any

from signal_routes.bindings import (
    PACKET,
    SPLIT_SHA256,
    VARIANTS,
    admit_runtime,
    canonical,
    digest,
    load_cell,
    read_object,
    source_identity,
    write_new,
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


def _validate_profile(
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
        or identity.get("signal_routes_cell") != cell
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
    from wnn.training.sequence_config import load_config

    cell_descriptor, native_path = load_cell(PACKET / "configs" / f"{cell['id']}.json")
    base_config = {key: value for key, value in config.items() if key != "signal_routes"}
    if cell_descriptor != cell or base_config != load_config(native_path):
        raise ValueError("profile normalized recipe differs from the immutable PR75 cell recipe")
    extension = config.get("signal_routes")
    if (
        not isinstance(extension, dict)
        or extension.get("schema") != "signal-routes.model.v1"
        or extension.get("cell") != cell
        or identity.get("config_sha256") != canonical(config)
        or identity.get("data", {}).get("signal_routes_preprocessing_sha256")
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


def main() -> None:
    """Create an immutable human approval receipt after reviewing measured evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--max-host-ram-gib", type=float, required=True)
    parser.add_argument("--max-vram-gib-per-rank", type=float, required=True)
    parser.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    if not args.approve or not args.reviewer.strip() or not args.note.strip():
        parser.error("explicit --approve, named reviewer, and review note are required")
    if not math.isfinite(args.max_host_ram_gib) or args.max_host_ram_gib <= 0:
        parser.error("--max-host-ram-gib must be finite and positive")
    if not math.isfinite(args.max_vram_gib_per_rank) or args.max_vram_gib_per_rank <= 0:
        parser.error("--max-vram-gib-per-rank must be finite and positive")
    if args.profile.is_symlink() or not args.profile.is_file():
        parser.error("--profile must name a regular non-symlink receipt")
    cell, _ = load_cell(args.cell)
    if cell["variant"] not in VARIANTS:
        parser.error("cell is outside the frozen Signal Routes panel")
    runtime_root = admit_runtime(args.runtime_root)
    profile, identity = _validate_profile(
        args.profile.resolve(strict=True),
        cell,
        runtime_root,
        args.max_host_ram_gib,
        args.max_vram_gib_per_rank,
    )
    receipt = dict(profile)
    receipt["approved"] = True
    receipt["approval"] = {
        "schema": "signal-routes.profile-approval.v1",
        "reviewer": args.reviewer,
        "note": args.note,
        "profile_sha256": digest(args.profile),
        "source_sha256": identity["source_sha256"],
        "cell": cell,
        "approved_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "max_host_ram_gib": args.max_host_ram_gib,
        "max_vram_gib_per_rank": args.max_vram_gib_per_rank,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.output, receipt)


if __name__ == "__main__":
    main()
