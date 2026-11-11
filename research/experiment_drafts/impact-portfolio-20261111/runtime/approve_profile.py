from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path

from impact81.bindings import (
    admit_runtime,
    canonical,
    load_cell,
    read_object,
    source_identity,
    write_new,
)


def main() -> None:
    """Record an operator's explicit profile decision; never infer permission or run a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--max-host-ram-gib", type=float, required=True)
    parser.add_argument("--max-vram-gib-per-rank", type=float, required=True)
    parser.add_argument("--approve", action="store_true", required=True)
    args = parser.parse_args()
    if not args.reviewer.strip() or not args.note.strip():
        parser.error("reviewer and an explicit resource/admission decision are required")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.max_host_ram_gib, args.max_vram_gib_per_rank)
    ):
        parser.error("positive resource limits are required")
    root = admit_runtime(args.runtime_root)
    cell, _ = load_cell(args.cell)
    receipt = read_object(args.profile)
    metadata = read_object(args.profile.parent / "run_metadata.json")
    identity = receipt.get("identity", {})
    config = metadata.get("config", {})
    current_source = source_identity(root)
    if (
        receipt.get("receipt_schema") != "wnn.sequence-profile.v1"
        or receipt.get("status") != "profile_complete"
        or receipt.get("approved") is not False
        or receipt.get("updates", 0) < 50
        or receipt.get("errors") != []
        or receipt.get("error_count") != 0
        or receipt.get("scientific_checkpoint_written") is not False
        or identity.get("world_size") != 2
        or identity.get("source_sha256") != current_source["sha256"]
        or identity.get("source_files") != current_source["files"]
        or identity.get("config_sha256") != canonical(config)
        or config.get("impact81", {}).get("cell") != cell
    ):
        raise ValueError("profile is not a completed source-bound production-shape experiment")
    reload = receipt.get("engineering_reload", {})
    checks = reload.get("by_rank", [])
    if (
        reload.get("status") != "passed"
        or len(checks) != 2
        or any(item.get("model_optimizer_scheduler_rng_equal") is not True for item in checks)
    ):
        raise ValueError("both ranks require the canonical engineering reload proof")
    memory = receipt.get("memory_by_rank", [])
    if len(memory) != 2 or any(
        any(
            not isinstance(item.get(key), (int, float))
            or not math.isfinite(item[key])
            or item[key] <= 0
            for key in ("peak_reserved_bytes", "host_peak_rss_bytes")
        )
        or item["peak_reserved_bytes"] > args.max_vram_gib_per_rank * 2**30
        for item in memory
    ):
        raise ValueError("profile exceeds declared per-rank VRAM or lacks measurements")
    if sum(item.get("host_peak_rss_bytes", 0) for item in memory) > args.max_host_ram_gib * 2**30:
        raise ValueError("conservative sum of rank RSS exceeds declared host RAM")
    approved = receipt | {
        "approved": True,
        "reviewer": args.reviewer,
        "reviewed_at": dt.datetime.now(dt.UTC).isoformat(),
        "approval_note": args.note,
    }
    write_new(args.output, approved)


if __name__ == "__main__":
    main()
