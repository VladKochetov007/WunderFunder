"""Review and admit one complete, source-matched 50-update engineering profile."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from bilinear_entmax.bindings import (
    FULL_SHAPE,
    FULL_SHAPE_ACCEPTANCE_SCHEMA,
    VARIANTS,
    admit_runtime,
    digest,
    load_cell,
    source_profile_sha256,
    write_new,
)
from bilinear_entmax.profile_approval import validate_profile


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
        parser.error("cell is outside the frozen Bilinear Entmax panel")
    runtime_root = admit_runtime(args.runtime_root)
    profile, identity = validate_profile(
        args.profile.resolve(strict=True),
        cell,
        runtime_root,
        args.max_host_ram_gib,
        args.max_vram_gib_per_rank,
    )
    receipt = dict(profile)
    source_profile_hash = source_profile_sha256(profile)
    if digest(args.profile) != source_profile_hash:
        raise ValueError("measured profile is not the canonical PR75 receipt byte sequence")
    receipt["full_shape_acceptance"] = {
        "schema": FULL_SHAPE_ACCEPTANCE_SCHEMA,
        "status": "passed",
        "required_before_fit": True,
        "human_approval_required": True,
        "human_approved": True,
        "reviewer": args.reviewer,
        "profile_updates": 50,
        "source_profile_sha256": source_profile_hash,
        "source_sha256": identity["source_sha256"],
        "cell": cell,
        "requested_shape": FULL_SHAPE,
        "observed_shape": FULL_SHAPE,
        "resource_limits": {
            "max_host_ram_gib": args.max_host_ram_gib,
            "max_vram_gib_per_rank": args.max_vram_gib_per_rank,
        },
        "observed_resources": {
            "host_peak_rss_bytes": profile["host_peak_rss_bytes"],
            "peak_reserved_by_rank_bytes": [
                item["peak_reserved_bytes"] for item in profile["memory_by_rank"]
            ],
            "device_total_memory_by_rank_bytes": profile["device_total_memory_bytes"],
        },
    }
    receipt["approved"] = True
    reviewed_receipt = dict(receipt)
    reviewed_receipt["approved"] = False
    reviewed_bytes = (
        json.dumps(
            reviewed_receipt,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    reviewed_profile_sha256 = hashlib.sha256(reviewed_bytes).hexdigest()
    receipt["approval"] = {
        "schema": "bilinear-entmax.profile-approval.v1",
        "reviewer": args.reviewer,
        "note": args.note,
        "profile_sha256": reviewed_profile_sha256,
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
