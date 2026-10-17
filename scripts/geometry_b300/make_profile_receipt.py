"""Validate eight overlapping single-card profiles and seal their measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import PROFILE_CONFIG, write_json


def read_profiles(paths: list[Path]) -> list[dict[str, Any]]:
    """Load eight complete per-card profile receipts with one shared source."""
    if len(paths) != 8:
        raise ValueError("provide exactly eight per-card profile JSON files")
    profiles = [json.loads(path.read_text()) for path in paths]
    if any(item.get("schema") != "causal_geometry_b300_card_profile_v1" for item in profiles):
        raise ValueError("unsupported per-card profile receipt")
    if any(item.get("profile_config") != PROFILE_CONFIG for item in profiles):
        raise ValueError("per-card profiles do not use the frozen H512/271-input workload")
    if any(not item.get("strict_fp32") or not item.get("finite_profile") for item in profiles):
        raise ValueError("each profile must show strict FP32 and finite optimizer updates")
    for item in profiles:
        resources = item.get("resources", {})
        updates = item.get("updates", {})
        if (
            not item.get("timing")
            or int(resources.get("cuda_peak_allocated_bytes") or 0) <= 0
            or int(resources.get("cuda_peak_reserved_bytes") or 0) <= 0
            or int(resources.get("process_peak_rss_bytes") or 0) <= 0
            or not updates.get("valid_for_throughput")
        ):
            raise ValueError(
                "each card profile must contain timing, memory and finite-update evidence"
            )
    commits = {item.get("source_commit") for item in profiles}
    devices = {item.get("device", {}).get("gpu_id") for item in profiles}
    if len(commits) != 1 or None in commits or len(devices) != 8 or None in devices:
        raise ValueError("profiles must bind one source commit to eight unique GPU IDs")
    if any("B300" not in item["device"].get("name", "").upper() for item in profiles):
        raise ValueError("all eight visible accelerators must identify as B300")
    if max(item["started_epoch"] for item in profiles) >= min(
        item["finished_epoch"] for item in profiles
    ):
        raise ValueError("profile process windows do not overlap; run all eight concurrently")
    return profiles


def make_receipt(paths: list[Path]) -> dict[str, Any]:
    """Combine measured card evidence into the fit-admission receipt."""
    profiles = read_profiles(paths)
    cards = []
    for profile in profiles:
        resources = profile["resources"]
        updates = profile["updates"]
        cards.append(
            {
                "device": profile["device"],
                "timing": profile["timing"],
                "resources": resources,
                "updates": updates,
                "profile_log": profile["log"],
                "profile_log_sha256": profile["log_sha256"],
                "started_epoch": profile["started_epoch"],
                "finished_epoch": profile["finished_epoch"],
            }
        )
    return {
        "schema": "causal_geometry_b300_profile_v1",
        "source_commit": profiles[0]["source_commit"],
        "profile_config": PROFILE_CONFIG,
        "profiled_workers": len(cards),
        "overlap_confirmed": True,
        "strict_fp32": True,
        "cards": cards,
    }


def main() -> int:
    """Validate profile files and write a portable, source-bound receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = make_receipt(args.profile)
    write_json(args.output, receipt)
    print(json.dumps({"output": str(args.output), "source_commit": receipt["source_commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
