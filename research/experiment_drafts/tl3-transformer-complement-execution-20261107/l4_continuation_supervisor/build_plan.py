from __future__ import annotations

import argparse
import json
from pathlib import Path

from supervisor_checks import DEADLINE, digest, read


def spec(path: Path) -> dict[str, str]:
    """Freeze actual ROOT-selected native bytes without altering their contents."""
    return {"path": str(path.resolve(strict=True)), "sha256": digest(path)}


def main() -> None:
    """Prepare the concrete ROOT-authorized two-cell transition plan; launch or stop nothing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--observed-worker", type=Path, required=True)
    parser.add_argument("--first-profile-receipt", type=Path, required=True)
    parser.add_argument("--approved-profile-101001", type=Path, required=True)
    parser.add_argument("--approved-profile-101002", type=Path, required=True)
    parser.add_argument("--new-source-manifest", type=Path, required=True)
    parser.add_argument("--patch-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime, tools = args.runtime.resolve(strict=True), args.tools.resolve(strict=True)
    cells = []
    for seed, gpu in ((101001, 0), (101002, 1)):
        old = runtime / f"runs/gru_s{seed}/fit"
        profile = getattr(args, f"approved_profile_{seed}")
        if read(profile).get("approved") is not True or read(profile).get("reviewer") != "ROOT":
            raise ValueError("both approved profiles must be ROOT-reviewed before plan creation")
        request = {
            "schema": "wnn.gru-monitor-transition-request.v1",
            "approved": True,
            "reviewer": "ROOT",
            "old_output": str(old),
            "new_output": str(runtime / f"runs/gru_s{seed}_monitor_opt/fit"),
            "old_metadata": spec(old / "run_metadata.json"),
            "new_config": spec(runtime / f"configs/gru-monitor-continuation-r3/gru_s{seed}.json"),
            "approved_profile": spec(profile),
            "old_source_root": str(runtime / "source-r1"),
            "new_source_root": str(runtime / "source-gru-monitor-archive-r1"),
            "old_source_manifest": spec(runtime / "manifests/l4-source-r1-manifest.json"),
            "new_source_manifest": spec(args.new_source_manifest),
            "source_patch_receipt": spec(args.patch_receipt),
            "first_profile_receipt": spec(args.first_profile_receipt),
            "not_after_epoch": DEADLINE,
        }
        cells.append(
            {
                "seed": seed,
                "gpu_index": gpu,
                "old_owned_receipt": str(runtime / f"receipts/owned-fit-gru_s{seed}.json"),
                "request": request,
            }
        )
    pins = [
        spec(tools / name)
        for name in (
            "migrate_monitor_checkpoint.py",
            "transition_support.py",
            "archive_safe_state.py",
            "supervisor.py",
            "supervisor_checks.py",
            "preflight.py",
        )
    ]
    pins.append(spec(args.observed_worker))
    value = {
        "schema": "wnn.gru-monitor-supervisor-plan.v1",
        "approved": True,
        "reviewer": "ROOT",
        "runtime": str(runtime),
        "python": str(args.python.resolve(strict=True)),
        "tools": str(tools),
        "control": str(runtime / "control/gru-monitor-transition-r1"),
        "observed_worker": str(args.observed_worker.resolve(strict=True)),
        "tool_pins": pins,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
    print(
        json.dumps(
            {"plan": str(args.output), "sha256": digest(args.output), "deadline_epoch": DEADLINE}
        )
    )


if __name__ == "__main__":
    main()
