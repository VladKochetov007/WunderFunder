from __future__ import annotations

import argparse
import json
from pathlib import Path

from recovery_checks import digest, read


def spec(path: Path) -> dict[str, str]:
    """Pin native ROOT-selected files, preserving their source and original paths."""
    return {"path": str(path.absolute()), "sha256": digest(path)}


def main() -> None:
    """Freeze explicit recovery from the two original stoppedS1E8 states; launch nothing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--legacy-control", type=Path, required=True)
    parser.add_argument("--new-source-root", type=Path, required=True)
    parser.add_argument("--new-source-manifest", type=Path, required=True)
    parser.add_argument("--patch-receipt", type=Path, required=True)
    parser.add_argument("--root-cleanup-proof", type=Path, required=True)
    parser.add_argument("--observed-worker", type=Path, required=True)
    parser.add_argument("--approved-profile-101001", type=Path, required=True)
    parser.add_argument("--approved-profile-101002", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runtime, tools, legacy = (
        args.runtime.absolute(),
        args.tools.absolute(),
        args.legacy_control.absolute(),
    )
    control = runtime / "control/gru-monitor-recovery-r2"
    cells = []
    for seed, gpu in ((101001, 0), (101002, 1)):
        parent = legacy / f"{seed}-transition-request.json"
        request = read(parent)
        request.update(
            new_output=str(runtime / f"runs/gru_s{seed}_monitor_opt_r2/fit"),
            new_source_root=str(args.new_source_root.absolute()),
            new_source_manifest=spec(args.new_source_manifest),
            source_patch_receipt=spec(args.patch_receipt),
            approved_profile=spec(getattr(args, f"approved_profile_{seed}")),
        )
        cells.append(
            {"seed": seed, "gpu_index": gpu, "parent_request": spec(parent), "request": request}
        )
    names = (
        "recovery_supervisor.py",
        "recovery_checks.py",
        "canonical_resume_probe.py",
        "run_probe.py",
        "migrate_monitor_checkpoint_s4.py",
        "transition_support_s4.py",
        "archive_safe_state_s4.py",
    )
    pins = [spec(tools / name) for name in names]
    pins += [spec(legacy / name) for name in ("supervisor.py", "supervisor_checks.py")]
    pins.append(spec(args.observed_worker))
    historical = [spec(path) for path in sorted(legacy.glob("owned-*.json"))]
    historical.append(spec(legacy / "supervisor-owner.json"))
    historical += [
        spec(runtime / f"receipts/owned-fit-gru_s{seed}.json") for seed in (101001, 101002)
    ]
    value = {
        "schema": "wnn.gru-monitor-recovery.v2",
        "approved": True,
        "reviewer": "ROOT",
        "runtime": str(runtime),
        "python": str(args.python.absolute()),
        "tools": str(tools),
        "control": str(control),
        "observed_worker": str(args.observed_worker.absolute()),
        "root_cleanup_proof": spec(args.root_cleanup_proof),
        "historical_owned_receipts": historical,
        "tool_pins": pins,
        "cells": cells,
        "scientific_fit_count_increment": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps({"plan": str(args.output), "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
