from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from impact81.bindings import admit_runtime, load_cell
from impact81.runtime_family import (
    environment_for_card,
    family_for_card,
    runtime_commit_for_card,
)


def main() -> None:
    """Bootstrap the pinned runtime before importing any model code; receiving host only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "check", "profile", "fit", "export", "score"))
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--preprocessing", type=Path)
    parser.add_argument("--profile-receipt", type=Path)
    parser.add_argument("--profile-updates", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--checks-receipt", type=Path)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("model/data stages are restricted to the receiving Linux server")
    if not args.owner.strip() or not 1 <= args.timeout_seconds <= 172800:
        parser.error("named owner and bounded timeout (1..172800 seconds) required")
    cell, native_config = load_cell(args.cell)
    family = family_for_card(cell["card"])
    for name, value in environment_for_card(cell["card"]).items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    runtime_root = admit_runtime(args.runtime_root, runtime_commit_for_card(cell["card"]))
    if args.mode == "prepare":
        from wnn.training.sequence_config import load_config
        from wnn.training.sequence_data import build_data

        data = build_data(load_config(native_config), args.data_root, args.split_manifest)
        if family == "transformer":
            from impact81.preprocessing import prepare

            prepare(data, args.output)
        else:
            from impact81.preparation_later import prepare

            prepare(data, args.output, cell)
    elif args.mode == "check":
        if args.preprocessing is None:
            parser.error("check requires --preprocessing")
        if family == "transformer":
            from impact81.receiver_checks import run_checks

            run_checks(args.preprocessing, args.output, runtime_root)
        else:
            from impact81.receiver_later import run_checks

            run_checks(args.preprocessing, args.output, runtime_root)
    elif args.mode == "score":
        from wnn.training.sequence_score import score_export

        if args.predictions is None:
            parser.error("score requires --predictions")
        score_export(args.data_root, args.split_manifest, args.predictions, args.output)
    elif args.mode == "export":
        from impact81.exporting import export_endpoint

        if args.checkpoint is None:
            parser.error("export requires --checkpoint")
        export_endpoint(
            args.checkpoint, args.data_root, args.split_manifest, args.output, runtime_root
        )
    else:
        from impact81.fitting import run

        if args.preprocessing is None:
            parser.error("profile/fit requires --preprocessing")
        if args.checks_receipt is None:
            parser.error("profile/fit requires --checks-receipt from server acceptance")
        if (args.mode == "profile") != (args.profile_updates > 0):
            parser.error("profile alone requires --profile-updates > 0")
        run(args, runtime_root)


if __name__ == "__main__":
    main()
