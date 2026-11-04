"""Queue matched Stage2 proposals or launch one independently reviewed successor fit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for candidate in (Path(__file__).resolve().parents[3] / "src", Path.cwd() / "src"):
    if (candidate / "wnn" / "multistage_search_reviewed_successor_fit.py").is_file():
        sys.path.insert(0, str(candidate))
        break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage2 = commands.add_parser("queue-stage2", help="prepare exact LR-multiplier continuations")
    stage2.add_argument("--run-dir", required=True, type=Path)
    stage2.add_argument("--parent-result", action="append", required=True, type=Path)
    stage2.add_argument("--lr-multiplier", action="append", required=True, type=float)
    launch = commands.add_parser("launch", help="launch one reviewed Stage2/3 fit")
    launch.add_argument("--run-dir", required=True, type=Path)
    launch.add_argument("--stage", required=True, type=int, choices=(2, 3))
    launch.add_argument("--trial-number", required=True, type=int)
    launch.add_argument("--review", required=True, type=Path)
    launch.add_argument("--review-sha256", required=True)
    launch.add_argument("--operator-root", required=True, type=Path)
    launch.add_argument("--guardian", type=Path)
    args = parser.parse_args()

    from wnn.multistage_search_reviewed_successor_fit import launch_reviewed_successor_fit
    from wnn.multistage_search_successor_queue import queue_stage2

    if args.command == "queue-stage2":
        result = queue_stage2(args.run_dir, args.parent_result, args.lr_multiplier)
    else:
        result = launch_reviewed_successor_fit(
            args.run_dir,
            args.stage,
            args.trial_number,
            args.review,
            args.review_sha256,
            args.operator_root,
            args.guardian,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
