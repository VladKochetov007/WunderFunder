"""Prepare frozen parent caches or run the admitted online readout panel."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """Run one explicit cache-preparation or admitted fit command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("WNN_PROJECT_ROOT", ROOT)),
        help="project checkout containing the frozen recipe, data and artifacts",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-cache", help="build immutable parent/readout caches")
    prepare.add_argument("--recipe", type=Path, required=True)
    prepare.add_argument("--fold", choices=("0", "1", "all"), default="all")
    prepare.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    prepare.add_argument("--cache-dir", type=Path)
    prepare.add_argument("--sample-fit", type=int)
    prepare.add_argument("--sample-eval", type=int)
    fit = commands.add_parser(
        "fit-panel", help="fit all twelve cells from complete validated caches"
    )
    fit.add_argument("--recipe", type=Path, required=True)
    fit.add_argument("--device", choices=("cpu", "cuda"), required=True)
    profile = commands.add_parser(
        "profile-panel", help="profile twelve sampled fits and project full-panel costs"
    )
    profile.add_argument("--recipe", type=Path, required=True)
    profile.add_argument("--output-dir", type=Path, required=True)
    profile.add_argument("--sample-fit", type=int, default=4)
    profile.add_argument("--sample-eval", type=int, default=2)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from wnn.training.online_relationship_data import build_parent_caches
    from wnn.training.online_relationship_delivery import run_online_relationship_panel
    from wnn.training.online_relationship_panel_profile import profile_online_relationship_panel

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "prepare-cache":
        if args.fold == "all" and args.cache_dir is not None:
            parser.error("--cache-dir applies to one fold; select --fold 0 or --fold 1")
        folds = (0, 1) if args.fold == "all" else (int(args.fold),)
        for fold in folds:
            record = build_parent_caches(
                args.recipe,
                fold,
                root=root,
                device=args.device,
                cache_dir=args.cache_dir,
                sample_fit=args.sample_fit,
                sample_eval=args.sample_eval,
            )
            print(json.dumps(record, sort_keys=True), flush=True)
        return 0
    if args.command == "profile-panel":
        result = profile_online_relationship_panel(
            args.recipe,
            args.output_dir,
            root=root,
            sample_fit=args.sample_fit,
            sample_eval=args.sample_eval,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return 0
    result = run_online_relationship_panel(args.recipe, root=root, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
