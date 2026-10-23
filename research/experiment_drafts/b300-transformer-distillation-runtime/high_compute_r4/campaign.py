"""Initialize, materialize, compare, and manually select R4 campaign cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from campaign_io import REPO_ROOT, create_campaign, load_campaign, require
from failure_receipts import record_failure
from materialize import SELECTABLE_STAGES, STAGES, get_cell, prepare_stage
from results import compare_stage, select_variant


def _parser() -> argparse.ArgumentParser:
    """Build the staged campaign CLI without importing training code."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    initialize = subparsers.add_parser(
        "init", help="bind immutable runtime, recipe, and split inputs"
    )
    initialize.add_argument("--campaign-root", type=Path, required=True)
    initialize.add_argument("--outer-f0", type=Path, required=True)
    initialize.add_argument("--outer-f2", type=Path, required=True)
    initialize.add_argument("--inner-f0", type=Path, required=True)
    initialize.add_argument("--inner-f2", type=Path, required=True)
    initialize.add_argument("--data-root", type=Path, default=REPO_ROOT)

    prepare = subparsers.add_parser(
        "prepare", help="materialize one conditional stage; never runs fits"
    )
    prepare.add_argument("--campaign-root", type=Path, required=True)
    prepare.add_argument("--stage", choices=STAGES, required=True)

    selection = subparsers.add_parser("select", help="record a manual B-F recipe choice")
    selection.add_argument("--campaign-root", type=Path, required=True)
    selection.add_argument("--stage", choices=SELECTABLE_STAGES, required=True)
    selection.add_argument("--variant", required=True)
    selection.add_argument("--reason", required=True)

    compare = subparsers.add_parser(
        "compare", help="compare fixed endpoints without choosing a winner"
    )
    compare.add_argument("--campaign-root", type=Path, required=True)
    compare.add_argument("--stage", choices=STAGES, required=True)

    failed = subparsers.add_parser(
        "record-failure", help="bind one terminal failed attempt to logs and source"
    )
    failed.add_argument("--campaign-root", type=Path, required=True)
    failed.add_argument("--stage", choices=STAGES, required=True)
    failed.add_argument("--cell", required=True)
    failed.add_argument("--reason", required=True)
    failed.add_argument("--evidence", type=Path, action="append", required=True)
    failed.add_argument(
        "--phase",
        choices=(
            "profile_one",
            "profile_eight",
            "profile_peak",
            "admission",
            "fit",
            "evaluate",
        ),
        default="fit",
    )
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one metadata-only campaign operation."""
    if args.command == "init":
        create_campaign(
            args.campaign_root,
            args.data_root,
            {0: args.outer_f0, 2: args.outer_f2},
            {0: args.inner_f0, 2: args.inner_f2},
        )
        campaign = load_campaign(args.campaign_root)
        return {
            "campaign_root": str(args.campaign_root.expanduser().resolve()),
            "campaign_sha256": campaign["campaign_sha256"],
            "runtime_source_sha256": campaign["runtime_source_sha256"],
            "automatic_fit": False,
        }
    if args.command == "prepare":
        return prepare_stage(args.campaign_root, args.stage)
    if args.command == "select":
        return select_variant(args.campaign_root, args.stage, args.variant, args.reason)
    if args.command == "compare":
        return compare_stage(args.campaign_root, args.stage)
    if args.command == "record-failure":
        campaign = load_campaign(args.campaign_root)
        cell = get_cell(args.campaign_root, args.stage, args.cell)
        require(
            not cell.get("reused"), "cannot record a new attempt against an exact reused endpoint"
        )
        return record_failure(
            args.campaign_root, campaign, cell, args.reason, args.evidence, args.phase
        )
    raise ValueError(f"unsupported command: {args.command}")


def main() -> None:
    """Run one safe metadata operation and report validation errors cleanly."""
    parser = _parser()
    args = parser.parse_args()
    try:
        result = _run(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
