"""Validate and deliver only already completed R01 fits; this CLI has no fit mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from r01_recovery_common import read_json, sha256_file
from r01_recovery_delivery import export_existing_fit, grade_pair
from r01_recovery_fit import EXPECTED_PANEL_SHA256, validate_existing_fit


def _panel_job(panel_path: Path, job_id: str) -> dict[str, Any]:
    """Select one exact existing cell from the frozen panel."""
    if sha256_file(panel_path) != EXPECTED_PANEL_SHA256:
        raise ValueError("R01 panel differs from the frozen source panel")
    panel = read_json(panel_path)
    selected = [job for job in panel.get("jobs", []) if job.get("id") == job_id]
    if len(selected) != 1:
        raise ValueError(f"expected one frozen R01 panel job named {job_id}")
    return selected[0]


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add immutable project and recovery packet paths to one command."""
    parser.add_argument(
        "--root", type=Path, required=True, help="project root /data/epanov_s/tabdata"
    )
    parser.add_argument("--packet", type=Path, required=True, help="frozen runtime/b300-r01-v1")
    parser.add_argument(
        "--source-root", type=Path, required=True, help="frozen runtime/source-a266-lf-subset-v1"
    )
    parser.add_argument(
        "--recovery-root",
        type=Path,
        required=True,
        help="new isolated state/r01-delivery-recovery-v2 directory",
    )


def main() -> None:
    """Run one read-only fit validation, one fixed-endpoint export, or one paired grade."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    validate_parser = subparsers.add_parser(
        "validate", help="CPU validate one existing complete fit"
    )
    _common_arguments(validate_parser)
    validate_parser.add_argument("--job-id", required=True)
    export_parser = subparsers.add_parser("export", help="export a validated existing epoch-5 fit")
    _common_arguments(export_parser)
    export_parser.add_argument("--job-id", required=True)
    grade_parser = subparsers.add_parser(
        "grade", help="officially grade one same-seed wide/deep pair"
    )
    _common_arguments(grade_parser)
    grade_parser.add_argument("--wide-id", required=True)
    grade_parser.add_argument("--deep-id", required=True)
    args = parser.parse_args()
    panel = args.packet / "PANEL.json"
    if args.phase == "validate":
        result = validate_existing_fit(
            args.root,
            args.packet,
            args.source_root,
            args.recovery_root,
            _panel_job(panel, args.job_id),
        )
    elif args.phase == "export":
        _panel_job(panel, args.job_id)
        result = export_existing_fit(
            args.root,
            args.packet,
            args.source_root,
            args.recovery_root,
            args.job_id,
        )
    else:
        _panel_job(panel, args.wide_id)
        _panel_job(panel, args.deep_id)
        result = grade_pair(
            args.root,
            args.packet,
            args.source_root,
            args.recovery_root,
            args.wide_id,
            args.deep_id,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
