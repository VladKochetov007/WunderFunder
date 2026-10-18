"""Replay the unchanged TL3 ZIP on source-qualified F0 rows without labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from f0_receipt import write_replay_receipts
from f0_replay import replay_groups
from f0_source import validate_f0_source


def parse_args() -> argparse.Namespace:
    """Parse bounded replay paths and optional profile/reset controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--output-sidecar", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--reset-check", action="store_true")
    parser.add_argument("--projection-rows", type=int, default=39_400_000)
    parser.add_argument("--resource-profile", default="unspecified")
    return parser.parse_args()


def validate_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    """Resolve immutable inputs and refuse to overwrite prior replay outputs."""
    repo_root = args.repo_root.resolve(strict=True)
    archive_path = args.zip.resolve(strict=True)
    split_path = args.split.resolve(strict=True)
    scratch = args.scratch.resolve(strict=True)
    predictions_path = args.output_predictions.resolve()
    sidecar_path = args.output_sidecar.resolve()
    receipt_path = args.output_receipt.resolve()
    if not scratch.is_dir():
        raise NotADirectoryError(scratch)
    if args.max_groups is not None and args.max_groups <= 0:
        raise ValueError("--max-groups must be positive")
    if args.projection_rows <= 0:
        raise ValueError("--projection-rows must be positive")
    if any(path.exists() for path in (predictions_path, sidecar_path, receipt_path)):
        raise FileExistsError("refusing to replace a prior prediction cache or receipt")
    return (
        repo_root,
        archive_path,
        split_path,
        scratch,
        predictions_path,
        sidecar_path,
        receipt_path,
    )


def main() -> None:
    """Validate source identity, run the exact callback, and bind receipts."""
    args = parse_args()
    paths = validate_paths(args)
    repo_root, archive_path, split_path, scratch, predictions_path, sidecar_path, receipt_path = (
        paths
    )
    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root))
    source = validate_f0_source(repo_root, split_path, args.max_groups)
    result = replay_groups(
        source,
        archive_path,
        scratch,
        predictions_path,
        args.reset_check,
    )
    report = write_replay_receipts(
        source,
        result,
        archive_path,
        split_path,
        predictions_path,
        sidecar_path,
        receipt_path,
        args.projection_rows,
        args.resource_profile,
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "groups": report["evaluation_groups_replayed"],
                "rows": report["rows_total"],
                "callback_seconds": report["timing"]["callback_only_seconds"],
                "max_rss_kib": report["memory_max_rss_kib"],
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
