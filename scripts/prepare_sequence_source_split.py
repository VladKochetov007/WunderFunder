"""Prepare train-only fitting and full-valid evaluation on the receiving worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wnn.sequence_source_splits import prepare_sequence_source_split


def main() -> None:
    """Build full feature-only graph/index inputs and publish an immutable split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = prepare_sequence_source_split(args.data_root, args.output, args.workers)
    print(
        json.dumps(
            {
                "manifest": str(result.path),
                "sha256": result.sha256,
                "fold": result.fold,
                "n_folds": result.n_folds,
                "fit_train": len(result.train_groups),
                "fit_valid": len(result.valid_groups),
                "evaluation_valid": len(result.evaluation_groups),
                "calendar_chronology": "unverified",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
