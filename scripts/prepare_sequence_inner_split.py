"""Prepare a deterministic component-clean inner split from one outer fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wnn.sequence_inner_splits import prepare_sequence_inner_split


def main() -> None:
    """Validate an outer manifest and publish its immutable inner split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41001)
    parser.add_argument("--evaluation-fraction", type=float, default=0.2)
    args = parser.parse_args()
    result = prepare_sequence_inner_split(
        args.parent_manifest,
        args.data_root,
        args.output,
        seed=args.seed,
        evaluation_fraction=args.evaluation_fraction,
    )
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
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
