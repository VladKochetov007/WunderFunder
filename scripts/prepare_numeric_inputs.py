from __future__ import annotations

import argparse
import json
from pathlib import Path

from wnn.training.sequence_numeric_preprocessing import (
    DEFAULT_CALIBRATION_SEED,
    DEFAULT_RESERVOIR_ROWS,
    prepare_numeric_artifact,
)


def main() -> int:
    """Create an immutable fitting-only numeric preprocessing artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reservoir-rows", type=int, default=DEFAULT_RESERVOIR_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_CALIBRATION_SEED)
    args = parser.parse_args()
    artifact = prepare_numeric_artifact(
        args.data_root,
        args.split_manifest,
        args.output,
        reservoir_rows=args.reservoir_rows,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "preparation_hash": artifact["preparation_hash"],
                "fit_sequences": len(artifact["fit_members"]),
                "rows": artifact["preparation_identity"]["full_scanned_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
