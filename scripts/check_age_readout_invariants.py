"""Run bounded synthetic parity checks for age features and a fresh head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wnn.training.age_readout_features import validate_age_invariants
from wnn.training.age_readout_model import zero_head_parity
from wnn.training.age_readout_tests import run_delivery_invariants


def main() -> None:
    """Verify model, feature, checkpoint, ordering and evaluation invariants."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmp-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "zero_head_parent_parity": zero_head_parity(),
                "age_invariants": validate_age_invariants(),
                "delivery_invariants": run_delivery_invariants(args.tmp_root),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
