"""Measure the feature-only lower bound for the frozen online relationship panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wnn.training.online_relationship_profile import (
    SCRATCH_ROOT,
    profile_feature_floor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--samples-per-fold", type=int, default=8)
    parser.add_argument("--backend", choices=("reference", "compiled"), default="reference")
    args = parser.parse_args()
    output = args.output or SCRATCH_ROOT / "online-relationships-20261022" / "feature-profile.json"
    report = profile_feature_floor(
        args.recipe,
        output,
        root=args.root,
        samples_per_fold=args.samples_per_fold,
        backend=args.backend,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
