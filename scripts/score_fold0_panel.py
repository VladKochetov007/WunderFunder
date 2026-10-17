"""Backward-compatible CLI for the fixed fold-0 evaluation panel.

    python scripts/score_fold0_panel.py RUN[@epN] [--epoch N]
        [--reference RUN@epN ...] --out grade.json

New source-qualified folds should use ``score_fold_panel.py`` with explicit
``--split`` and ``--expected-fold`` arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from score_fold_panel import BOOTSTRAP_REPLICATES, score_panel  # noqa: E402
from score_fold_panel import _blend as _blend  # noqa: E402, F401
from score_fold_panel import _member as _member  # noqa: E402, F401

DEFAULT_SPLIT = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
EVALUATION_GROUP_COUNT = 468


def main() -> None:
    """Run the generic scorer with the historical fold-0 contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="RUN, RUN@epN, or a fixed blend RUN@epN*W,...")
    parser.add_argument("--reference", action="append", default=[], help="RUN@epN or blend spec")
    parser.add_argument("--split", type=Path, default=ROOT / DEFAULT_SPLIT)
    parser.add_argument("--epoch", type=int, help="default epoch for an unqualified run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()

    report = score_panel(
        args.run,
        references=args.reference,
        split_path=args.split,
        expected_fold=0,
        output_path=args.out,
        n_boot=args.n_boot,
        expected_group_count=EVALUATION_GROUP_COUNT,
        candidate_epoch=args.epoch if args.epoch is not None else 1,
        exposure_description=(
            "fold0 excluded-468 panel; repeatedly inspected development evidence, not a public forecast"
        ),
        include_fold_metadata=False,
    )
    print(json.dumps({"candidate": args.run, "wp": report["official"]["wp"]}))


if __name__ == "__main__":
    main()
