"""Seed-ensemble and cross-recipe blend report on one fold: how much does averaging seeds / adding a second recipe add?

Usage: seed_ensemble_report.py --bank BANK --fold 1 --group a=run1,run2,run3 --group b=run4,run5,run6
Prints single-run WP, seed-average WP per group, pairwise correlation of group averages and the 50/50 blend of two groups.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crossfold_blend import load_fold  # noqa: E402
from grade_prediction_bank import score  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--group", action="append", required=True, help="label=run1,run2,...")
    args = parser.parse_args()
    split = ROOT / "research/campaign_2026-09-29/artifacts/splits" / f"fold{args.fold}.json"
    y, mask, _comp, models = load_fold(args.bank, args.fold, split, ROOT / "cache/valid_targets.npz")
    by_name = {name: (array, order) for name, array, order in models.values()}
    averages = {}
    for spec in args.group:
        label, runs = spec.split("=")
        total, singles = None, []
        for run in runs.split(","):
            array, order = by_name[run]
            p = np.asarray(array[order], dtype=np.float32)
            singles.append(score(y, mask, p)["wp"])
            total = p if total is None else total + p
            del p
        average = total / len(singles)
        averages[label] = average
        print(f"{label}: singles {[round(s, 5) for s in singles]} mean {np.mean(singles):.5f} sd {np.std(singles, ddof=1):.5f} "
              f"| seed-average of {len(singles)}: {score(y, mask, average)['wp']:.5f}")
    for a, b in itertools.combinations(averages, 2):
        sel = mask[..., None] & np.ones_like(averages[a], dtype=bool)
        corr = float(np.corrcoef(averages[a][sel], averages[b][sel])[0, 1])
        blend = 0.5 * (averages[a] + averages[b])
        print(f"{a} vs {b}: prediction correlation {corr:.4f}; 50/50 blend WP {score(y, mask, blend)['wp']:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
