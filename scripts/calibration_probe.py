"""Post-hoc scale calibration by group: how much does knowing class and/or label-run age improve pooled weighted Pearson?

Predictions p are multiplied by a scale fitted per group on one half of the sequences (w-weighted least squares through the
origin, a = sum(w p y) / sum(w p^2)) and scored with the official WP on the other half (and vice versa).  Groups: none, class,
age bucket (oracle: needs the labels), class x age.  Class is causally decodable from the first 500 rows; age is NOT available at
serving - the age rows are an upper bound on what an age estimate could give.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from crossfold_blend import load_fold  # noqa: E402
from grade_prediction_bank import score  # noqa: E402

from wnn.data import market_class  # noqa: E402

EDGES = np.array([0, 2, 8, 32, 128, 100000])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    split = ROOT / "research/campaign_2026-09-29/artifacts/splits" / f"fold{args.fold}.json"
    y, mask, _comp, models = load_fold(args.bank, args.fold, split, ROOT / "cache/valid_targets.npz")
    by_name = {name: (array, order) for name, array, order in models.values()}
    array, order = by_name[args.run]
    p = np.asarray(array[order], dtype=np.float32)
    n, t, _ = y.shape
    yc = np.clip(y, -2, 2)
    cls = np.array([market_class(y[i][mask[i]]) for i in range(n)])
    changed = np.zeros((n, t), bool)
    changed[:, 1:] = (y[:, 1:] != y[:, :-1]).any(-1)
    age = np.zeros((n, t), np.int32)
    for i in range(1, t):
        age[:, i] = np.where(changed[:, i], 0, age[:, i - 1] + 1)
    bucket = np.digitize(age, EDGES[1:-1])
    w = np.abs(yc) * mask[..., None]
    rng = np.random.default_rng(0)
    base = score(y, mask, p)["wp"]
    print(f"run {args.run} fold {args.fold}: baseline WP {base:.5f}")
    for name, group in (("class", lambda: cls[:, None] + 0 * bucket), ("age bucket (oracle)", lambda: bucket),
                        ("class x age (oracle)", lambda: cls[:, None] * 8 + bucket)):
        gains = []
        for rep in range(4):
            half = rng.random(n) < 0.5
            g = group()
            out = p.copy()
            for fit_mask, apply_mask in ((half, ~half), (~half, half)):
                for key in np.unique(g):
                    sel_fit = (g == key) & fit_mask[:, None]
                    num = float((w * yc * p * sel_fit[..., None]).sum())
                    den = float((w * p * p * sel_fit[..., None]).sum())
                    a = num / den if den > 0 else 1.0
                    sel_apply = (g == key) & apply_mask[:, None]
                    out[sel_apply] = p[sel_apply] * a
            gains.append(score(y, mask, out)["wp"] - base)
        print(f"  scale by {name:<22} mean WP change {np.mean(gains):+.5f}  (sd over splits {np.std(gains):.5f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
