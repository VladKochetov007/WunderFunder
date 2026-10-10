"""Causal output post-processing probes on saved predictions (no retraining, serving-side cost: a few multiply-adds per row).

Tested families, all strictly causal (row t uses predictions up to t only):
  ema        p_s[t] = (1-a) p_s[t-1] + a p[t]                        (low-pass of the output)
  hold       keep the previous output sign unless the new prediction disagrees with magnitude above m0*(1+k*log1p(age));
             output sign * |p[t]|                                  (decreasing-hazard sign hysteresis)
  holdema    hold, then ema of the result
Score: official global_wp_clip2_v1 on the fold; also per class.  Ideas from external ideators (see external_ideation_20261010).
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


def ema(p: np.ndarray, a: float) -> np.ndarray:
    out = np.empty_like(p)
    out[:, 0] = p[:, 0]
    for t in range(1, p.shape[1]):
        out[:, t] = (1 - a) * out[:, t - 1] + a * p[:, t]
    return out


def hold(p: np.ndarray, m0: float, k: float) -> np.ndarray:
    n, steps, c = p.shape
    out = np.empty_like(p)
    sign = np.sign(p[:, 0]) + (p[:, 0] == 0)
    age = np.zeros((n, c), dtype=np.float32)
    out[:, 0] = p[:, 0]
    for t in range(1, steps):
        row = p[:, t]
        disagree = (np.sign(row) != sign) & (row != 0)
        margin = m0 * (1 + k * np.log1p(age))
        flip = disagree & (np.abs(row) > margin)
        sign = np.where(flip, np.sign(row), sign)
        age = np.where(flip, 0.0, age + 1.0)
        out[:, t] = sign * np.abs(row)
    return out


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
    base = score(y, mask, p)["wp"]
    print(f"run {args.run} fold {args.fold}: baseline WP {base:.5f}")
    results = []
    for a in (0.8, 0.6, 0.4, 0.25, 0.15, 0.08):
        results.append((score(y, mask, ema(p, a))["wp"], f"ema a={a}"))
    for m0, k in itertools.product((0.0, 0.05, 0.1, 0.2), (0.0, 0.5, 1.0)):
        if m0 == 0.0 and k > 0:
            continue
        results.append((score(y, mask, hold(p, m0, k))["wp"], f"hold m0={m0} k={k}"))
    for wp, name in sorted(results, reverse=True):
        print(f"{wp:.5f} {wp - base:+.5f}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
