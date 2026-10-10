"""Class-conditional causal smoothing: does a per-class EMA of the output raise pooled WP?

Class is decodable from the first 500 rows (99.4%), so a class-dependent time constant is causal.  This probe uses the label-derived class
as a stand-in for the decoder.  For each EMA coefficient a the script reports per-class WP, then the pooled WP when each class uses
its own best a (selected on half of the sequences, scored on the other half).
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

ALPHAS = (1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12)


def ema(p: np.ndarray, a: float) -> np.ndarray:
    if a >= 1.0:
        return p
    out = np.empty_like(p)
    out[:, 0] = p[:, 0]
    for t in range(1, p.shape[1]):
        out[:, t] = (1 - a) * out[:, t - 1] + a * p[:, t]
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
    n = len(y)
    cls = np.array([market_class(y[i][mask[i]]) for i in range(n)])
    smoothed = {a: ema(p, a) for a in ALPHAS}
    print(f"run {args.run} fold {args.fold}: per-class WP by EMA coefficient a (a=1 is no smoothing)")
    table = {}
    for a, q in smoothed.items():
        row = []
        for c in range(4):
            idx = np.where(cls == c)[0]
            table[(a, c)] = score(y[idx], mask[idx], q[idx])["wp"]
            row.append(table[(a, c)])
        print(f"  a={a:<5} " + "  ".join(f"c{c} {v:.4f}" for c, v in enumerate(row)) + f"   pooled {score(y, mask, q)['wp']:.5f}")
    best = {c: max(ALPHAS, key=lambda a: table[(a, c)]) for c in range(4)}
    print("best a per class (in-sample, optimistic):", best)
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(6):
        half = rng.random(n) < 0.5
        out = p.copy()
        for fit, app in ((half, ~half), (~half, half)):
            for c in range(4):
                fit_idx = np.where((cls == c) & fit)[0]
                app_idx = np.where((cls == c) & app)[0]
                if len(fit_idx) == 0 or len(app_idx) == 0:
                    continue
                a_best = max(ALPHAS, key=lambda a: score(y[fit_idx], mask[fit_idx], smoothed[a][fit_idx])["wp"])
                out[app_idx] = smoothed[a_best][app_idx]
        gains.append(score(y, mask, out)["wp"] - score(y, mask, p)["wp"])
    print(f"cross-fitted class-conditional EMA: pooled WP change {np.mean(gains):+.5f} (sd {np.std(gains):.5f} over 6 splits)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
