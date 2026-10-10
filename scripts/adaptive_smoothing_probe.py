"""Change-detecting causal smoothing: EMA that resets to the raw output when the output jumps by more than tau.

out_t = p_t if |p_t - out_{t-1}| > tau (per target) else (1-a) out_{t-1} + a p_t.  Oracle boundaries gain +0.025 WP in class 3 (gate_atom_probe),
so this tests how much of that a causal self-detected boundary recovers.  Reported per class and pooled, for a grid of (a, tau).
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


def adaptive(p: np.ndarray, a: float, tau: float) -> np.ndarray:
    out = np.empty_like(p)
    out[:, 0] = p[:, 0]
    for t in range(1, p.shape[1]):
        prev = out[:, t - 1]
        jump = np.abs(p[:, t] - prev) > tau
        out[:, t] = np.where(jump, p[:, t], (1 - a) * prev + a * p[:, t])
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
    c3 = np.where(cls == 3)[0]
    base3 = score(y[c3], mask[c3], p[c3])["wp"]
    scale = float(np.std(p[mask]))
    print(f"run {args.run} fold {args.fold}: class-3 baseline WP {base3:.4f}; output std {scale:.3f}")
    for a in (0.5, 0.3, 0.15):
        for tau_rel in (0.1, 0.25, 0.5, 1.0):
            q = adaptive(p[c3], a, tau_rel * scale)
            wp3 = score(y[c3], mask[c3], q)["wp"]
            print(f"  a={a} tau={tau_rel:.2f}*std  class-3 WP {wp3:.4f} ({wp3 - base3:+.4f}; pooled effect about {0.5 * (wp3 - base3):+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
