"""Compare two K-fold model families on their own held-out chain folds.

    scripts/limited .venv/bin/python scripts/fold_compare.py PREFIX_A PREFIX_B 0,1 [--lam 0,0.02]
For each fold k, both cache/pred_<PREFIX>_f<k>_valid.npy are scored on the validation sequences of
chain fold k only (the fold each model did not train on); reports per-fold and pooled deltas with a
chain bootstrap.
"""

from __future__ import annotations

import sys

import numpy as np

import decide


def main():
    a, b, folds = sys.argv[1], sys.argv[2], [int(f) for f in sys.argv[3].split(",")]
    lams = [float(v) for v in (sys.argv[5] if len(sys.argv) > 5 else "0,0.02").split(",")]
    chain, fold = decide._chains()
    sa = np.zeros((len(chain), 2, 2, 6))
    sb = np.zeros_like(sa)
    keep = np.zeros(len(chain), bool)
    for k in folds:
        idx = np.flatnonzero(fold == k)
        sa[idx] = decide.load_sums(f"{a}_f{k}")[idx]
        sb[idx] = decide.load_sums(f"{b}_f{k}")[idx]
        keep[idx] = True
    rng = np.random.default_rng(0)
    for lam in lams:
        parts = []
        for k in folds + ["pooled"]:
            idx = np.flatnonzero(fold == k) if k != "pooled" else np.flatnonzero(keep)
            d = decide.wp(sa[idx], lam) - decide.wp(sb[idx], lam)
            boots = [decide.wp(sa[r], lam) - decide.wp(sb[r], lam) for r in (decide._resample(idx, rng, chain) for _ in range(200))]
            parts.append(f"{k}: {d:+.4f}±{np.std(boots):.4f} ({decide.wp(sa[idx], lam):.4f} vs {decide.wp(sb[idx], lam):.4f})")
        print(f"{a} - {b} lam={lam}: " + "  ".join(parts))


if __name__ == "__main__":
    main()
