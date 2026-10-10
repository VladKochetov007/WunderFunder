"""Linear lag structure between features and labels: how much of y_t is explained by raw features at row t+k, for k in a range.

Ridge regression (z-scored raw columns plus first differences) fitted on half of the sampled validation sequences and scored on the
other half with the official weighted Pearson (weights |clip(y,2)|), per lag k (negative = past, positive = future).  A pure
diagnostic of the information structure: a sharp peak at positive k means labels are delayed functions of the features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import FEATURE_COLUMNS  # noqa: E402


def wpearson(p: np.ndarray, y: np.ndarray) -> float:
    w = np.abs(y)
    sw = w.sum()
    pm, ym = (w * p).sum() / sw, (w * y).sum() / sw
    return float((w * (p - pm) * (y - ym)).sum() / np.sqrt((w * (p - pm) ** 2).sum() * (w * (y - ym) ** 2).sum()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, default=48)
    parser.add_argument("--lags", type=int, nargs="*", default=[-64, -16, -4, -1, 0, 1, 2, 4, 8, 16, 32, 64, 128])
    args = parser.parse_args()
    pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    targets = ["t0", "t1"]
    step = max(1, pf.num_row_groups // args.sequences)
    seqs = []
    for g in list(range(0, pf.num_row_groups, step))[: args.sequences]:
        t = pf.read_row_group(g, columns=[*FEATURE_COLUMNS, *targets], use_threads=False)
        x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        y = np.column_stack([t[c].to_numpy() for c in targets]).astype(np.float32)
        seqs.append((x, np.clip(y, -2, 2)))
    mu = np.mean([s[0].mean(0) for s in seqs], 0)
    sd = np.mean([s[0].std(0) for s in seqs], 0) + 1e-6
    half = len(seqs) // 2
    print(f"{len(seqs)} sequences; fit on first {half}, score on rest; official-style weighted Pearson per target (t0, t1), ridge on 112 z-scored columns + first differences")
    for k in args.lags:
        def design(group, k=k):
            xs, ys = [], []
            for x, y in group:
                z = (x - mu) / sd
                d = np.vstack([np.zeros((1, z.shape[1]), np.float32), np.diff(z, axis=0)])
                f = np.hstack([z, d])
                lo, hi = max(99, -k), len(f) - max(0, k)
                xs.append(f[lo + k : hi + k])
                ys.append(y[lo:hi])
            return np.vstack(xs), np.vstack(ys)

        xa, ya = design(seqs[:half])
        xb, yb = design(seqs[half:])
        a = xa.T @ xa + 300.0 * np.eye(xa.shape[1], dtype=np.float32)
        wts = np.linalg.solve(a, xa.T @ ya)
        pb = xb @ wts
        print(f"lag {k:>5}: WP t0 {wpearson(pb[:, 0], yb[:, 0]):.4f}  t1 {wpearson(pb[:, 1], yb[:, 1]):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
