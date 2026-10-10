"""Per-column coarse level codebook (clusters of near-duplicate values) from TRAIN rows only, for the --snap-codebook input probe.

A column is snapped only if its values collapse to at most --max-levels clusters at gap tolerance --tol (else it stays continuous).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--max-levels", type=int, default=120)
    ap.add_argument("--max-jitter-sd", type=float, default=None, help="keep a column only if the sd of value-minus-centre is below this (pure lattice columns)")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    pf = pq.ParquetFile(ROOT / "datasets/train.parquet")
    X = np.vstack([np.column_stack([pf.read_row_group(g, columns=list(FEATURE_COLUMNS), use_threads=False)[c].to_numpy() for c in FEATURE_COLUMNS])[:8000].astype(np.float64) for g in range(0, 600, 12)])
    book = {}
    for j in range(X.shape[1]):
        u, c = np.unique(X[:, j], return_counts=True)
        lab = np.concatenate([[0], np.cumsum(np.diff(u) > a.tol)])
        if lab[-1] + 1 > a.max_levels or lab[-1] + 1 < 2:
            continue
        centers = np.bincount(lab, weights=u * c) / np.bincount(lab, weights=c)
        if a.max_jitter_sd is not None:
            jit_sd = np.sqrt((c * (u - centers[lab]) ** 2).sum() / c.sum())
            if jit_sd > a.max_jitter_sd:
                continue
        book[f"c{j}"] = centers.astype(np.float32)
    np.savez(a.output, **book)
    print(f"snapped columns: {len(book)} of {X.shape[1]}; levels per column: {[len(v) for v in book.values()][:30]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
