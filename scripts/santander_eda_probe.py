"""EDA inspired by the Santander transcript: value rounding/lattice, within- and cross-sequence value reuse, frequency-vs-target signal.

For each column: decimals needed, distinct-value share inside a sequence (of update rows), share of values that also occur in OTHER
sequences (cross-sequence reuse; the Santander 'fake row' signature), and the weighted correlation of the (log) global value count
with |y| and with the sign of y.  All read-only; prints a compact table.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from utils import FEATURE_COLUMNS  # noqa: E402


def load(path, groups, rows):
    pf = pq.ParquetFile(path)
    out = []
    for g in groups:
        t = pf.read_row_group(int(g), columns=list(FEATURE_COLUMNS), use_threads=False)
        out.append(np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS])[:rows].astype(np.float64))
    return out


def main() -> int:
    rng = np.random.default_rng(1)
    pt = pq.ParquetFile(ROOT / "datasets/train.parquet")
    pv = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    tr = load(ROOT / "datasets/train.parquet", rng.choice(pt.num_row_groups, 120, replace=False), 6000)
    va = load(ROOT / "datasets/valid.parquet", rng.choice(pv.num_row_groups, 40, replace=False), 6000)
    x0 = tr[0]
    dec = []
    for j in range(x0.shape[1]):
        v = np.unique(x0[:, j])[:200]
        d = next((k for k in range(0, 9) if np.allclose(np.round(v, k), v, atol=1e-12)), 9)
        dec.append(d)
    print("decimals per column (histogram):", dict(Counter(dec)))
    # cross-sequence reuse: values of valid seq column j found in any of 120 train sequences
    rows = []
    for j in range(x0.shape[1]):
        pool = set(np.concatenate([s[:, j] for s in tr]).tolist())
        within = np.mean([len(np.unique(s[:, j])) / len(s) for s in va])
        reuse = np.mean([np.mean(np.isin(s[:, j], list(pool))) for s in va[:10]])
        rows.append((j, dec[j], within, reuse))
    r = np.array(rows)
    print("distinct share within sequence: min/med/max", r[:, 2].min().round(4), np.median(r[:, 2]).round(4), r[:, 2].max().round(4))
    print("share of valid values that occur in 120 other train sequences: min/med/max", r[:, 3].min().round(3), np.median(r[:, 3]).round(3), r[:, 3].max().round(3))
    for lo, hi in ((0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 1.01)):
        m = (r[:, 3] >= lo) & (r[:, 3] < hi)
        print(f"  reuse in [{lo},{hi}): {int(m.sum())} columns, e.g. {r[m][:4, 0].astype(int).tolist()}")
    # sequence-level duplicates: any valid sequence sharing >=50% of rows with a train sequence?
    best = []
    for s in va[:15]:
        key = {tuple(row[:6]) for row in s[:2000]}
        best.append(max(len(key & {tuple(row[:6]) for row in t[:2000]}) / len(key) for t in tr))
    print("max row-prefix overlap (first 6 cols, 2000 rows) valid vs 120 train seqs:", np.round(best, 3).tolist())
    return 0


if __name__ == "__main__":
    sys.exit(main())
