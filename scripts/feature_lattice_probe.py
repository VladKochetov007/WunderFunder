"""Measure whether any of the 32 anonymous feature columns is a small discrete alphabet, and how label changes relate to them.

Per column over a sample of validation sequences: distinct values, share of rows covered by the 16 most frequent values, share of
exact zeros, whether values sit on a lattice (value * 10^k integer for the smallest k <= 6), and the change-rate lift: P(label
changes | feature changes) / P(label changes).  Also the co-change rate of t0 and t1 versus independence.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, default=40)
    args = parser.parse_args()
    pf = pq.ParquetFile(ROOT / "datasets" / "valid.parquet")
    names = pf.schema_arrow.names
    targets = [c for c in names if c.startswith("t") and c[1:].isdigit()][:2] or ["t0", "t1"]
    xs, ys = [], []
    step = max(1, pf.num_row_groups // args.sequences)
    for g in range(0, pf.num_row_groups, step)[: args.sequences]:
        table = pf.read_row_group(g, columns=[*FEATURE_COLUMNS, *targets], use_threads=False)
        xs.append(np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64))
        ys.append(np.column_stack([table[c].to_numpy() for c in targets]).astype(np.float64))
    x = np.concatenate(xs)
    y_seq = ys
    print(f"sample: {len(xs)} sequences, {len(x)} rows, targets {targets}")
    print("col  distinct  top16share  zeros%  lattice_k  featchg%  P(labelchg|featchg)/P(labelchg)")
    label_change = np.concatenate([np.r_[False, (np.abs(np.diff(s, axis=0)) > 0).any(axis=1)] for s in y_seq])
    base = label_change.mean()
    for j, name in enumerate(FEATURE_COLUMNS):
        col = x[:, j]
        vals, counts = np.unique(col, return_counts=True)
        top = np.sort(counts)[::-1][:16].sum() / len(col)
        k = next((k for k in range(0, 7) if np.allclose(col * 10.0**k, np.round(col * 10.0**k), atol=1e-9)), -1)
        chg = np.concatenate([np.r_[False, np.diff(c) != 0] for c in [seq[:, j] for seq in xs]])
        lift = label_change[chg].mean() / base if chg.any() else float("nan")
        print(f"{j:>3}  {len(vals):>8}  {top:10.3f}  {100 * (col == 0).mean():6.1f}  {k:>9}  {100 * chg.mean():7.1f}  {lift:8.2f}")
    c0 = np.concatenate([np.r_[False, np.diff(s[:, 0]) != 0] for s in y_seq])
    c1 = np.concatenate([np.r_[False, np.diff(s[:, 1]) != 0] for s in y_seq])
    both = (c0 & c1).mean()
    print(f"\nlabel change rate: t0 {c0.mean():.4f}  t1 {c1.mean():.4f}  both {both:.4f}  independence {c0.mean() * c1.mean():.4f}  "
          f"co-change lift {both / max(c0.mean() * c1.mean(), 1e-12):.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
