"""Do whole feature rows (all 112 columns) of validation sequences occur verbatim in other sequences? (Santander 'fake/duplicate row' check)"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS  # noqa: E402


def rows(pf, g, n=None):
    t = pf.read_row_group(int(g), columns=list(FEATURE_COLUMNS), use_threads=False)
    x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float64)
    bits = np.ascontiguousarray(x[:n]).view(np.uint64)
    mult = (np.arange(bits.shape[1], dtype=np.uint64) * np.uint64(2) + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
    return (bits * mult).sum(1, dtype=np.uint64)  # 64-bit row hash


def main() -> int:
    rng = np.random.default_rng(2)
    pt, pv = pq.ParquetFile(ROOT / "datasets/train.parquet"), pq.ParquetFile(ROOT / "datasets/valid.parquet")
    tg = rng.choice(pt.num_row_groups, 150, replace=False)
    pool = np.unique(np.concatenate([rows(pt, g, 20000) for g in tg]))
    print("train pool rows (unique):", len(pool))
    for g in rng.choice(pv.num_row_groups, 12, replace=False):
        r = rows(pv, g)
        within = 1 - len(np.unique(r)) / len(r)
        cross = float(np.isin(r, pool).mean())
        print(f"valid group {g}: rows {len(r)}, within-seq duplicate share {within:.3f}, found in 150 train seqs {cross:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
