"""Cache targets, scoring mask and market class of a labelled parquet file.

    scripts/limited .venv/bin/python scripts/cache_targets.py valid
Writes cache/<split>_targets.npz: y (B, T, 2) float32, mask (B, T) bool (all
False for train), seq_ix, market_class. Reads only the needed columns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, market_class

ROOT = Path(__file__).resolve().parents[1]


def main():
    split = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    pf = pq.ParquetFile(ROOT / "datasets" / f"{split}.parquet")
    has_mask = "is_scored" in pf.schema_arrow.names
    groups = [g for g in range(pf.num_row_groups) if split != "train" or g not in CORRUPT_TRAIN_ROW_GROUPS]
    if limit:
        groups = groups[:limit]
    cols = ["seq_ix", "t0", "t1"] + (["is_scored"] if has_mask else [])
    y = np.empty((len(groups), 20_000, 2), np.float32)
    m = np.zeros((len(groups), 20_000), bool)
    seq = np.empty(len(groups), np.int64)
    for i, g in enumerate(groups):
        t = pf.read_row_group(g, columns=cols, use_threads=False)
        y[i, :, 0] = t["t0"].to_numpy()
        y[i, :, 1] = t["t1"].to_numpy()
        if has_mask:
            m[i] = t["is_scored"].to_numpy()
        seq[i] = t["seq_ix"][0].as_py()
    cls = np.array([market_class(yy) for yy in y])
    np.savez(ROOT / "cache" / f"{split}_targets.npz", y=y, mask=m, seq_ix=seq, row_group=np.array(groups),
             market_class=cls)
    print(split, y.shape, np.bincount(cls + 1))


if __name__ == "__main__":
    main()
