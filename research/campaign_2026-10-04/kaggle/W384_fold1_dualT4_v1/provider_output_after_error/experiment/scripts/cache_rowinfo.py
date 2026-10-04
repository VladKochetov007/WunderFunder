"""Per-row causal context for validation, used by the EDA notebook's bin analyses.

    scripts/limited .venv/bin/python scripts/cache_rowinfo.py
Writes cache/valid_rowinfo.npz with (B, T) arrays: rows since the last i0 book-price
change, rows since the last i0 trade, count of i0 book-price changes so far, and
rows since the last i1 price change.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from wnn.data import SequenceFile

ROOT = Path(__file__).resolve().parents[1]


def _since(ev):
    idx = np.arange(len(ev))
    last = np.maximum.accumulate(np.where(ev, idx, 0))
    return (idx - last).astype(np.int32)


def _run(g):
    x = SequenceFile(ROOT / "datasets" / "valid.parquet").read(g, targets=False, mask=False).features
    ch = np.r_[np.zeros((1, 112), bool), x[1:] != x[:-1]]
    book = ch[:, 0:22].any(1)
    trade = ch[:, 48:52].any(1)
    i1 = ch[:, 52:74].any(1)
    return _since(book), _since(trade), np.cumsum(book).astype(np.int32), _since(i1)


def main():
    groups = np.load(ROOT / "cache" / "valid_targets.npz")["row_group"]
    out = [np.empty((len(groups), 20_000), np.int32) for _ in range(4)]
    with ProcessPoolExecutor(int(os.environ.get("WNN_THREADS", "4"))) as ex:
        for i, r in enumerate(ex.map(_run, [int(g) for g in groups], chunksize=8)):
            for k in range(4):
                out[k][i] = r[k]
    np.savez(ROOT / "cache" / "valid_rowinfo.npz", since_book=out[0], since_trade=out[1], n_book=out[2], since_i1=out[3])


if __name__ == "__main__":
    main()
