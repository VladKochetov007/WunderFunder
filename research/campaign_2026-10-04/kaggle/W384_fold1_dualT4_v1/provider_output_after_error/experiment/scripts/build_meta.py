"""Per-sequence metadata table (market class, activity rates) for stratified analysis.

Usage: python scripts/build_meta.py [--workers N]
Writes cache/meta_{train,valid}.npz.
"""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from wnn.data import I0, I1, market_class

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "row_group",
    "seq_ix",
    "market_class",
    "t0_std",
    "t1_std",
    "t0_zero",
    "t01_corr",
    "px_event_rate",
    "trade_rate",
    "i1_px_event_rate",
    "scored_frac",
)


def _one(args):
    path, index = args
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    cols = ["seq_ix", "i0_p0", "i0_p11", "i0_dv0", "i1_p0", "i1_p11", "t0", "t1"]
    if "is_scored" in names:
        cols.append("is_scored")
    t = pf.read_row_group(index, columns=cols, use_threads=False)
    y = np.column_stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()])
    bid, ask = t.column("i0_p0").to_numpy(), t.column("i0_p11").to_numpy()
    px = (np.diff(bid) != 0) & (np.diff(ask) != 0)
    i1 = (np.diff(t.column("i1_p0").to_numpy()) != 0) | (
        np.diff(t.column("i1_p11").to_numpy()) != 0
    )
    trade = np.diff(t.column("i0_dv0").to_numpy()) != 0
    scored = t.column("is_scored").to_numpy().mean() if "is_scored" in names else np.nan
    corr = np.corrcoef(y.T)[0, 1] if y.std(axis=0).min() > 0 else np.nan
    return (
        index,
        int(t.column("seq_ix")[0].as_py()),
        market_class(y),
        y[:, 0].std(),
        y[:, 1].std(),
        (y[:, 0] == 0).mean(),
        corr,
        px.mean(),
        trade.mean(),
        i1.mean(),
        scored,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    assert I0.bid_px.start == 0 and I1.bid_px.start == 52
    out_dir = ROOT / "cache"
    out_dir.mkdir(exist_ok=True)
    for split in ("valid", "train"):
        path = str(ROOT / "datasets" / f"{split}.parquet")
        n = pq.ParquetFile(path).num_row_groups
        with Pool(args.workers) as pool:
            rows = pool.map(_one, [(path, i) for i in range(n)], chunksize=16)
        arr = np.array(rows, dtype=np.float64)
        np.savez(out_dir / f"meta_{split}.npz", **{k: arr[:, j] for j, k in enumerate(FIELDS)})
        cls = arr[:, 2].astype(int)
        print(split, n, "class counts", {int(c): int((cls == c).sum()) for c in np.unique(cls)})
        for c in np.unique(cls):
            sel = cls == c
            print(
                f"  class {c}: t0_std {arr[sel, 3].mean():.3f} zero {arr[sel, 5].mean():.3f} "
                f"corr {np.nanmean(arr[sel, 6]):.3f} px_rate {arr[sel, 7].mean():.3f} "
                f"trade {arr[sel, 8].mean():.3f} i1_px {arr[sel, 9].mean():.3f} "
                f"scored {np.nanmean(arr[sel, 10]):.3f}"
            )


if __name__ == "__main__":
    main()
