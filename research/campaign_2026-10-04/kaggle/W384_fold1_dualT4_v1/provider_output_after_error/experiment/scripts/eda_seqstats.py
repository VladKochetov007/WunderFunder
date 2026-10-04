"""Per-sequence summary table for exploratory analysis.

    scripts/limited .venv/bin/python scripts/eda_seqstats.py valid
    scripts/limited .venv/bin/python scripts/eda_seqstats.py train 2000

One row per sequence: market class, target moments (all rows and scored rows,
|y|-weighted like the metric), sum/diff moments, event and trade rates, and per
feature column mean, std, change rate, number of distinct values and the two
most frequent values with their shares (two-state geometry). Written to
cache/seqstats_<split>.parquet.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, FEATURE_COLUMNS, SequenceFile, market_class

ROOT = Path(__file__).resolve().parents[1]
QUANTA = np.array([0.12543, 0.24133, 0.46854, 0.65687])


def _top2(col):
    v, n = np.unique(col, return_counts=True)
    o = np.argsort(-n)[:2]
    a = v[o[0]]
    b = v[o[1]] if len(o) > 1 else a
    fa = n[o[0]] / len(col)
    fb = (n[o[1]] / len(col)) if len(o) > 1 else 0.0
    return len(v), a, fa, b, fb


def summarize(args):
    split, g = args
    s = SequenceFile(ROOT / "datasets" / f"{split}.parquet").read(g)
    x, y = s.features, s.targets
    m = s.scored if s.scored is not None else np.zeros(len(x), bool)
    m = m & (np.arange(len(x)) >= 99)
    cls = market_class(y)
    q = QUANTA[cls] if cls >= 0 else np.nan
    row = {"row_group": g, "seq_ix": s.seq_ix, "cls": cls, "scored_frac": m.mean()}
    yc = np.clip(y[99:], -2, 2).astype(np.float64)
    for k in range(2):
        yk = yc[:, k]
        row[f"t{k}_mean"] = yk.mean()
        row[f"t{k}_std"] = yk.std()
        row[f"t{k}_zero"] = (y[99:, k] == 0).mean()
        if m.any():
            ys = np.clip(y[m, k], -2, 2).astype(np.float64)
            w = np.abs(ys)
            row[f"t{k}_wmean_sc"] = (w * ys).sum() / max(w.sum(), 1e-12)
            row[f"t{k}_W_sc"] = w.sum()
    S = (y[99:, 0] + y[99:, 1]) / q
    D = (y[99:, 0] - y[99:, 1]) / q
    row.update(S_mean_q=S.mean(), S_std_q=S.std(), D_std_q=D.std(), t01_corr=np.corrcoef(y[99:, 0], y[99:, 1])[0, 1])
    chg = np.diff(x, axis=0) != 0
    row["ev_i0_bid"] = chg[:, 0].mean()
    row["ev_i0_ask"] = chg[:, 11].mean()
    row["ev_i0_both"] = (chg[:, 0] & chg[:, 11]).mean()
    row["ev_i0_trade"] = chg[:, 48:52].any(1).mean()
    row["ev_i1_px"] = (chg[:, 52] | chg[:, 63]).mean()
    row["ev_target"] = (np.diff(y[:, 0]) != 0).mean()
    row["n_states_i0"] = len(np.unique(x[:, 0:22], axis=0))
    for j, name in enumerate(FEATURE_COLUMNS):
        col = x[:, j]
        n, a, fa, b, fb = _top2(col)
        row[f"{name}_mean"] = col.mean()
        row[f"{name}_std"] = col.std()
        row[f"{name}_chg"] = chg[:, j].mean()
        row[f"{name}_nuniq"] = n
        row[f"{name}_a"], row[f"{name}_fa"], row[f"{name}_b"], row[f"{name}_fb"] = a, fa, b, fb
    return row


def main():
    split = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    n = SequenceFile(ROOT / "datasets" / f"{split}.parquet").__len__()
    groups = [g for g in range(n) if split != "train" or g not in CORRUPT_TRAIN_ROW_GROUPS]
    if limit:
        groups = list(np.random.default_rng(0).choice(groups, limit, replace=False))
    workers = int(os.environ.get("WNN_THREADS", "8"))
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(summarize, [(split, int(g)) for g in groups], chunksize=8))
    df = pl.DataFrame(rows)
    df.write_parquet(ROOT / "cache" / f"seqstats_{split}.parquet")
    print(df.shape, df.group_by("cls").len().sort("cls"))


if __name__ == "__main__":
    main()
