"""Scored-row table for bin analyses in the EDA notebook.

    scripts/limited .venv/bin/python scripts/eda_scored_table.py
Writes cache/valid_scored.parquet: one row per scored validation row with sequence index,
step, class, split (A = devA, B = devB, H = holdout), targets, predictions of cached
models and causal event-recency context from cache/valid_rowinfo.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from wnn.splits import dev_halves, valid_row_groups

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"base": ("baseline", 0.25), "ab16": ("ab16_wmse", 1.0), "ab18": ("ab18_wmse_struct", 1.0), "ab9": ("ab9_pow2_h128", 1.0)}


def main():
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    y, m, cls, rg = d["y"], d["mask"], d["market_class"], d["row_group"]
    path = ROOT / "datasets" / "valid.parquet"
    _, hold = valid_row_groups(path)
    dev_a, _ = dev_halves(path)
    split = np.where(np.isin(rg, np.asarray(hold)), "H", np.where(np.isin(rg, np.asarray(dev_a)), "A", "B"))
    seq, step = np.nonzero(m)
    cols = {"seq": seq.astype(np.int32), "step": step.astype(np.int32), "cls": cls[seq].astype(np.int8),
            "split": split[seq], "y0": y[seq, step, 0], "y1": y[seq, step, 1]}
    for k, (name, scale) in MODELS.items():
        p = np.load(ROOT / "cache" / f"pred_{name}_valid.npy", mmap_mode="r")
        v = np.asarray(p[seq, step]) * scale
        cols[f"{k}0"], cols[f"{k}1"] = v[:, 0].astype(np.float32), v[:, 1].astype(np.float32)
    info = np.load(ROOT / "cache" / "valid_rowinfo.npz")
    for key in info.files:
        cols[key] = info[key][seq, step]
    pl.DataFrame(cols).write_parquet(ROOT / "cache" / "valid_scored.parquet")
    print(len(seq))


if __name__ == "__main__":
    main()
