"""Ceiling of online per-sequence recalibration from labels known with a lag.

    scripts/limited .venv/bin/python scripts/eda_online_ceiling.py ab18_wmse_struct 1.0
At row s the true targets of rows <= s - L are assumed known (all rows, scored or not).
Per sequence and target, a running |y|-weighted least squares fit y ~ a + b p with a
Gaussian prior (a=0, b=1; strength lam in weight units) recalibrates the prediction.
Reports global WP on scored rows (devB + holdout) per (L, lam, variant) to
cache/online_ceiling_<name>.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from wnn.gmetric import wp_from_sums, wp_sums
from wnn.splits import dev_halves, valid_row_groups

ROOT = Path(__file__).resolve().parents[1]
LAGS = (1, 10, 50, 200, 1000, 5000)
LAMS = (10.0, 100.0, 1000.0)


def lagged(c, lag):
    out = np.zeros_like(c)
    out[:, lag:] = c[:, :-lag]
    return out


def main():
    name, scale = sys.argv[1], float(sys.argv[2])
    d = np.load(ROOT / "cache" / "valid_targets.npz")
    rg = d["row_group"]
    path = ROOT / "datasets" / "valid.parquet"
    _, hold = valid_row_groups(path)
    _, dev_b = dev_halves(path)
    keep = np.flatnonzero(np.isin(rg, np.r_[np.asarray(dev_b), np.asarray(hold)]))
    pred = np.load(ROOT / "cache" / f"pred_{name}_valid.npy", mmap_mode="r")
    sums = {}
    for c0 in range(0, len(keep), 150):
        idx = keep[c0:c0 + 150]
        y = np.clip(d["y"][idx].astype(np.float64), -2, 2)
        m = d["mask"][idx]
        p = np.asarray(pred[idx], dtype=np.float64) * scale
        for k in range(2):
            yk, pk = y[..., k], p[..., k]
            w = np.abs(yk)
            cs = {n: np.cumsum(v, 1) for n, v in (("w", w), ("wp", w * pk), ("wy", w * yk), ("wpp", w * pk * pk), ("wpy", w * pk * yk))}
            key = ("none", 0, 0.0)
            sums[key + (k,)] = sums.get(key + (k,), 0) + wp_sums(yk, pk, m).reshape(-1, 6).sum(0)
            for lag in LAGS:
                c = {n: lagged(v, lag) for n, v in cs.items()}
                for lam in LAMS:
                    a_lvl = (c["wy"] - c["wp"]) / (c["w"] + lam)
                    q = pk + a_lvl
                    kk = ("level", lag, lam, k)
                    sums[kk] = sums.get(kk, 0) + wp_sums(yk, q, m).reshape(-1, 6).sum(0)
                    # affine: solve [[w+lam, wp], [wp, wpp+lam]] [a, b] = [wy, wpy+lam]
                    a11, a12, a22 = c["w"] + lam, c["wp"], c["wpp"] + lam
                    r1, r2 = c["wy"], c["wpy"] + lam
                    det = a11 * a22 - a12 * a12
                    a = (r1 * a22 - a12 * r2) / det
                    b = (a11 * r2 - a12 * r1) / det
                    kk = ("affine", lag, lam, k)
                    sums[kk] = sums.get(kk, 0) + wp_sums(yk, a + b * pk, m).reshape(-1, 6).sum(0)
    rows = {}
    for (var, lag, lam, k), s in sums.items():
        rows.setdefault((var, lag, lam), [0.0, 0.0])[k] = float(wp_from_sums(s))
    df = pl.DataFrame([{"variant": v, "lag": l, "lam": lm, "t0": r[0], "t1": r[1], "wp": 0.5 * (r[0] + r[1])}
                       for (v, l, lm), r in rows.items()]).sort("wp", descending=True)
    df.write_parquet(ROOT / "cache" / f"online_ceiling_{name}.parquet")
    print(df.head(12))
    print(df.filter(pl.col("variant") == "none"))


if __name__ == "__main__":
    main()
