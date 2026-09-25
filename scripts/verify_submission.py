"""Run a submission zip exactly like the scorer callback and report both metrics.

    python scripts/verify_submission.py SUBMISSION.zip DATA.parquet OUT.npz
Unzips into a temp dir, imports solution.PredictionModel, feeds every row via
utils.DataPoint in file order, and records predictions, wall time per row, the
official per-block result (utils.BlockAccumulator) and the global pooled WP.
Designed to run inside python:3.11-slim-bookworm with --cpus 1.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, BlockAccumulator, DataPoint  # noqa: E402


def global_wp(y, p, m):
    out = []
    for k in range(2):
        yk = np.clip(y[..., k][m], -2, 2).astype(np.float64)
        pk = np.clip(p[..., k][m], -2, 2).astype(np.float64)
        w = np.abs(yk)
        W = w.sum()
        my, mp = (w * yk).sum() / W, (w * pk).sum() / W
        cov = (w * (yk - my) * (pk - mp)).sum() / W
        sy = np.sqrt((w * (yk - my) ** 2).sum() / W)
        sp = np.sqrt((w * (pk - mp) ** 2).sum() / W)
        out.append(0.0 if sy <= 1e-8 or sp <= 1e-8 else float(cov / (sy * sp)))
    return out


def main():
    zpath, data, out = sys.argv[1:4]
    tmp = Path(tempfile.mkdtemp())
    zipfile.ZipFile(zpath).extractall(tmp)
    sys.path.insert(0, str(tmp))
    t0 = time.perf_counter()
    model = importlib.import_module("solution").PredictionModel()
    load_s = time.perf_counter() - t0
    pf = pq.ParquetFile(data)
    acc = BlockAccumulator()
    ys, ps, ms = [], [], []
    rows = 0
    busy = 0.0
    for g in range(pf.num_row_groups):
        t = pf.read_row_group(g)
        x = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        y = np.column_stack([t[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        need = t["need_prediction"].to_numpy().astype(bool)
        seq = int(t["seq_ix"][0].as_py())
        p = np.zeros((len(x), 2), np.float32)
        s0 = time.perf_counter()
        for i in range(len(x)):
            v = model.predict(DataPoint(seq, i, bool(need[i]), x[i]))
            if need[i]:
                v = np.asarray(v, dtype=np.float32)
                assert v.shape == (2,) and np.isfinite(v).all(), "bad prediction"
                p[i] = v
            else:
                assert v is None, "warm-up must return None"
        busy += time.perf_counter() - s0
        rows += len(x)
        m = t["is_scored"].to_numpy().astype(bool) & need
        acc.add(y, p, m)
        ys.append(y), ps.append(p), ms.append(m)
    y, p, m = np.stack(ys), np.stack(ps), np.stack(ms)
    g = global_wp(y, p, m)
    res = acc.result()
    print({"load_s": round(load_s, 2), "us_per_row": round(1e6 * busy / rows, 1), "rows": rows,
           "block_wp": round(res["weighted_pearson"], 6), "global_wp": round(float(np.mean(g)), 6),
           "global_t0": round(g[0], 6), "global_t1": round(g[1], 6),
           "eta_min_37M_rows": round(37.46e6 * busy / rows / 60, 1)})
    np.savez(out, pred=p, y=y, mask=m)


if __name__ == "__main__":
    main()
