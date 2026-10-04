"""Surprise miner falsifier: does the raw current-row book (112 features) carry class-3 tier/direction
information that the since-FT pair misses?

    WNN_CPU_PCT=19 WNN_MEM_MAX=8G scripts/limited .venv/bin/python scripts/cr_surprise_rawprobe.py ROWS.npz OUT.json

Class-3 scored rows, fold-crossfit MLPs: (a) winner-tier classifier on summary features + raw 112 at row t
(compare with cr_surprise_tier.py AUC 0.679 / p-only 0.666); (b) |y|-weighted residual regressor on the same
inputs, applied with a shrink factor chosen on the fitting folds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_surprise_hazard import CAUSAL, features, fit_mlp  # noqa: E402
from cr_surprise_lib import wp  # noqa: E402
from cr_surprise_tier import auc, fit_clf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(3)


def main() -> None:
    z = np.load(sys.argv[1])
    d = {k: z[k] for k in dict.fromkeys(CAUSAL + ["cls", "y0", "y1", "ft0", "ft1", "seq", "seq_fold"])}
    s3 = d["cls"] == 3
    d = {k: (v[s3] if k != "seq_fold" else v) for k, v in d.items()}
    fold = d["seq_fold"][d["seq"]]
    pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    names = pf.schema_arrow.names[4:116]
    raw = np.zeros((len(d["seq"]), 112), np.float32)
    for g in np.unique(d["seq"]):
        r = np.flatnonzero(d["seq"] == g)
        t = pf.read_row_group(int(g), columns=names, use_threads=False)
        x = np.column_stack([t.column(c).to_numpy() for c in names]).astype(np.float32)
        raw[r] = x[d["t"][r].astype(np.int64)]
    raw = (raw - raw.mean(0)) / (raw.std(0) + 1e-6)
    x = np.concatenate([features(d), raw], 1)
    lab = np.maximum(d["y0"], d["y1"]) > 0.3
    yy = np.column_stack([np.clip(d["y0"], -2, 2), np.clip(d["y1"], -2, 2)]).astype(np.float64)
    ww = np.abs(yy).mean(1)
    p = np.column_stack([d["ft0"], d["ft1"]]).astype(np.float64)
    sc = np.zeros(len(lab))
    res = np.zeros_like(p)
    for k in range(4):
        fit, app = fold != k, fold == k
        sc[app] = fit_clf(x[fit], lab[fit], x[app])
        res[app] = fit_mlp(x[fit], yy[fit] - p[fit], ww[fit], x[app], out=2)
        print("fold", k, flush=True)
    out = {"auc_tier_summary_plus_raw": auc(sc, lab), "base_c3": wp(d, "ft")}
    for lam in (0.25, 0.5, 1.0):
        q = p + lam * res
        out[f"residual_lam{lam}"] = wp(d, "ft", None, q[:, 0], q[:, 1]) - out["base_c3"]
        out[f"residual_lam{lam}_by_fold"] = [wp(d, "ft", fold == k, q[:, 0], q[:, 1]) - wp(d, "ft", fold == k)
                                             for k in range(4)]
    print(json.dumps(out, indent=1))
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
