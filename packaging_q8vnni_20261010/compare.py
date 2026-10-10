"""Host-side comparison of delivered-ZIP callback predictions (i16 control vs q8vnni) with the official metric."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils import TARGET_COLUMNS  # noqa: E402
from wnn.gmetric import global_wp  # noqa: E402

gates = Path(__file__).resolve().parent / "gates"
a = np.load(gates / "i16_v3.npz")
b = np.load(gates / "q8vnni_v3.npz")
groups = [int(g) for g in a["groups"]]
pf = pq.ParquetFile(MAIN / "datasets/valid.parquet")
T, M, PA, PB = [], [], [], []
for g in groups:
    t = pf.read_row_group(g, use_threads=False)
    T.append(np.column_stack([t[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32))
    M.append(t["is_scored"].to_numpy().astype(bool) & t["need_prediction"].to_numpy().astype(bool))
    PA.append(a[f"pred_{g}"])
    PB.append(b[f"pred_{g}"])
T, M, PA, PB = map(np.concatenate, (T, M, PA, PB))
ra, rb = global_wp(T, PA, M), global_wp(T, PB, M)
d = np.abs(PA[M] - PB[M])
rng = np.random.default_rng(0)
ends = np.cumsum([0] + [20000] * len(groups))
deltas = []
for _ in range(300):
    pick = rng.integers(0, len(groups), len(groups))
    idx = np.concatenate([np.arange(ends[i], ends[i + 1]) for i in pick])
    deltas.append(global_wp(T[idx], PB[idx], M[idx])["wp"] - global_wp(T[idx], PA[idx], M[idx])["wp"])
out = {
    "sequences": len(groups),
    "groups": groups,
    "scored_rows": int(M.sum()),
    "i16": {k: ra[k] for k in ("wp", "t0", "t1")},
    "q8vnni": {k: rb[k] for k in ("wp", "t0", "t1")},
    "wp_q8vnni_minus_i16": rb["wp"] - ra["wp"],
    "paired_bootstrap_95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
    "max_abs_diff_scored_rows": float(d.max()),
    "rms_diff_scored_rows": float(np.sqrt((d**2).mean())),
    "max_abs_diff_all_rows": float(np.abs(PA - PB).max()),
}
(gates / "metric_agreement.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
