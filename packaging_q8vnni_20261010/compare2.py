"""Metric agreement q8vnni vs i16 for the upload candidates (diagnostic: these weights saw fold 1)."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(HERE.parents[0] / "src"))
from utils import TARGET_COLUMNS  # noqa: E402
from wnn.gmetric import global_wp  # noqa: E402

gates = HERE / "gates_upload"
pf = pq.ParquetFile(MAIN / "datasets/valid.parquet")
res = {}
for label, ref, new in (("z5_single", "z5_i16_v3", "z5_q8vnni_v3"), ("z5_z4_pair", "z5z4pair_i16_v3", "pair_q8vnni_v3")):
    a, b = np.load(gates / f"{ref}.npz"), np.load(gates / f"{new}.npz")
    groups = [int(g) for g in a["groups"]]
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
    res[label] = {
        "reference": ref, "candidate": new, "scored_rows": int(M.sum()),
        "ref_wp": ra["wp"], "cand_wp": rb["wp"], "wp_delta": rb["wp"] - ra["wp"],
        "bootstrap_95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        "max_abs_diff_scored": float(d.max()), "rms_diff_scored": float(np.sqrt((d**2).mean())),
    }
(gates / "metric_agreement.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))
