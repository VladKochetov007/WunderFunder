"""Zero-fit: weighted target variance (zero/constant-predictor WMSE) and per-target mass on scored valid rows."""
import json, sys
import numpy as np, pyarrow.parquet as pq
pf = pq.ParquetFile(sys.argv[1])
acc = {k: np.zeros(4) for k in ("t0", "t1")}  # sum w, sum w y, sum w y^2, n
for i in range(pf.num_row_groups):
    t = pf.read_row_group(i, columns=["t0", "t1", "need_prediction", "is_scored"])
    m = t["need_prediction"].to_numpy(zero_copy_only=False) & t["is_scored"].to_numpy(zero_copy_only=False)
    for k in acc:
        y = np.clip(t[k].to_numpy()[m].astype(np.float64), -2, 2); w = np.abs(y)
        acc[k] += [w.sum(), (w*y).sum(), (w*y*y).sum(), m.sum()]
out = {}
for k, (sw, swy, swyy, n) in acc.items():
    out[k] = {"weight_sum": sw, "scored_rows": n, "wmse_zero_pred": swyy/sw, "weighted_var": swyy/sw-(swy/sw)**2}
tot = sum(a[0] for a in acc.values())
out["t0_mass_share"] = acc["t0"][0]/tot
out["pooled_wmse_zero_pred"] = sum(a[2] for a in acc.values())/tot
print(json.dumps(out, indent=1))
