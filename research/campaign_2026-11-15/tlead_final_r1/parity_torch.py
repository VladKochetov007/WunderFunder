"""Native TL3-wrapper probe vs torch FP32 predictions on the first N valid sequences (need_prediction rows)."""
import importlib.util, json, sys, time, zipfile
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

zip_path, ref_path, N, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
d = Path("/tmp/probe"); d.mkdir(parents=True, exist_ok=True)
zipfile.ZipFile(zip_path).extractall(d); sys.path.insert(0, str(d))
spec = importlib.util.spec_from_file_location("solution", d / "solution.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); model = mod.PredictionModel()
ref = np.load(ref_path, mmap_mode="r")
class DP:
    def __init__(s, a, b, c, x): s.seq_ix, s.step_in_seq, s.need_prediction, s.state = a, b, c, x
pf = pq.ParquetFile("/inputs/valid.parquet"); meta = {"seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"}
P, Rf = [], []; t0 = time.time()
for g in range(N):
    t = pf.read_row_group(g); cols = [c for c in t.column_names if c not in meta]
    X = np.stack([t.column(c).to_numpy().astype(np.float32) for c in cols], 1)
    sx, st = t.column("seq_ix").to_numpy(), t.column("step_in_seq").to_numpy()
    need = t.column("need_prediction").to_numpy(zero_copy_only=False)
    for r in range(len(X)):
        p = model.predict(DP(int(sx[r]), int(st[r]), bool(need[r]), X[r]))
        if need[r]: P.append(p); Rf.append(ref[g, int(st[r])])
P, Rf = np.asarray(P, np.float64), np.asarray(Rf, np.float64)
res = {"zip": zip_path, "n_seq": N, "rows": len(P), "seconds": time.time() - t0, "finite": bool(np.isfinite(P).all()),
       "max_abs": float(np.abs(P - Rf).max()), "mean_abs": float(np.abs(P - Rf).mean()),
       "corr": [float(np.corrcoef(P[:, j], Rf[:, j])[0, 1]) for j in range(2)], "ref_abs_mean": float(np.abs(Rf).mean())}
print(json.dumps(res)); Path(out).write_text(json.dumps(res, indent=1))
