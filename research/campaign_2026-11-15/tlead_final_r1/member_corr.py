"""Native member-output correlation matrix on first N valid sequences (scored rows, mean over targets)."""
import importlib.util, json, shutil, sys, zipfile
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
N, OUT, ZIPS = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
pf = pq.ParquetFile("/inputs/valid.parquet"); meta = {"seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"}
class DP:
    def __init__(s, a, b, c, x): s.seq_ix, s.step_in_seq, s.need_prediction, s.state = a, b, c, x
data = []
for g in range(N):
    t = pf.read_row_group(g); cols = [c for c in t.column_names if c not in meta]
    data.append((t.column("seq_ix").to_numpy(), t.column("step_in_seq").to_numpy(), t.column("need_prediction").to_numpy(zero_copy_only=False),
                 t.column("is_scored").to_numpy(zero_copy_only=False), np.stack([t.column(c).to_numpy().astype(np.float32) for c in cols], 1),
                 np.stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()], 1)))
P = {}
for k, z in enumerate(ZIPS):
    d = Path(f"/tmp/mc{k}"); shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True); zipfile.ZipFile(z).extractall(d)
    for m in [x for x in sys.modules if x.startswith(("wnn_cpp", "sol"))]: del sys.modules[m]
    sys.path[:] = [p for p in sys.path if not p.startswith("/tmp/mc")]; sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(f"sol{k}", d / "solution.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    m = mod.PredictionModel(); rows = []
    for sx, st, need, sc, X, Y in data:
        for r in range(len(X)):
            p = m.predict(DP(int(sx[r]), int(st[r]), bool(need[r]), X[r]))
            if need[r] and sc[r]: rows.append(p)
    P[Path(z).stem] = np.asarray(rows, np.float64)
names = list(P); C = {f"{a}~{b}": float(np.mean([np.corrcoef(P[a][:, j], P[b][:, j])[0, 1] for j in range(2)])) for i, a in enumerate(names) for b in names[i + 1:]}
print(json.dumps(C, indent=1)); Path(OUT).write_text(json.dumps({"n_seq": N, "corr": C}, indent=1))
