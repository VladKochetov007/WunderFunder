"""Official-metric WP of native probes on the first N valid sequences (scored rows). Tests valid exposure of members."""
import importlib.util, json, shutil, sys, zipfile
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "/src")
from wnn.gmetric import global_wp
N, OUT, ZIPS = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
pf = pq.ParquetFile("/inputs/valid.parquet"); meta = {"seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"}
class DP:
    def __init__(s, a, b, c, x): s.seq_ix, s.step_in_seq, s.need_prediction, s.state = a, b, c, x
models = []
for k, z in enumerate(ZIPS):
    d = Path(f"/tmp/ss{k}"); shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True); zipfile.ZipFile(z).extractall(d)
    for m in [x for x in sys.modules if x.startswith(("wnn_cpp", "sol"))]: del sys.modules[m]
    sys.path[:] = [p for p in sys.path if not p.startswith("/tmp/ss")]; sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(f"sol{k}", d / "solution.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    models.append(mod.PredictionModel())
P = [[] for _ in ZIPS]; Y = []
for g in range(N):
    t = pf.read_row_group(g); cols = [c for c in t.column_names if c not in meta]
    X = np.stack([t.column(c).to_numpy().astype(np.float32) for c in cols], 1)
    sx, st = t.column("seq_ix").to_numpy(), t.column("step_in_seq").to_numpy()
    need, sc = t.column("need_prediction").to_numpy(zero_copy_only=False), t.column("is_scored").to_numpy(zero_copy_only=False)
    yy = np.stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()], 1)
    for r in range(len(X)):
        dp = DP(int(sx[r]), int(st[r]), bool(need[r]), X[r])
        outs = [m.predict(dp) for m in models]
        if need[r] and sc[r]:
            Y.append(yy[r]); [P[k].append(o) for k, o in enumerate(outs)]
Y = np.asarray(Y, np.float32)
res = {"n_seq": N, "rows": len(Y), "wp": {Path(z).stem: global_wp(Y, np.asarray(P[k], np.float32), np.ones(len(Y), bool)) for k, z in enumerate(ZIPS)}}
print(json.dumps(res, indent=1)); Path(OUT).write_text(json.dumps(res, indent=1))
