"""Package check: candidate ZIP vs linear reconstruction from other ZIPs on first N valid sequences.

Usage: pkgcheck.py N OUT CAND_ZIP REF_ZIP:COEF ...  Reports finite, A-B-A reset, recon max abs, runtime ratios.
"""
import importlib.util, json, shutil, sys, time, zipfile
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

N, OUT, CAND = int(sys.argv[1]), sys.argv[2], sys.argv[3]
TERMS = [(t.rsplit(":", 1)[0], float(t.rsplit(":", 1)[1])) for t in sys.argv[4:]]
WORK = Path("/tmp/pk")
pf = pq.ParquetFile("/inputs/valid.parquet"); meta = {"seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"}
seqs = []
for g in range(N):
    t = pf.read_row_group(g); cols = [c for c in t.column_names if c not in meta]
    seqs.append((t.column("seq_ix").to_numpy(), t.column("step_in_seq").to_numpy(),
                 t.column("need_prediction").to_numpy(zero_copy_only=False),
                 np.stack([t.column(c).to_numpy().astype(np.float32) for c in cols], 1)))
class DP:
    def __init__(s, a, b, c, x): s.seq_ix, s.step_in_seq, s.need_prediction, s.state = a, b, c, x
def load(z, tag):
    d = WORK / tag; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True); zipfile.ZipFile(z).extractall(d)
    for m in [k for k in sys.modules if k.startswith(("wnn_cpp", "solution"))]: del sys.modules[m]
    sys.path[:] = [p for p in sys.path if not p.startswith(str(WORK))]; sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(f"sol_{tag}", d / "solution.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod.PredictionModel()
def run(m, order):
    outs = []
    for i in order:
        sx, st, need, X = seqs[i]; rows = []
        for r in range(len(X)):
            p = m.predict(DP(int(sx[r]), int(st[r]), bool(need[r]), X[r]))
            if need[r]: rows.append(np.asarray(p, np.float64))
        outs.append(np.stack(rows))
    return outs
res = {"n_seq": N, "cand": CAND, "terms": TERMS, "runs": {}}
preds = {}
for k, z in enumerate([CAND] + [t[0] for t in TERMS]):
    m = load(z, f"m{k}"); t0 = time.time(); o = run(m, list(range(N)) + [0]); dt = time.time() - t0
    preds[z] = np.concatenate(o[:N])
    res["runs"][z] = {"seconds": dt, "finite": bool(np.isfinite(preds[z]).all()), "aba_max_abs": float(np.abs(o[N] - o[0]).max())}
    print(z, res["runs"][z], flush=True); del m
recon = sum(c * preds[z] for z, c in TERMS)
res["recon_max_abs"] = float(np.abs(preds[CAND] - recon).max())
print(json.dumps({"recon_max_abs": res["recon_max_abs"]})); Path(OUT).write_text(json.dumps(res, indent=1))
