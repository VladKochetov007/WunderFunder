"""Bounded parity/reset check for TL3+W18 candidate ZIPs (first N valid sequences)."""
import importlib.util, json, sys, time, zipfile, shutil
from pathlib import Path
import numpy as np, pyarrow.parquet as pq

N = int(sys.argv[1]); OUT = Path("/out"); WORK = Path("/tmp/pkgs")
ZIPS = {"tl3": "/zips/team_tl3_original_20261018.zip", "w18own": "/zips/tabdata-w18-final.zip",
        **{k: f"/zips/{k}.zip" for k in ["probe_w18_only", "probe_s5_only", "tl3_w18swap", "tl3_s5w18"]}}

class DP:
    __slots__ = ("seq_ix", "step_in_seq", "need_prediction", "state")
    def __init__(s, a, b, c, d): s.seq_ix, s.step_in_seq, s.need_prediction, s.state = a, b, c, d

def load(name):
    d = WORK / name
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    with zipfile.ZipFile(ZIPS[name]) as z: z.extractall(d)
    for m in [k for k in sys.modules if k.startswith(("wnn_cpp", "wnn_native", "solution"))]: del sys.modules[m]
    sys.path[:] = [p for p in sys.path if not p.startswith(str(WORK))]
    spec = importlib.util.spec_from_file_location(f"solution_{name}", d / "solution.py")
    mod = importlib.util.module_from_spec(spec); sys.path.insert(0, str(d)); spec.loader.exec_module(mod)
    return mod.PredictionModel()

pf = pq.ParquetFile("/inputs/valid.parquet")
seqs = []
for g in range(N):
    t = pf.read_row_group(g)
    meta = {"seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1"}
    cols = [c for c in t.column_names if c not in meta]
    X = np.stack([t.column(c).to_numpy().astype(np.float32) for c in cols], axis=1)
    seqs.append((t.column("seq_ix").to_numpy(), t.column("step_in_seq").to_numpy(), t.column("need_prediction").to_numpy(zero_copy_only=False), X))

def run(model, order):
    outs = []
    for i in order:
        sx, st, need, X = seqs[i]; rows = []
        for r in range(len(X)):
            p = model.predict(DP(int(sx[r]), int(st[r]), bool(need[r]), X[r]))
            if need[r]: rows.append(np.asarray(p, np.float64))
        outs.append(np.stack(rows))
    return outs

res = {"n_sequences": N, "feature_width": int(seqs[0][3].shape[1]), "runs": {}}
preds = {}
for name in ZIPS:
    m = load(name); t0 = time.time()
    o = run(m, list(range(N)) + [0]) if name != "w18own" else run(m, list(range(N)))
    dt = time.time() - t0
    preds[name] = np.concatenate(o[:N])
    r = {"seconds": dt, "finite": bool(np.isfinite(preds[name]).all()), "rows": int(len(preds[name]))}
    if name != "w18own": r["aba_max_abs"] = float(np.abs(o[N] - o[0]).max())
    res["runs"][name] = r; print(name, r, flush=True)
    del m

def mad(a, b): return float(np.abs(a - b).max())
def corr(a, b): return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
P = preds
res["checks"] = {
    "w18_wrapper_vs_own_max_abs": mad(P["probe_w18_only"], P["w18own"]),
    "w18_wrapper_vs_own_corr": corr(P["probe_w18_only"], P["w18own"]),
    "w18_vs_s5_corr": corr(P["probe_w18_only"], P["probe_s5_only"]),
    "swap_vs_recon_max_abs": mad(P["tl3_w18swap"], P["tl3"] - .34 * P["probe_s5_only"] + .34 * P["probe_w18_only"]),
    "s5w18_vs_recon_max_abs": mad(P["tl3_s5w18"], P["tl3"] - .17 * P["probe_s5_only"] + .17 * P["probe_w18_only"]),
    "swap_vs_tl3_corr": corr(P["tl3_w18swap"], P["tl3"]),
    "pred_abs_scale_tl3": float(np.abs(P["tl3"]).mean()),
}
print(json.dumps(res["checks"], indent=1))
(OUT / f"PARITY_N{N}_R1.json").write_text(json.dumps(res, indent=1))
