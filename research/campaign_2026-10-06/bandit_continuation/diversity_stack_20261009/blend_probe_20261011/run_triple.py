import sys, json, itertools
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from run_probe import FLAT, P, halves, bootstrap, pearson_k, wp, wridge
fl = FLAT[1]; mem = ["h230_s3_ep3", "vt_gru512", "vt_chunk500"]
p = [P(1, m) for m in mem]; y = fl["y"]; seq = fl["seq"]
h = halves(fl["comp"])
simplex = [np.array(w) / 10 for w in itertools.product(range(11), repeat=3) if sum(w) == 10]
def fit_w(rows, per_target):
    ws = []
    for k in range(2):
        ws.append(max(simplex, key=lambda w: pearson_k(y[rows], sum(wi * pi[rows] for wi, pi in zip(w, p)), k)))
    if not per_target:
        w = max(simplex, key=lambda w: wp(y[rows], sum(wi * pi[rows] for wi, pi in zip(w, p)))); ws = [w, w]
    return ws
def apply_w(ws, rows):
    out = np.zeros((rows.sum(), 2), np.float32)
    for k in range(2): out[:, k] = sum(w * pi[rows][:, k] for w, pi in zip(ws[k], p))
    return out
def fit_r(rows):
    par = []
    for k in range(2):
        X = np.stack([pi[rows][:, k] for pi in p] + [np.abs(p[0][rows][:, k] - p[1][rows][:, k]), np.abs(p[0][rows][:, k] - p[2][rows][:, k])], 1).astype(np.float64)
        yy = y[rows][:, k].astype(np.float64); par.append(wridge(X, yy, np.abs(yy), 1e-3))
    return par
def apply_r(par, rows):
    out = np.zeros((rows.sum(), 2), np.float32)
    for k in range(2):
        X = np.stack([pi[rows][:, k] for pi in p] + [np.abs(p[0][rows][:, k] - p[1][rows][:, k]), np.abs(p[0][rows][:, k] - p[2][rows][:, k])], 1).astype(np.float64)
        out[:, k] = X @ par[k][0] + par[k][1]
    return out
n = len(y); pools = {k: np.zeros((n, 2), np.float32) for k in ("T1 simplex shared", "T2 simplex per-target", "T3 ridge[p1,p2,p3,|d12|,|d13|]")}
ref = sum(p) / 3; ref2 = 0.5 * (p[0] + p[1]); params = {}
for fs in (h, ~h):
    fr, er = fs[seq], ~fs[seq]
    ws1, ws2, r = fit_w(fr, False), fit_w(fr, True), fit_r(fr)
    pools["T1 simplex shared"][er] = apply_w(ws1, er); pools["T2 simplex per-target"][er] = apply_w(ws2, er); pools["T3 ridge[p1,p2,p3,|d12|,|d13|]"][er] = apply_r(r, er)
    params.setdefault("T1", []).append([w.tolist() for w in ws1[:1]]); params.setdefault("T2", []).append([w.tolist() for w in ws2])
    print("fit wp", wp(y[fr], apply_w(ws1, fr)), flush=True)
rows = np.ones(n, bool); res = {"members": mem, "params": params, "singles": [wp(y, pi) for pi in p], "static3": wp(y, ref), "static_pair01": wp(y, ref2)}
for k, v in pools.items():
    a = bootstrap(fl, rows, v, ref); b = bootstrap(fl, rows, v, p[0])
    res[k] = {"heldout_wp": a["candidate_wp"], "delta_vs_static3": a["mean_diff"], "ci": a["ci95"], "delta_vs_A": b["mean_diff"], "ci_A": b["ci95"]}
c = bootstrap(fl, rows, ref, p[0]); res["static3_vs_A"] = {"delta": c["mean_diff"], "ci": c["ci95"]}
Path(__file__).with_name("res_triple_crossfit.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
