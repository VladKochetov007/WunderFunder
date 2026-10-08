#!/usr/bin/env python3
"""H-281 step 1: estimate the target horizon from feature columns on the fold0 468 panel.

For each lag L, regress y_t (t0 and t1) on features at row t+L. Reads only the 468
evaluation row groups of valid.parquet with column projection. Feature rows at t+L
are used for diagnosis only, never for prediction. Cross-validated R^2 uses the same
5-fold component split as the blend lab (seed 271828).
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
OUT = ROOT / "research/campaign_2026-10-06/bandit_continuation/delayed_label_H281"
SEED = 271828
N_FOLDS = 5
STRIDE = 5
T_MIN = 99
LAGS = list(range(0, 31)) + [40, 50, 60, 80, 100]
t0_clock = time.time()

spec = importlib.util.spec_from_file_location(
    "campaign_score", ROOT / "research/campaign_2026-09-30/audit/campaign_score.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)

split = json.loads((ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json").read_text())
EVAL = [int(g) for g in split["evaluation"]["valid"]]
COMP_OF = {int(i["row_group"]): int(i["component"]) for i in split["sequences"]["valid"]}
COMPS = np.asarray([COMP_OF[g] for g in EVAL])
UNIQ = np.unique(COMPS)
rng = np.random.default_rng(SEED)
perm = rng.permutation(len(UNIQ))
fold_of_comp = {int(UNIQ[p]): i % N_FOLDS for i, p in enumerate(perm)}
FOLD = {g: fold_of_comp[int(COMP_OF[g])] for g in EVAL}

targets = scorer.mmap_stored_npz(ROOT / "cache/valid_targets.npz")
MC = {g: int(targets["market_class"][g]) for g in EVAL}

pf = pq.ParquetFile(ROOT / "datasets/valid.parquet")
names = pf.schema_arrow.names
FEAT = [n for n in names if n.startswith("i0_") or n.startswith("i1_") or (n.startswith("a") and n[1:].isdigit())]
assert len(FEAT) == 112, len(FEAT)
print("features", len(FEAT), flush=True)

D = len(FEAT) + 1  # with intercept
CLASSES = [0, 1, 2, 3, "all"]
# Accumulators per (class, fold, lag, target): normal equations and sums.
acc = {}
uni = {}
def slot(store, key, shape):
    if key not in store:
        store[key] = np.zeros(shape)
    return store[key]

for g in EVAL:
    tbl = pf.read_row_group(g, columns=FEAT + ["t0", "t1"])
    F = np.column_stack([tbl.column(c).to_numpy().astype(np.float64) for c in FEAT])
    Y = np.column_stack([tbl.column("t0").to_numpy(), tbl.column("t1").to_numpy()]).astype(np.float64)
    n_rows = F.shape[0]
    f = FOLD[g]
    mc = MC[g]
    for L in LAGS:
        ts = np.arange(T_MIN, n_rows - L, STRIDE)
        X = np.hstack([np.ones((len(ts), 1)), F[ts + L]])
        for k in range(2):
            y = Y[ts, k]
            for cls in (mc, "all"):
                key = (cls, f, L, k)
                A = slot(acc, ("xtx",) + key, (D, D))
                A += X.T @ X
                b = slot(acc, ("xty",) + key, (D,))
                b += X.T @ y
                s = slot(acc, ("yty",) + key, (1,))
                s += y @ y
                c = slot(acc, ("ysum",) + key, (1,))
                c += y.sum()
                m = slot(acc, ("n",) + key, (1,))
                m += len(y)
                # univariate correlation sums per feature (weighted by nothing)
                fx = F[ts + L]
                su = slot(uni, ("sf",) + key, (len(FEAT),))
                su += fx.sum(0)
                sq = slot(uni, ("sff",) + key, (len(FEAT),))
                sq += (fx * fx).sum(0)
                sy_ = slot(uni, ("sfy",) + key, (len(FEAT),))
                sy_ += fx.T @ y
    if EVAL.index(g) % 100 == 0:
        print("seq", EVAL.index(g), f"t={time.time()-t0_clock:.0f}s", flush=True)

def oof_r2(cls, L, k):
    sse = 0.0
    sst = 0.0
    total = {}
    for name in ("xtx", "xty", "yty", "ysum", "n"):
        total[name] = sum(acc[(name, cls, f, L, k)] for f in range(N_FOLDS))
    ybar = total["ysum"][0] / total["n"][0]
    for f in range(N_FOLDS):
        tr = {nm: total[nm] - acc[(nm, cls, f, L, k)] for nm in total}
        ridge = 1e-8 * np.trace(tr["xtx"]) / D * np.eye(D)
        beta = np.linalg.solve(tr["xtx"] + ridge, tr["xty"])
        Xa, Xb = acc[("xtx", cls, f, L, k)], acc[("xty", cls, f, L, k)]
        yy = acc[("yty", cls, f, L, k)][0]
        nf = acc[("n", cls, f, L, k)][0]
        sse += yy - 2 * beta @ Xb + beta @ Xa @ beta
        sst += yy - 2 * ybar * acc[("ysum", cls, f, L, k)][0] + nf * ybar * ybar
    return float(1 - sse / sst)

def uni_best(cls, L, k):
    n = acc[("n", cls, 0, L, k)][0] + 0
    n = sum(acc[("n", cls, f, L, k)][0] for f in range(N_FOLDS))
    sf = sum(uni[("sf", cls, f, L, k)] for f in range(N_FOLDS))
    sff = sum(uni[("sff", cls, f, L, k)] for f in range(N_FOLDS))
    sfy = sum(uni[("sfy", cls, f, L, k)] for f in range(N_FOLDS))
    sy = sum(acc[("ysum", cls, f, L, k)][0] for f in range(N_FOLDS))
    syy = sum(acc[("yty", cls, f, L, k)][0] for f in range(N_FOLDS))
    cov = sfy / n - (sf / n) * (sy / n)
    vf = sff / n - (sf / n) ** 2
    vy = syy / n - (sy / n) ** 2
    r2 = cov * cov / (vf * vy + 1e-300)
    j = int(np.argmax(r2))
    return FEAT[j], float(r2[j])

res = {"features": FEAT, "lags": LAGS, "stride": STRIDE, "t_min": T_MIN, "classes": {}}
for cls in CLASSES:
    res["classes"][str(cls)] = {}
    for k, tname in enumerate(("t0", "t1")):
        rows = []
        for L in LAGS:
            name, r2u = uni_best(cls, L, k)
            rows.append({"lag": L, "cv_r2_all_features": oof_r2(cls, L, k),
                         "best_single_feature": name, "best_single_r2": r2u})
        res["classes"][str(cls)][tname] = rows
        best = max(rows, key=lambda r: r["cv_r2_all_features"])
        print(cls, tname, "best lag", best["lag"], round(best["cv_r2_all_features"], 4),
              "L0 r2", round(rows[0]["cv_r2_all_features"], 4), flush=True)

(OUT / "H281_horizon_20261008.json").write_text(json.dumps(res, indent=2) + "\n")
print("wrote", f"t={time.time()-t0_clock:.0f}s", flush=True)
