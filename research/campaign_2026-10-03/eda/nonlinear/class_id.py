# How fast can the per-sequence tick/vol class (target unit) be identified causally from the first N rows?
import numpy as np, sys
sys.path.insert(0, "research/campaign_2026-10-03/eda/nonlinear")
from nlcommon import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
NS = [20, 50, 99, 300, 1000, 5000]
cols = [f"i0_p{i}" for i in range(22)] + [f"i0_dp{i}" for i in range(4)] + [f"i0_dv{i}" for i in range(4)] + ["a2", "a3", "a5", "a7", "t0", "t1"]
pf = pfile("train"); ids = seq_ids("train", 300, seed=11)
F = {n: [] for n in NS}; Y = []
for rg in ids:
    tb = pf.read_row_group(int(rg), columns=cols)
    d = {c: tb.column(c).to_numpy() for c in cols}
    Y.append(round(unit_of(np.concatenate([d["t0"], d["t1"]])), 3))
    P = np.stack([d[f"i0_p{i}"] for i in range(22)], 1)
    for n in NS:
        p = P[:n]; dp = np.diff(p, axis=0)
        nz = np.abs(dp[dp != 0])
        f = [np.mean([len(np.unique(p[:, i])) for i in range(22)]) / n, (np.abs(dp).sum(1) > 0).mean(),
             np.median(nz) if len(nz) else 0, np.log1p(len(np.unique(p[:, 0]))),
             d["a2"][:n].mean(), d["a3"][:n].mean(), d["a5"][:n].mean(), d["a7"][:n].mean(),
             (np.diff(np.stack([d[f"i0_dv{i}"] for i in range(4)], 1)[:n], axis=0) != 0).any(1).mean()]
        F[n].append(f)
Y = np.array(Y); print("classes", np.unique(Y, return_counts=True))
for n in NS:
    X = np.array(F[n]); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    pr = cross_val_predict(LogisticRegression(max_iter=2000, C=3.0), X, Y.astype(str), cv=5)
    acc = (pr == Y.astype(str)).mean()
    coarse = np.isin(Y, [0.469, 0.657]); prc = np.isin(pr, ["0.469", "0.657"])
    print(f"first {n:5d} rows: 4-class acc {acc:.3f}   coarse(>=0.469)-vs-fine acc {(coarse == prc).mean():.3f}")
