# Pearson vs Spearman vs binned MI of each feature (level and 1-step diff) with targets, on scored valid rows.
import numpy as np, sys
sys.path.insert(0, "research/campaign_2026-10-03/eda/nonlinear")
from nlcommon import *
from scipy.stats import spearmanr
ids = seq_ids("valid", 40, seed=1)
X=[];DX=[];Y=[];U=[]
for rg in ids:
    d = load("valid", rg)
    m = d["is_scored"]
    x = np.stack([d[c] for c in FEATS], 1)
    dx = np.vstack([np.zeros((1, x.shape[1]), np.float32), np.diff(x, axis=0)])
    X.append(x[m]); DX.append(dx[m]); Y.append(np.stack([d["t0"], d["t1"]], 1)[m])
    U.append(np.full(m.sum(), unit_of(d["t0"]), np.float32))
X=np.concatenate(X);DX=np.concatenate(DX);Y=np.concatenate(Y);U=np.concatenate(U)
print("rows", len(Y), "units", np.unique(U, return_counts=True))
np.savez_compressed(OUT+"/cache_scored40.npz", X=X, DX=DX, Y=Y, U=U)

def mi_binned(x, y, nb=16):
    qx = np.quantile(x, np.linspace(0, 1, nb + 1)[1:-1]); bx = np.searchsorted(qx, x)
    qy = np.quantile(y, np.linspace(0, 1, nb + 1)[1:-1]); by = np.searchsorted(qy, y)
    j = np.zeros((nb, nb)); np.add.at(j, (bx, by), 1); j /= j.sum()
    px = j.sum(1, keepdims=True); py = j.sum(0, keepdims=True)
    nz = j > 0
    mi = (j[nz] * np.log(j[nz] / (px @ py)[nz])).sum()
    # Miller-Madow-ish bias correction
    mi -= ((nz.sum() - (px > 0).sum() - (py > 0).sum() + 1) / (2 * len(x)))
    return max(mi, 0.0)

def wcorr(a, b, w):
    a = a - np.average(a, weights=w); b = b - np.average(b, weights=w)
    return np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))

rows=[]
for nm, M in [("lvl", X), ("d1", DX)]:
    for j, c in enumerate(FEATS):
        x = M[:, j]
        if np.std(x) == 0: continue
        r = []
        for k in range(2):
            y = np.clip(Y[:, k], -2, 2)
            pc = np.corrcoef(x, y)[0, 1]
            wc = wcorr(x, y, np.abs(y) + 1e-9)
            sc = spearmanr(x, y).statistic
            mi = mi_binned(x + 1e-6 * np.random.randn(len(x)), y + 1e-6 * np.random.randn(len(y)))
            rlin = np.sqrt(1 - np.exp(-2 * mi))  # "information coefficient of correlation"
            r += [pc, wc, sc, mi, rlin]
        rows.append((nm, c, *r))
import pandas as pd
df = pd.DataFrame(rows, columns=["kind", "feat"] + [f"{s}_{t}" for t in ["t0", "t1"] for s in ["pear", "wpear", "spear", "mi", "rMI"]])
df["gap_t0"] = df.rMI_t0 - df.pear_t0.abs()
df["gap_t1"] = df.rMI_t1 - df.pear_t1.abs()
df.to_csv(OUT + "/feat_mi.csv", index=False)
pd.set_option("display.width", 250)
print(df.sort_values("rMI_t0", ascending=False).head(30).round(4).to_string())
print(df.sort_values("gap_t0", ascending=False).head(20).round(4).to_string())
