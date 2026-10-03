# Within-class dependence screen on scored rows: Pearson r vs cross-fitted correlation ratio eta (nonlinear mean)
# vs binned MI with discrete target; MI_gauss = -0.5 log(1-r^2) is what a linear-Gaussian world would give.
import numpy as np, sys, pandas as pd
sys.path.insert(0, "research/campaign_2026-10-03/eda/nonlinear")
from nlcommon import *
VA = dict(np.load(OUT + "/cache_valid40.npz")); SN = list(VA["SN"])
names = FEATS + ["d_" + f for f in FEATS] + SN
rows = []
for u in np.unique(VA["U"]):
    m = VA["SC"] & (VA["U"] == u)
    M = np.hstack([VA["X"][m], VA["DX"][m], VA["S"][m]]); g = VA["G"][m]
    ug = np.unique(g); fold = np.isin(g, ug[::2])
    for k in range(2):
        y = np.clip(VA["Y"][m, k], -2, 2); kk = np.clip(VA["K"][m, k], -8, 12)
        cats, yc = np.unique(kk, return_inverse=True)
        for j in range(M.shape[1]):
            x = M[:, j]
            if x.std() == 0 or len(np.unique(x)) < 3: continue
            r = np.corrcoef(x, y)[0, 1]
            q = np.unique(np.quantile(x, np.linspace(0, 1, 17)[1:-1])); b = np.searchsorted(q, x); nb = len(q) + 1
            # cross-fitted bin means
            pred = np.zeros_like(y)
            for tr, te in [(fold, ~fold), (~fold, fold)]:
                s = np.bincount(b[tr], y[tr], nb); c = np.bincount(b[tr], minlength=nb)
                mu = np.where(c > 0, s / np.maximum(c, 1), y[tr].mean()); pred[te] = mu[b[te]]
            eta = np.corrcoef(pred, y)[0, 1]
            J = np.zeros((nb, len(cats))); np.add.at(J, (b, yc), 1); J /= J.sum()
            px = J.sum(1, keepdims=True); py = J.sum(0, keepdims=True); nz = J > 0
            mi = (J[nz] * np.log(J[nz] / (px @ py)[nz])).sum() - (nz.sum() - (px > 0).sum() - (py > 0).sum() + 1) / (2 * len(y))
            rows.append((u, k, names[j], r, eta, mi, -0.5 * np.log(1 - r * r)))
df = pd.DataFrame(rows, columns=["unit", "k", "feat", "r", "eta_cv", "mi", "mi_gauss"])
df["mi_ratio"] = df.mi / (df.mi_gauss + 1e-4)
df.to_csv(OUT + "/mi_within.csv", index=False)
pd.set_option("display.width", 200)
for u in np.unique(df.unit):
    s = df[(df.unit == u) & (df.k == 0)]
    print(f"\n=== unit {u:.3f} t0: top by cv-eta"); print(s.sort_values("eta_cv", ascending=False).head(12).round(4).to_string(index=False))
    print(f"--- unit {u:.3f} t0: largest eta_cv - |r| (nonlinear mean)"); s2 = s.assign(gap=s.eta_cv - s.r.abs()); print(s2.sort_values("gap", ascending=False).head(8).round(4).to_string(index=False))
