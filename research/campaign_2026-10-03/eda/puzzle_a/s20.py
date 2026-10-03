import numpy as np, pandas as pd
from util import *
for rg in [0, 3]:
    df = load("train", rg)
    X0 = df[[c for c in df.columns if c.startswith("i0_")]].values; X1 = df[[c for c in df.columns if c.startswith("i1_")]].values
    e0 = np.r_[True, np.any(X0[1:] != X0[:-1], 1)]; e1 = np.r_[True, np.any(X1[1:] != X1[:-1], 1)]
    P0 = df[[f"i0_p{i}" for i in range(22)]].values; pc0 = np.r_[True, np.any(P0[1:] != P0[:-1], 1)]
    P1 = df[[f"i1_p{i}" for i in range(22)]].values; pc1 = np.r_[True, np.any(P1[1:] != P1[:-1], 1)]
    D0 = df[[f"i0_dv{i}" for i in range(4)]+[f"i0_dp{i}" for i in range(4)]].values; tr0 = np.r_[True, np.any(D0[1:] != D0[:-1], 1)]
    D1 = df[[f"i1_dv{i}" for i in range(4)]+[f"i1_dp{i}" for i in range(4)]].values; tr1 = np.r_[True, np.any(D1[1:] != D1[:-1], 1)]
    ev = {"e0": e0, "e1": e1, "pc0": pc0, "pc1": pc1, "tr0": tr0, "tr1": tr1, "e0only": e0 & ~e1, "e1only": e1 & ~e0, "both": e0 & e1}
    print("rg", rg, {k: round(v.mean(), 3) for k, v in ev.items()})
    for a in range(8):
        x = df[f"a{a}"].values
        mn = x <= x.min() + 1e-6
        u = np.unique(x)
        rnk = np.searchsorted(u, x)
        inc = np.r_[0, np.diff(rnk)]
        line = f"  a{a}: nuniq {len(u)} P(min) {mn.mean():.3f}"
        line += " | P(min|ev): " + " ".join(f"{k}={mn[v].mean():.2f}" for k, v in ev.items())
        line += f" | corr(x, e0)={np.corrcoef(x, e0)[0,1]:.2f} corr(x,tr0)={np.corrcoef(x, tr0)[0,1]:.2f} corr(x,tr1)={np.corrcoef(x, tr1)[0,1]:.2f} corr(x,pc0)={np.corrcoef(x, pc0)[0,1]:.2f}"
        print(line)
    # a7 rank increments
    x = df.a7.values; u = np.unique(x); r = np.searchsorted(u, x)
    print("  a7 rank seq sample:", r[1000:1060])
    print("  e0 sample          :", e0[1000:1060].astype(int))
    print("  e1 sample          :", e1[1000:1060].astype(int))
    print("  tr1 sample         :", tr1[1000:1060].astype(int))
