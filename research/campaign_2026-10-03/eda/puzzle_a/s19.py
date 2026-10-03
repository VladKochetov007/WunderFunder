import numpy as np, pandas as pd
from util import *
d = np.load("scan1_valid.npz"); qa = np.round(np.minimum(d["q0"], d["q1"]), 5)
sc_all = np.load("scored_valid.npy")
print("scored frac by class:", {c: sc_all[qa==c].mean().round(4) for c in np.unique(qa)})
print("scored frac by step bucket:", [sc_all[:, i:i+2000].mean().round(4) for i in range(0, 20000, 2000)])
print("scored in warmup rows <99:", sc_all[:, :99].sum())
for rg in [0, 1, 2, 3, 10]:
    df = load("valid", rg)
    s = df.is_scored.values
    X0 = df[[c for c in df.columns if c.startswith("i0_")]].values; X1 = df[[c for c in df.columns if c.startswith("i1_")]].values
    P = df[[f"i0_p{i}" for i in range(22)]].values
    e0 = np.r_[True, np.any(X0[1:] != X0[:-1], 1)]; e1 = np.r_[True, np.any(X1[1:] != X1[:-1], 1)]; pc = np.r_[True, np.any(P[1:] != P[:-1], 1)]
    D = df[[f"i0_dv{i}" for i in range(4)]].values; tr = np.r_[True, np.any(D[1:] != D[:-1], 1)]
    tchg = np.r_[True, (np.diff(df.t0.values)!=0)|(np.diff(df.t1.values)!=0)]
    print(f"rg {rg} q {qa[rg]} scored {s.mean():.3f} | P(s|i0ev) {s[e0].mean():.3f} P(s|~i0ev) {s[~e0].mean():.3f} | P(s|pricechg) {s[pc].mean():.3f} P(s|~pc) {s[~pc].mean():.3f} | P(s|i0trade) {s[tr].mean():.3f} | P(s|tchg) {s[tchg].mean():.3f}")
    a = np.abs(df.t0.values) + np.abs(df.t1.values)
    print("    mean|t| scored/not", a[s].mean().round(3), a[~s].mean().round(3), " mean a7 scored/not", df.a7.values[s].mean().round(3), df.a7.values[~s].mean().round(3))
