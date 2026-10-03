import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from util import *; from metric import wcorr
d = np.load("scan1_train.npz"); qa = np.round(np.minimum(d["q0"], d["q1"]), 5)
dv = np.load("scan1_valid.npz"); qv = np.round(np.minimum(dv["q0"], dv["q1"]), 5)
pc = [f"i0_p{i}" for i in range(22)]
def get(name, rg):
    df = load(name, rg, pc + ["t0", "t1"] + (["is_scored"] if name == "valid" else []))
    X = df[pc].values.astype(np.float32)
    m = df.is_scored.values.copy() if name == "valid" else np.ones(len(df), bool)
    keep = np.arange(len(df)) >= 99
    return X[keep], df.t0.values[keep], df.t1.values[keep], m[keep]
for cls in [0.65687, 0.12543]:
    tr = np.flatnonzero(qa == cls)[:30]; va = np.flatnonzero(qv == cls)[:30]
    A = [get("train", r) for r in tr]
    X = np.concatenate([a[0][::3] for a in A]); s = np.concatenate([a[1][::3] + a[2][::3] for a in A])
    ms = HistGradientBoostingRegressor(max_iter=200).fit(X, s)
    B = [get("valid", r) for r in va]
    Xv = np.concatenate([b[0] for b in B]); sv = np.concatenate([b[1] + b[2] for b in B]); mv = np.concatenate([b[3] for b in B])
    sh = ms.predict(Xv)
    r2 = lambda y, p: 1 - ((y - p)**2).mean()/y.var()
    print(f"class {cls}: R2 sum all rows {r2(sv, sh):.3f}  scored rows {r2(sv[mv], sh[mv]):.3f}  corr all {np.corrcoef(sv, sh)[0,1]:.3f} corr scored {np.corrcoef(sv[mv], sh[mv])[0,1]:.3f}")
    # per-sequence: correlation of in-sample per-seq linear fit vs global
    within = []
    for b in B:
        Xb = np.c_[b[0], np.ones(len(b[0]))]; yb = b[1] + b[2]
        coef = np.linalg.lstsq(Xb, yb, rcond=None)[0]; within.append(r2(yb, Xb @ coef))
    print(f"   per-seq in-sample linear R2 of sum on 22 i0 price cols: median {np.median(within):.3f}")
    # mean of sum on scored vs unscored
    print(f"   mean sum scored {sv[mv].mean():.3f} unscored {sv[~mv].mean():.3f}; std scored {sv[mv].std():.3f} unscored {sv[~mv].std():.3f}")
