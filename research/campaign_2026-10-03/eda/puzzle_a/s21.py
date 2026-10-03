import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from util import *; from metric import wcorr
d = np.load("scan1_train.npz"); qa = np.round(np.minimum(d["q0"], d["q1"]), 5)
dv = np.load("scan1_valid.npz"); qv = np.round(np.minimum(dv["q0"], dv["q1"]), 5)
sc = np.load("scored_valid.npy")
pc = [f"i0_p{i}" for i in range(22)]
def get(name, rg, cls):
    df = load(name, rg, pc + ["t0", "t1"] + (["is_scored"] if name == "valid" else []))
    X = df[pc].values.astype(np.float32)
    m = df.is_scored.values.copy() if name == "valid" else np.ones(len(df), bool)
    m[:99] = False
    return X[m], df.t0.values[m], df.t1.values[m]
for cls in [0.12543, 0.24133, 0.46854, 0.65687]:
    tr = np.flatnonzero(qa == cls)[:30]; va = np.flatnonzero(qv == cls)[:40]
    A = [get("train", r, cls) for r in tr]
    X = np.concatenate([a[0][::3] for a in A]); t0 = np.concatenate([a[1][::3] for a in A]); t1 = np.concatenate([a[2][::3] for a in A])
    B = [get("valid", r, cls) for r in va]
    Xv = np.concatenate([b[0] for b in B]); v0 = np.concatenate([b[1] for b in B]); v1 = np.concatenate([b[2] for b in B])
    ms = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1).fit(X, t0 + t1)
    md = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1).fit(X, t0 - t1)
    s_hat = ms.predict(Xv); d_hat = md.predict(Xv)
    sv = v0 + v1; dvv = v0 - v1
    r2s = 1 - ((sv - s_hat)**2).mean()/sv.var(); r2d = 1 - ((dvv - d_hat)**2).mean()/dvv.var()
    print(f"class {cls}: ntrain {len(X)} nvalid(scored) {len(Xv)} | R2(sum from i0 prices)={r2s:.3f} R2(diff)={r2d:.3f}")
    print(f"   score using only sum part: t0 {wcorr(s_hat/2, v0):.3f} t1 {wcorr(s_hat/2, v1):.3f} | sum+diff: t0 {wcorr((s_hat+d_hat)/2, v0):.3f} t1 {wcorr((s_hat-d_hat)/2, v1):.3f} | oracle-sum only: t0 {wcorr(sv/2, v0):.3f} t1 {wcorr(sv/2, v1):.3f}")
