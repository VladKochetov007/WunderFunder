import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from util import *
dv = np.load("scan1_valid.npz"); qv = np.round(np.minimum(dv["q0"], dv["q1"]), 5)
allc = None
def feats(df):
    X0 = df[[c for c in df.columns if c.startswith("i0_")]].values; X1 = df[[c for c in df.columns if c.startswith("i1_")]].values
    e0 = np.r_[True, np.any(X0[1:] != X0[:-1], 1)]; e1 = np.r_[True, np.any(X1[1:] != X1[:-1], 1)]
    P0 = df[[f"i0_p{i}" for i in range(22)]].values; pc0 = np.r_[True, np.any(P0[1:] != P0[:-1], 1)]
    D0 = df[[f"i0_dv{i}" for i in range(4)]].values; tr0 = np.r_[True, np.any(D0[1:] != D0[:-1], 1)]
    D1 = df[[f"i1_dv{i}" for i in range(4)]].values; tr1 = np.r_[True, np.any(D1[1:] != D1[:-1], 1)]
    base = df[[c for c in df.columns if c.startswith(("i0_", "i1_", "a"))]].values
    ex = np.c_[e0, e1, pc0, tr0, tr1].astype(np.float32)
    # lagged a-cols
    A = df[[f"a{i}" for i in range(8)]].values
    Al = np.r_[A[:1], A[:-1]]
    return np.c_[base, ex, Al].astype(np.float32)
for cls in [0.65687, 0.12543]:
    va = np.flatnonzero(qv == cls)[:30]
    Xs, ys = [], []
    for r in va:
        df = load("valid", r); df = df.iloc[99:]
        Xs.append(feats(df)); ys.append(df.is_scored.values)
    X = np.concatenate(Xs[:20]); y = np.concatenate(ys[:20]); Xt = np.concatenate(Xs[20:]); yt = np.concatenate(ys[20:])
    m = HistGradientBoostingClassifier(max_iter=200).fit(X, y)
    p = m.predict_proba(Xt)[:, 1]
    print(f"class {cls}: AUC of is_scored from current inputs (+event flags, lagged a) = {roc_auc_score(yt, p):.3f}  base rate {yt.mean():.3f}")
    # single feature AUCs
    names = [c for c in df.columns if c.startswith(("i0_", "i1_", "a"))] + ["e0", "e1", "pc0", "tr0", "tr1"] + [f"a{i}_lag" for i in range(8)]
    aucs = [(roc_auc_score(yt, Xt[:, j]) if Xt[:, j].std() > 0 else .5) for j in range(Xt.shape[1])]
    o = np.argsort(-np.abs(np.array(aucs) - .5))[:10]
    print("   top single-feature AUCs:", [(names[j], round(aucs[j], 3)) for j in o])
