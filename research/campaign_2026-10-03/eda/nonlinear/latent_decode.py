# Coarse-tick classes: joint categorical model of (k0,k1) + metric-optimal decoding vs direct weighted ridge.
import numpy as np, sys, gc
sys.path.insert(0, "research/campaign_2026-10-03/eda/nonlinear")
from nlcommon import *
from sklearn.linear_model import LogisticRegression
def design(D, m):
    S = D["S"][m].copy(); S[:, -1] /= 20000.0
    return np.hstack([D["X"][m], D["DX"][m], S]).astype(np.float32)
def wp(y, p):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    a = y - np.average(y, weights=w); b = p - np.average(p, weights=w)
    return np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))
def cat(K):
    k0, k1 = K[:, 0], K[:, 1]
    c = np.full(len(k0), 8)
    c[(k0 == 1) & (k1 <= -2)] = 0; c[(k0 == 2) & (k1 <= -2)] = 1; c[(k0 >= 3) & (k1 <= -2)] = 2
    c[(k1 == 1) & (k0 <= -2)] = 3; c[(k1 == 2) & (k0 <= -2)] = 4; c[(k1 >= 3) & (k0 <= -2)] = 5
    c[(k0 == -1) & (k1 == 0)] = 6; c[(k0 == 0) & (k1 == -1)] = 7
    return c
TR = dict(np.load(OUT + "/cache_train48.npz")); VA = dict(np.load(OUT + "/cache_valid40.npz"))
for u in [0.6568677, 0.46853608]:
    mt = np.isclose(TR["U"], u); mv = np.isclose(VA["U"], u) & VA["SC"]
    Xt = design(TR, mt); Xv = design(VA, mv); mu = Xt.mean(0); sd = Xt.std(0) + 1e-6
    Xt = (Xt - mu) / sd; Xv = (Xv - mu) / sd
    Yt = np.clip(TR["Y"][mt], -2, 2); Yv = VA["Y"][mv]; ct = cat(TR["K"][mt])
    print(f"unit {u:.3f}: train rows {mt.sum()}, valid scored {mv.sum()}, cat shares", np.round(np.bincount(ct, minlength=9) / len(ct), 3), flush=True)
    # direct weighted ridge per target (class-specific)
    res = {}
    for k in range(2):
        w = np.abs(Yt[:, k]) + 1e-3; A = np.hstack([Xt, np.ones((len(Xt), 1), np.float32)])
        G = (A * w[:, None]).T @ A + 10.0 * len(w) / 1000 * np.eye(A.shape[1]); r = (A * w[:, None]).T @ Yt[:, k]
        b = np.linalg.solve(G, r); res[f"ridge{k}"] = np.hstack([Xv, np.ones((len(Xv), 1), np.float32)]) @ b
    clf = LogisticRegression(max_iter=300, C=0.05)
    clf.fit(Xt, ct)
    P = clf.predict_proba(Xv); cl = clf.classes_
    for k in range(2):
        y = Yt[:, k]
        num = np.array([np.mean(y[ct == c] * np.abs(y[ct == c])) for c in cl]); den = np.array([np.mean(np.abs(y[ct == c])) for c in cl])
        mean = np.array([np.mean(y[ct == c]) for c in cl])
        res[f"dec_opt{k}"] = (P @ num) / (P @ den + 1e-9); res[f"dec_mean{k}"] = P @ mean
    for k in range(2):
        print(f"  t{k}: class-specific ridge WP {wp(Yv[:, k], res[f'ridge{k}']):.4f} | categorical decode E[y] {wp(Yv[:, k], res[f'dec_mean{k}']):.4f} | decode E[y|y|]/E|y| {wp(Yv[:, k], res[f'dec_opt{k}']):.4f}", flush=True)
    np.savez(OUT + f"/latent_decode_{u:.3f}.npz", **res, P=P, classes=cl)
    del Xt, Xv, A, G; gc.collect()
