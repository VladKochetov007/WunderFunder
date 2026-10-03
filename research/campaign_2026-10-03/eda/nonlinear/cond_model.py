# Baseline models: weighted ridge (linear) and HistGBR (nonlinear) on train sample; predictions on valid sample.
import numpy as np, sys, time
sys.path.insert(0, "research/campaign_2026-10-03/eda/nonlinear")
from nlcommon import *
from sklearn.ensemble import HistGradientBoostingRegressor
TR = dict(np.load(OUT + "/cache_train48.npz")); VA = dict(np.load(OUT + "/cache_valid40.npz"))

def design(D):
    S = D["S"].copy(); S[:, -1] /= 20000.0
    return np.hstack([D["X"], D["DX"], S]).astype(np.float32)

def wp(y, p):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    a = y - np.average(y, weights=w); b = p - np.average(p, weights=w)
    return np.sum(w * a * b) / np.sqrt(np.sum(w * a * a) * np.sum(w * b * b))

Xtr = design(TR); Xva = design(VA)
mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
Ztr = (Xtr - mu) / sd; Zva = (Xva - mu) / sd
sc = VA["SC"]
preds = {}
for k in range(2):
    y = np.clip(TR["Y"][:, k], -2, 2); w = np.abs(y) + 1e-3
    A = Ztr * np.sqrt(w)[:, None]; b = y * np.sqrt(w)
    A1 = np.hstack([A, np.sqrt(w)[:, None]])
    lam = 10.0 * len(y) / 1000
    G = A1.T @ A1 + lam * np.eye(A1.shape[1]); G[-1, -1] -= lam
    beta = np.linalg.solve(G, A1.T @ b)
    p = np.hstack([Zva, np.ones((len(Zva), 1))]) @ beta
    preds[f"ridge_t{k}"] = p.astype(np.float32)
    print(f"ridge t{k}: WP scored {wp(VA['Y'][sc, k], p[sc]):.4f}  all-sample {wp(VA['Y'][:, k], p):.4f}", flush=True)
np.savez(OUT + "/preds_valid40.npz", **preds)
del Ztr, Zva, A, A1, G
import gc; gc.collect()
rs = np.random.default_rng(0).choice(len(Xtr), 150000, replace=False); rs.sort()
Xtr = Xtr[rs]; TRY = TR["Y"][rs]; del TR; gc.collect()
for k in range(2):
    t = time.time()
    y = np.clip(TRY[:, k], -2, 2); w = np.abs(y) + 1e-3
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200,
                                      l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=0)
    m.fit(Xtr, y, sample_weight=w)
    p = m.predict(Xva)
    preds[f"hgb_t{k}"] = p.astype(np.float32)
    print(f"hgb t{k}: iters {m.n_iter_} WP scored {wp(VA['Y'][sc, k], p[sc]):.4f}  all-sample {wp(VA['Y'][:, k], p):.4f}  {time.time()-t:.0f}s", flush=True)
    np.savez(OUT + "/preds_valid40.npz", **preds)
