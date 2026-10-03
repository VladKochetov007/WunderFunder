# Event-level price-move detector: at i0 price-state events, X(t) := (T0-T1)/2 jumps by -dP_now (current price step)
# unless the future end-reference moves. Learn dP_now from feature increments (causal: uses only rows t-1, t).
import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
import pandas as pd
d = pd.DataFrame(np.load(OUT + 'data/census_train.npy')[:, :3], columns=['seq', 'rg', 'u']); d['u'] = d.u.round(4)
pf = pq.ParquetFile(TRAIN)
rng = np.random.default_rng(11)
out = {}
for u in [0.6569, 0.4685, 0.2413, 0.1254]:
    rgs = rng.choice(d.rg[d.u == u].values.astype(int), 24, replace=False)
    A, B, C = [], [], []
    for rg in rgs:
        X, Y, uu, _ = load(pf, rg)
        T = np.round(Y / uu)
        pch = np.r_[False, np.any(np.diff(X[:, :22], axis=0) != 0, axis=1)]
        idx = np.where(pch)[0]
        dF = (X[idx] - X[idx - 1])[:, :44]
        Fcur = X[idx][:, :44]
        dX = ((T[idx, 0] - T[idx, 1]) - (T[idx - 1, 0] - T[idx - 1, 1])) / 2
        dS = (T[idx, 0] + T[idx, 1]) - (T[idx - 1, 0] + T[idx - 1, 1])
        A.append(np.hstack([dF, Fcur, X[idx - 1][:, :44]])); B.append(np.c_[dX, dS]); C.append(np.full(len(idx), rg))
    A = np.concatenate(A).astype(np.float64); B = np.concatenate(B); C = np.concatenate(C)
    tr = np.isin(C, rgs[:16]); te = ~tr
    for k, name in [(0, 'dX=(dT0-dT1)/2'), (1, 'dSum')]:
        y = B[:, k]
        for feats, fname in [(slice(0, 44), 'dF'), (slice(0, 132), 'dF+F+Fprev')]:
            Z = np.hstack([A[:, feats], np.ones((len(A), 1))])
            beta = np.linalg.solve(Z[tr].T @ Z[tr] + 1.0 * np.eye(Z.shape[1]), Z[tr].T @ y[tr])
            p = Z[te] @ beta
            r2 = 1 - np.mean((y[te] - p) ** 2) / np.var(y[te])
            sgn = np.mean(np.sign(np.round(p))[y[te] != 0] == np.sign(y[te][y[te] != 0]))
            print(f'u={u} {name} ~ {fname}: n_events={len(y)} R2_test={r2:.3f} corr={np.corrcoef(p, y[te])[0,1]:.3f}  sign-acc on nonzero={sgn:.3f}  P(y!=0)={np.mean(y!=0):.3f}')
    v, c = np.unique(B[:, 0], return_counts=True); o = np.argsort(-c)[:7]
    print('   dX values', dict(zip(v[o], (c[o] / c.sum()).round(3))))
