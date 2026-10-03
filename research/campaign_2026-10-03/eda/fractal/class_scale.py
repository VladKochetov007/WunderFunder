"""Scale-class (target quantum) aware calibration. Class detected causally from warm-up rows (0..98) by nearest centroid.
Compare pooled linear model vs per-class linear models (detected class) vs oracle class; plus oracle per-seq mean (upper bound)."""
import sys; sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/fractal')
import numpy as np, fx
F = fx.FEAT; n = len(F); CUT = np.array([0.2, 0.35, 0.55])
I1V = [fx.IDX[f'i1_v{k}'] for k in range(22)]


def warm_stats(X):
    W = X[:99]
    return np.r_[W.mean(0), W.std(0), [len(np.unique(W[:, j])) / 99 for j in I1V]]


XtX = np.zeros((5, 2, n + 1, n + 1)); Xty = np.zeros((5, 2, n + 1)); ST = []; CL = []
for g in range(1, 10607, 106):  # 101 train seqs
    d = fx.read_seq(fx.TRAIN, g, F + ['t0', 't1']); X = np.stack([d[c] for c in F], 1).astype(np.float64)
    y = np.abs(np.r_[d['t0'], d['t1']]); c = int(np.searchsorted(CUT, y[y > 0].min()))
    ST.append(warm_stats(X)); CL.append(c)
    A = np.c_[X, np.ones(20000)][99:]
    for j, tc in enumerate(['t0', 't1']):
        yy = d[tc][99:].astype(np.float64); w = np.abs(np.clip(yy, -2, 2)); Aw = A * w[:, None]
        G = Aw.T @ A; h = Aw.T @ yy
        XtX[4, j] += G; Xty[4, j] += h; XtX[c, j] += G; Xty[c, j] += h
ST = np.array(ST); CL = np.array(CL); print('train class counts', np.bincount(CL, minlength=4))
mu, sd = ST.mean(0), ST.std(0) + 1e-9; Zt = (ST - mu) / sd
cen = np.array([Zt[CL == k].mean(0) for k in range(4)])
lam = 1e-1
Bp = [np.linalg.solve(XtX[4, j] + lam * np.eye(n + 1), Xty[4, j]) for j in range(2)]
Bc = [[np.linalg.solve(XtX[k, j] + lam * np.eye(n + 1), Xty[k, j]) for j in range(2)] for k in range(4)]
Y = [[], []]; P = {k: [[], []] for k in ['pooled', 'per-class (detected)', 'per-class (oracle)', 'pooled + oracle seq-mean shift']}; acc = []
for g in range(5, 1873, 19):  # 99 valid seqs
    d = fx.read_seq(fx.VALID, g, F + ['t0', 't1', 'is_scored']); X = np.stack([d[c] for c in F], 1).astype(np.float64); m = d['is_scored']
    y = np.abs(np.r_[d['t0'], d['t1']]); c_true = int(np.searchsorted(CUT, y[y > 0].min()))
    z = (warm_stats(X) - mu) / sd; c_det = int(np.argmin(((cen - z) ** 2).sum(1))); acc.append(c_det == c_true)
    A = np.c_[X, np.ones(20000)]
    for j, tc in enumerate(['t0', 't1']):
        yy = d[tc]; Y[j].append(yy[m]); pp = A @ Bp[j]
        P['pooled'][j].append(pp[m]); P['per-class (detected)'][j].append((A @ Bc[c_det][j])[m]); P['per-class (oracle)'][j].append((A @ Bc[c_true][j])[m])
        P['pooled + oracle seq-mean shift'][j].append(pp[m] + (yy[99:].mean() - pp[99:].mean()))
print('class detection accuracy on valid: %.3f' % np.mean(acc))
for k in P:
    r = [fx.wp(np.concatenate(Y[j]), np.concatenate(P[k][j])) for j in range(2)]
    print('%-32s WP t0 %.4f t1 %.4f mean %.4f' % (k, r[0], r[1], np.mean(r)))
