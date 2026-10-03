# Centre-of-mass / relative-coordinate decomposition: S=(t0+t1)/2 (mid-like), D=(t0-t1)/2 (spread-like)
import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
pf = pq.ParquetFile(TRAIN)
rng = np.random.default_rng(3)
rgs = rng.choice(pf.metadata.num_row_groups, 48, replace=False)
stats = {}
Xtr, Ytr, Xte, Yte, Ctr, Cte = [], [], [], [], [], []
L = 2001; acfS = {}; acfD = {}
def acf(x):
    x = x - x.mean(); f = np.fft.rfft(x, 2 * len(x)); a = np.fft.irfft(f * np.conj(f))[:L]; return a
for i, rg in enumerate(rgs):
    X, Y, u, _ = load(pf, rg)
    c = ucls(u)
    T = Y[99:].astype(np.float64)
    S = (T[:, 0] + T[:, 1]) / 2; D = (T[:, 0] - T[:, 1]) / 2
    st = stats.setdefault(c, np.zeros(6))
    st += [1, (T[:, 0] ** 2).mean(), (T[:, 1] ** 2).mean(), (S ** 2).mean(), (D ** 2).mean(), (T[:, 0] * T[:, 1]).mean()]
    acfS[c] = acfS.get(c, 0) + acf(S); acfD[c] = acfD.get(c, 0) + acf(D)
    Xs = X[99::4]; Ys = Y[99::4]
    (Xtr if i < 36 else Xte).append(Xs); (Ytr if i < 36 else Yte).append(Ys); (Ctr if i < 36 else Cte).append(np.full(len(Ys), c))
for c, st in stats.items():
    n = st[0]; st = st / n
    print(f'class {c} n={int(n)}  E[t0^2]={st[1]:.3f} E[t1^2]={st[2]:.3f}  E[S^2]={st[3]:.3f} E[D^2]={st[4]:.3f}  corr~{st[5]/np.sqrt(st[1]*st[2]):.3f}  D-share={st[4]/(st[3]+st[4]):.3f}')
np.savez(OUT + 'data/acf_sumdiff.npz', **{f'S_{c}': acfS[c] / acfS[c][0] for c in acfS}, **{f'D_{c}': acfD[c] / acfD[c][0] for c in acfD})
for c in acfS:
    a, b = acfS[c] / acfS[c][0], acfD[c] / acfD[c][0]
    print(c, 'ACF S at 1,10,100,500,1000:', a[[1, 10, 100, 500, 1000]].round(3), ' D:', b[[1, 10, 100, 500, 1000]].round(3))
Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr); Xte = np.concatenate(Xte); Yte = np.concatenate(Yte); Cte = np.concatenate(Cte)
def ridge(X, y, w, lam=10.0):
    Xa = np.hstack([X, np.ones((len(X), 1), np.float32)]).astype(np.float64)
    A = (Xa * w[:, None]).T @ Xa + lam * np.eye(Xa.shape[1]); b = (Xa * w[:, None]).T @ y
    return np.linalg.solve(A, b)
def pred(X, beta): return np.hstack([X, np.ones((len(X), 1), np.float32)]) @ beta
res = {}
for wname in ['unweighted', 'absy']:
    P = np.zeros_like(Yte, dtype=np.float64)
    for k in range(2):
        y = np.clip(Ytr[:, k], -2, 2).astype(np.float64); w = np.ones_like(y) if wname == 'unweighted' else np.abs(y)
        P[:, k] = pred(Xte, ridge(Xtr, y, w))
    print(wname, 'direct ridge WP t0,t1:', round(wp(Yte[:, 0], P[:, 0]), 4), round(wp(Yte[:, 1], P[:, 1]), 4))
# sum/diff route
ys = np.clip(Ytr, -2, 2).astype(np.float64)
for wname in ['unweighted', 'absy']:
    S = (ys[:, 0] + ys[:, 1]) / 2; D = (ys[:, 0] - ys[:, 1]) / 2
    wS = np.ones_like(S) if wname == 'unweighted' else np.abs(S) + 1e-3
    wD = np.ones_like(D) if wname == 'unweighted' else np.abs(D) + 1e-3
    pS = pred(Xte, ridge(Xtr, S, wS)); pD = pred(Xte, ridge(Xtr, D, wD))
    Ste = (Yte[:, 0] + Yte[:, 1]) / 2; Dte = (Yte[:, 0] - Yte[:, 1]) / 2
    print(wname, 'corr(pS,S)=%.4f corr(pD,D)=%.4f' % (np.corrcoef(pS, Ste)[0, 1], np.corrcoef(pD, Dte)[0, 1]),
          ' WP t0 from D only %.4f, from S only %.4f, S+D %.4f' % (wp(Yte[:, 0], pD), wp(Yte[:, 0], pS), wp(Yte[:, 0], pS + pD)),
          ' t1: D-only %.4f S-only %.4f S-D %.4f' % (wp(Yte[:, 1], -pD), wp(Yte[:, 1], pS), wp(Yte[:, 1], pS - pD)))
    for c in np.unique(Cte):
        m = Cte == c
        print('   class', c, 'n', m.sum(), 'WP t0 %.4f t1 %.4f' % (wp(Yte[m, 0], (pS + pD)[m]), wp(Yte[m, 1], (pS - pD)[m])))
