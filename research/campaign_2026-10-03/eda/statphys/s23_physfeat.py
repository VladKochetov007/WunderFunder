# Cheapest test of the physics features: causal event-time integrator of the observed price step (move detector)
# + multi-scale event-time EMAs of it, + dwell-time features, fed with current features to an |y|-weighted ridge.
# Fit on train sample, score WP on valid scored rows (overall and per tick class).
import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
import numba as nb
pft = pq.ParquetFile(TRAIN); pfv = pq.ParquetFile(VALID)
rng = np.random.default_rng(14)
tr_rgs = rng.choice(pft.metadata.num_row_groups, 80, replace=False)
va_rgs = rng.choice(pfv.metadata.num_row_groups, 60, replace=False)
# 1) move detector on train events (first 30 seqs): dX(event) ~ [dF, F, Fprev] of i0 price+vol columns
A, B = [], []
for rg in tr_rgs[:30]:
    X, Y, u, _ = load(pft, rg); T = np.round(Y / u)
    pch = np.r_[False, np.any(np.diff(X[:, :22], axis=0) != 0, axis=1)]; idx = np.where(pch)[0]
    A.append(np.hstack([X[idx, :44] - X[idx - 1, :44], X[idx, :44], X[idx - 1, :44]]))
    B.append(((T[idx, 0] - T[idx, 1]) - (T[idx - 1, 0] - T[idx - 1, 1])) / 2)
A = np.concatenate(A).astype(np.float64); B = np.concatenate(B); keep = np.abs(B) <= 1
Z = np.hstack([A, np.ones((len(A), 1))])[keep]
beta_md = np.linalg.solve(Z.T @ Z + np.eye(Z.shape[1]), Z.T @ (-B[keep]))   # predicts observed step dP = -dX
print('move detector fitted on', keep.sum(), 'events; in-sample corr', np.corrcoef(Z @ beta_md, -B[keep])[0, 1].round(3))
HL = np.array([1.0, 4.0, 16.0, 64.0])
@nb.njit(cache=True)
def integrate(pch, dp, hl):
    n = len(pch); K = len(hl); out = np.zeros((n, K + 4), np.float32)
    a = 1.0 - np.exp(np.log(0.5) / hl)
    ema = np.zeros(K); last = 0.0; since = 0.0; rate = 0.0; cum = 0.0
    for t in range(n):
        since += 1.0
        if pch[t]:
            for k in range(K): ema[k] = (1 - a[k]) * ema[k] + a[k] * dp[t]
            last = dp[t]; rate = 0.95 * rate + 0.05 * (1.0 / since); since = 0.0; cum += dp[t]
        for k in range(K): out[t, k] = ema[k]
        out[t, K] = last; out[t, K + 1] = np.log1p(since); out[t, K + 2] = np.log(rate + 1e-4); out[t, K + 3] = cum
    return out
def phys(X):
    pch = np.r_[False, np.any(np.diff(X[:, :22], axis=0) != 0, axis=1)]
    Xp = np.vstack([X[:1], X[:-1]])
    dp = np.hstack([X[:, :44] - Xp[:, :44], X[:, :44], Xp[:, :44], np.ones((len(X), 1), np.float32)]) @ beta_md
    dp = np.where(pch, dp, 0.0)
    P = integrate(pch, dp.astype(np.float64), HL)
    # detrended integrated path: cum minus its slow EMA is a displacement ("potential" coordinate)
    return P
def build(pf, rgs, step, extra=()):
    Xs, Ps, Ys, Ss, Us = [], [], [], [], []
    for rg in rgs:
        X, Y, u, ex = load(pf, rg, extra=extra)
        P = phys(X)
        sl = slice(99, None, step)
        Xs.append(X[sl]); Ps.append(P[sl, :7]); Ys.append(Y[sl]); Us.append(np.full(len(Y[sl]), round(u, 4)))
        Ss.append(ex['is_scored'][sl].astype(bool) if 'is_scored' in ex else np.ones(len(Y[sl]), bool))
    return np.concatenate(Xs), np.concatenate(Ps), np.concatenate(Ys), np.concatenate(Ss), np.concatenate(Us)
Xtr, Ptr, Ytr, _, Utr = build(pft, tr_rgs[30:], 4)
Xva, Pva, Yva, Sva, Uva = build(pfv, va_rgs, 1, extra=('is_scored',))
m = Sva; Xva, Pva, Yva, Uva = Xva[m], Pva[m], Yva[m], Uva[m]
print('train rows', len(Ytr), 'valid scored rows', len(Yva))
def ridge(Z, y, w, lam=30.0):
    Za = np.hstack([Z, np.ones((len(Z), 1))]).astype(np.float64)
    return np.linalg.solve((Za * w[:, None]).T @ Za + lam * np.eye(Za.shape[1]), (Za * w[:, None]).T @ y)
def ev(Ztr, Zva, name, weighted=True):
    P = np.zeros_like(Yva, dtype=np.float64)
    for k in range(2):
        y = np.clip(Ytr[:, k], -2, 2).astype(np.float64); w = np.abs(y) if weighted else np.ones_like(y)
        b = ridge(Ztr, y, w); P[:, k] = np.hstack([Zva, np.ones((len(Zva), 1))]) @ b
    s = (wp(Yva[:, 0], P[:, 0]) + wp(Yva[:, 1], P[:, 1])) / 2
    per = {c: round((wp(Yva[Uva == c, 0], P[Uva == c, 0]) + wp(Yva[Uva == c, 1], P[Uva == c, 1])) / 2, 3) for c in np.unique(Uva)}
    print(f'{name:45s} WP={s:.4f}  per-class {per}')
    return P
ev(Xtr, Xva, 'current 112 features, unweighted', weighted=False)
ev(Xtr, Xva, 'current 112 features, |y|-weighted')
ev(np.hstack([Xtr, Ptr[:, :6]]), np.hstack([Xva, Pva[:, :6]]), '+ event-time EMAs of observed steps (|y|-w)')
ev(Ptr[:, :6], Pva[:, :6], 'event-time physics features only (6)')
