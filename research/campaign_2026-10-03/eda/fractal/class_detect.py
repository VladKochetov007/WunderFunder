"""Is the per-sequence target quantum q (scale class) identifiable causally from warm-up rows? (99 and 500 rows)."""
import sys; sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/fractal')
import numpy as np, fx
F = fx.FEAT; rows = []; Q = []
for g in range(7, 1873, 31):
    d = fx.read_seq(fx.VALID, g, F + ['t0', 't1'])
    y = np.abs(np.r_[d['t0'], d['t1']]); Q.append(y[y > 0].min())
    st = []
    for W in (99, 500):
        X = np.stack([d[c][:W] for c in F], 1).astype(np.float64); dX = np.abs(np.diff(X, axis=0))
        mn = np.array([dX[dX[:, j] > 1e-6, j].min() if (dX[:, j] > 1e-6).any() else np.nan for j in range(112)])
        nu = np.array([len(np.unique(X[:, j])) for j in range(112)]) / W
        st.append(np.r_[mn, nu, X.mean(0), X.std(0)])
    rows.append(np.r_[st[0], st[1]])
R = np.array(rows); Q = np.array(Q); cls = np.searchsorted(np.array([0.2, 0.35, 0.55]), Q)
print('class counts', np.bincount(cls))
names = [f'{s}{w}_{f}' for w in (99, 500) for s in ('mindiff', 'nuniq', 'mean', 'std') for f in F]
# separability: ratio of between-class to within-class variance of log-ish stat (eta^2)
eta = []
for j in range(R.shape[1]):
    v = R[:, j]; m = np.isfinite(v)
    if m.sum() < 40 or np.nanstd(v[m]) == 0: eta.append(0); continue
    v = v[m]; c = cls[m]; tot = ((v - v.mean()) ** 2).sum(); bw = sum(((v[c == k].mean() - v.mean()) ** 2) * (c == k).sum() for k in np.unique(c))
    eta.append(bw / tot)
eta = np.array(eta); o = np.argsort(-eta)
for i in o[:15]:
    print('%-22s eta2 %.3f  class medians %s' % (names[i], eta[i], [round(float(np.nanmedian(R[cls == k, i])), 4) for k in range(4)]))
# simple nearest-centroid classifier with leave-one-out on top-k stats (99 rows only)
top = [i for i in o if names[i].endswith(tuple(F)) and '99_' in names[i]][:10]
Z = R[:, top]; Z = (Z - np.nanmean(Z, 0)) / (np.nanstd(Z, 0) + 1e-9); Z = np.nan_to_num(Z)
ok = 0
for i in range(len(Z)):
    m = np.arange(len(Z)) != i; cen = np.array([Z[m & (cls == k)].mean(0) for k in range(4)])
    ok += np.argmin(((cen - Z[i]) ** 2).sum(1)) == cls[i]
print('LOO nearest-centroid accuracy from warm-up (99 rows), top-10 stats: %.3f (chance ~%.3f)' % (ok / len(Z), np.bincount(cls).max() / len(cls)))
