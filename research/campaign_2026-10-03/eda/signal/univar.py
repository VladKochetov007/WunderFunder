"""Univariate pooled correlation of every level input with t0, t1, s=t0+t1, d=t0-t1 (rows >= 99)."""
import numpy as np
import sio, feat
rng = np.random.default_rng(16)
idx = rng.choice(sio.n_seq(), 60, replace=False)
cols = feat.INP
Sx = Sxx = Sy = Syy = Sxy = 0; n = 0
for rg in idx:
    d = sio.load(int(rg))
    X, _ = feat.clean(feat.raw(d)); X = X[99:].astype(np.float64)
    y0, y1 = d['t0'][99:].astype(np.float64), d['t1'][99:].astype(np.float64)
    Y = np.c_[y0, y1, y0 + y1, y0 - y1]
    Sx += X.sum(0); Sxx += (X ** 2).sum(0); Sy += Y.sum(0); Syy += (Y ** 2).sum(0); Sxy += X.T @ Y; n += len(X)
mx, my = Sx / n, Sy / n
cov = Sxy / n - np.outer(mx, my); C = cov / np.sqrt(np.outer(Sxx / n - mx ** 2, Syy / n - my ** 2))
for j, t in enumerate(('t0', 't1', 's=t0+t1', 'd=t0-t1')):
    o = np.argsort(-np.abs(C[:, j]))[:14]
    print(t, ' '.join(f'{cols[i]}:{C[i, j]:+.3f}' for i in o))
np.savez('univar.npz', C=C, cols=np.array(cols))
