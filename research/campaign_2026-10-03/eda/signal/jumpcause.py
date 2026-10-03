"""What moves with the target at the moment it jumps? corr(dt(n), dx(n)) and corr(dt(n), x(n)) on i0-price-event rows."""
import numpy as np
import sio, feat
rng = np.random.default_rng(11)
idx = rng.choice(sio.n_seq(), 40, replace=False)
cols = feat.INP
acc = {}
for rg in idx:
    d = sio.load(int(rg))
    e0p, _, _ = feat.clocks(d)
    ev = np.where(e0p)[0]; ev = ev[ev >= 99]
    X = feat.raw(d)[ev]
    for t in ('t0', 't1'):
        y = d[t][ev].astype(float); dy = np.diff(y)
        if dy.std() == 0: continue
        dX = np.diff(X, axis=0)
        for name, M, yy in (('dx', dX, dy), ('x', X[1:], dy), ('dx_lag1', dX[:-1], dy[1:])):
            sd = M.std(0); ok = sd > 0
            c = np.zeros(M.shape[1]); c[ok] = [np.corrcoef(M[:, j], yy)[0, 1] for j in np.where(ok)[0]]
            a = acc.setdefault((t, name), [0, 0]); a[0] = a[0] + c; a[1] += 1
for (t, name), (s, n) in acc.items():
    c = s / n; o = np.argsort(-np.abs(c))[:12]
    print(t, name, ' '.join(f'{cols[j]}:{c[j]:+.3f}' for j in o))
