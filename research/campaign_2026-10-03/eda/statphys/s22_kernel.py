# Event-time memory kernel: regress remaining future move X(k)=(T0-T1)/2 on past observed half-steps dP(k-j)=-dX(k-j)
# (dX restricted to |dX|<=1 so that end-reference jumps are excluded). Martingale => zero kernel.
import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
import pandas as pd
d = pd.DataFrame(np.load(OUT + 'data/census_train.npy')[:, :3], columns=['seq', 'rg', 'u']); d['u'] = d.u.round(4)
pf = pq.ParquetFile(TRAIN)
rng = np.random.default_rng(13)
J = 30; res = {}
for u in [0.6569, 0.4685, 0.2413, 0.1254]:
    num = np.zeros(J); den = np.zeros(J); cnt = np.zeros(J)
    cond = {}
    for rg in rng.choice(d.rg[d.u == u].values.astype(int), 30, replace=False):
        t = pf.read_row_group(int(rg), columns=[f'i0_p{j}' for j in range(22)] + ['t0', 't1'])
        P = np.stack([t.column(f'i0_p{j}').to_numpy() for j in range(22)], 1)
        T = np.round(np.stack([t.column('t0').to_numpy(), t.column('t1').to_numpy()], 1) / u)
        pch = np.r_[False, np.any(np.diff(P, axis=0) != 0, axis=1)]
        idx = np.where(pch)[0]
        Xk = (T[idx, 0] - T[idx, 1]) / 2
        dX = Xk - (T[idx - 1, 0] - T[idx - 1, 1]) / 2
        dP = np.where(np.abs(dX) <= 1, -dX, 0.0)
        for j in range(J):
            a = Xk[j:]; b = dP[:len(dP) - j] if j > 0 else dP
            num[j] += (a * b).sum(); den[j] += (b * b).sum(); cnt[j] += len(a)
        # conditional mean of X after the last two observed half-steps
        s1 = np.sign(dP); 
        for k in range(2, len(idx)):
            key = (s1[k - 1], s1[k])
            c = cond.setdefault(key, [0, 0.0]); c[0] += 1; c[1] += Xk[k]
    res[u] = num / den
    print(f'u={u} kernel slope X(k)~dP(k-j), j=0..9:', (num / den)[:10].round(3), ' sum j<30:', (num / den).sum().round(3))
    print('    E[X | sign dP(k-1), sign dP(k)]:', {k: (v[0], round(v[1] / v[0], 3)) for k, v in sorted(cond.items()) if v[0] > 200})
np.savez(OUT + 'data/kernel.npz', **{f'u{u}': v for u, v in res.items()})
