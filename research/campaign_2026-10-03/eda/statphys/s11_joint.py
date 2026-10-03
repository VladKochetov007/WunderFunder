# Joint distribution of (T0,T1) in tick units per class; sign structure of t0+t1
import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
import pandas as pd
d = pd.DataFrame(np.load(OUT + 'data/census_train.npy')[:, :3], columns=['seq', 'rg', 'u']); d['u'] = d.u.round(4)
pf = pq.ParquetFile(TRAIN)
rng = np.random.default_rng(5)
for u in [0.6569, 0.4685, 0.2413, 0.1254]:
    rgs = rng.choice(d.rg[d.u == u].values.astype(int), 10, replace=False)
    T = []
    for rg in rgs:
        t = pf.read_row_group(int(rg), columns=['t0', 't1'])
        T.append(np.stack([t.column('t0').to_numpy(), t.column('t1').to_numpy()], 1)[99:] / u)
    T = np.round(np.concatenate(T)).astype(int)
    s = T[:, 0] + T[:, 1]; df = T[:, 0] - T[:, 1]
    print(f'unit {u}: mean T0 {T[:,0].mean():.3f} T1 {T[:,1].mean():.3f}  mean(T0+T1) {s.mean():.3f}  P(T0+T1<0) {np.mean(s<0):.3f} P(=0) {np.mean(s==0):.3f} P(>0) {np.mean(s>0):.3f}')
    vals, cnt = np.unique(s, return_counts=True); o = np.argsort(-cnt)[:9]
    print('   T0+T1 top values', dict(zip(vals[o], (cnt[o] / len(s)).round(3))))
    vals, cnt = np.unique(df, return_counts=True); o = np.argsort(-cnt)[:9]
    print('   T0-T1 top values', dict(zip(vals[o], (cnt[o] / len(s)).round(3))))
    tab = pd.crosstab(np.clip(T[:, 0], -4, 4), np.clip(T[:, 1], -4, 4), normalize=True).round(3)
    print(tab.to_string())
