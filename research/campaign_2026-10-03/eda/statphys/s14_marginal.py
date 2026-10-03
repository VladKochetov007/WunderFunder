# Marginal distribution of T0, T1 in ticks per class (look for barriers: floors/caps = first-passage stops)
import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
import pandas as pd
d = pd.DataFrame(np.load(OUT + 'data/census_train.npy')[:, :3], columns=['seq', 'rg', 'u']); d['u'] = d.u.round(4)
pf = pq.ParquetFile(TRAIN)
rng = np.random.default_rng(7)
H = {}
for u in [0.6569, 0.4685, 0.2413, 0.1254]:
    T = []
    for rg in rng.choice(d.rg[d.u == u].values.astype(int), 25, replace=False):
        t = pf.read_row_group(int(rg), columns=['t0', 't1'])
        T.append(np.stack([t.column('t0').to_numpy(), t.column('t1').to_numpy()], 1)[99:] / u)
    T = np.round(np.concatenate(T)).astype(int)
    for k in range(2):
        v, c = np.unique(T[:, k], return_counts=True)
        H[(u, k)] = (v, c)
        p = c / c.sum()
        print(f'u={u} T{k}: min {v.min()} ({v.min()*u:.2f}) max {v.max()} ({v.max()*u:.2f})')
        print('   ', ' '.join(f'{a}:{b:.3f}' for a, b in zip(v, p) if b > 0.002))
    s = T[:, 0] + T[:, 1]
    print('   P(T0>0 & T1>0)=', np.mean((T[:, 0] > 0) & (T[:, 1] > 0)).round(4), ' P(T0<=floor+1 & T1<=floor+1)', )
np.save(OUT + 'data/marginals.npy', np.array([(u, k, v, c) for (u, k), (v, c) in H.items()], dtype=object), allow_pickle=True)
