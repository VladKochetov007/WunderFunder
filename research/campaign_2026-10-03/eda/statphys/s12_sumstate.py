# Is T0+T1 a deterministic function of the CURRENT i0 book state (22 price-like columns)? Is T0-T1?
import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
import pandas as pd
d = pd.DataFrame(np.load(OUT + 'data/census_train.npy')[:, :3], columns=['seq', 'rg', 'u']); d['u'] = d.u.round(4)
pf = pq.ParquetFile(TRAIN)
rng = np.random.default_rng(6)
for u in [0.6569, 0.4685, 0.2413, 0.1254]:
    for rg in rng.choice(d.rg[d.u == u].values.astype(int), 3, replace=False):
        X, Y, uu, _ = load(pf, rg)
        T = np.round(Y / uu).astype(int)
        key = pd.util.hash_pandas_object(pd.DataFrame(X[:, :22]), index=False).values
        df = pd.DataFrame({'k': key, 's': T[:, 0] + T[:, 1], 'd': T[:, 0] - T[:, 1], 't0': T[:, 0]})
        g = df.groupby('k')
        # purity: fraction of rows whose s equals the modal s of their state
        pur_s = (g.s.transform(lambda x: x.mode().iloc[0]) == df.s).mean()
        pur_d = (g.d.transform(lambda x: x.mode().iloc[0]) == df.d).mean()
        pur_t0 = (g.t0.transform(lambda x: x.mode().iloc[0]) == df.t0).mean()
        print(f'u={u} rg={rg} states={g.ngroups} rows/state={len(df)/g.ngroups:.1f}  purity S={pur_s:.3f}  D={pur_d:.3f}  T0={pur_t0:.3f}')
