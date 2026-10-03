# Which feature increments carry the price increment that enters the target? corr(dt0(t), dF(t))
import pyarrow.parquet as pq, numpy as np
pf = pq.ParquetFile('datasets/train.parquet')
names = pf.schema_arrow.names
feat = names[3:115]
rng = np.random.default_rng(1)
rgs = rng.choice(pf.metadata.num_row_groups, 20, replace=False)
S = np.zeros((2, 112)); SS = np.zeros(112); T = np.zeros(2)
for rg in rgs:
    t = pf.read_row_group(int(rg), columns=feat + ['t0', 't1']).to_pandas()
    F = t[feat].values.astype(np.float32)
    dF = np.diff(F, axis=0)
    for k, c in enumerate(['t0', 't1']):
        y = t[c].values.astype(np.float64)
        dy = np.diff(y)
        S[k] += dy @ dF
        T[k] += dy @ dy
    SS += (dF.astype(np.float64) ** 2).sum(0)
for k in range(2):
    r = S[k] / np.sqrt(T[k] * SS + 1e-12)
    o = np.argsort(r)
    print('target', k, 'most negative:', [(feat[i], round(r[i], 3)) for i in o[:12]])
    print('   most positive:', [(feat[i], round(r[i], 3)) for i in o[-8:]])
