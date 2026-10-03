# Target quantisation and autocorrelation structure (horizon detection)
import pyarrow.parquet as pq, numpy as np, sys
pf = pq.ParquetFile('datasets/train.parquet')
rng = np.random.default_rng(0)
rgs = rng.choice(pf.metadata.num_row_groups, 30, replace=False)
units = []
acf0 = np.zeros(3001); acf1 = np.zeros(3001); n = 0
for rg in rgs:
    t = pf.read_row_group(int(rg), columns=['t0', 't1']).to_pandas()
    for c in ['t0', 't1']:
        v = np.unique(np.abs(t[c].values.astype(np.float64)))
        v = v[v > 1e-6]
        units.append((int(rg), c, v[:4].round(6).tolist(), len(v)))
    for c, acc in [('t0', acf0), ('t1', acf1)]:
        x = t[c].values.astype(np.float64); x = x - x.mean()
        f = np.fft.rfft(x, 2 * len(x))
        a = np.fft.irfft(f * np.conj(f))[:3001]
        acc += a / a[0]
    n += 1
for u in units[:20]: print(u)
acf0 /= n; acf1 /= n
np.savez('research/campaign_2026-10-03/eda/statphys/data/acf_targets.npz', acf0=acf0, acf1=acf1)
for L in [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000]:
    print(L, round(acf0[L], 4), round(acf1[L], 4))
