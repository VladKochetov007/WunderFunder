# ACF of target increments: a fixed horizon h gives a negative spike at lag h; a random/clock horizon gives a negative bump
import pyarrow.parquet as pq, numpy as np
pf = pq.ParquetFile('datasets/train.parquet')
rng = np.random.default_rng(2)
rgs = rng.choice(pf.metadata.num_row_groups, 40, replace=False)
L = 4000
out = {}
def xc(a, b):
    n = len(a); fa = np.fft.rfft(a, 2 * n); fb = np.fft.rfft(b, 2 * n)
    return np.fft.irfft(np.conj(fa) * fb)[:L + 1]   # sum_s a(s) b(s+lag)
acc = {k: np.zeros(L + 1) for k in ['d0d0', 'd1d1', 'd0d1', 'd1d0']}
units = []
for rg in rgs:
    t = pf.read_row_group(int(rg), columns=['t0', 't1']).to_pandas()
    y0 = t.t0.values.astype(np.float64); y1 = t.t1.values.astype(np.float64)
    u = np.unique(np.abs(y0)); u = u[u > 1e-6][0]
    units.append(u)
    d0 = np.diff(y0) / u; d1 = np.diff(y1) / u
    acc['d0d0'] += xc(d0, d0); acc['d1d1'] += xc(d1, d1); acc['d0d1'] += xc(d0, d1); acc['d1d0'] += xc(d1, d0)
n0, n1 = acc['d0d0'][0], acc['d1d1'][0]
acc = {'d0d0': acc['d0d0'] / n0, 'd1d1': acc['d1d1'] / n1, 'd0d1': acc['d0d1'] / np.sqrt(n0 * n1), 'd1d0': acc['d1d0'] / np.sqrt(n0 * n1)}
np.savez('research/campaign_2026-10-03/eda/statphys/data/dacf.npz', **acc)
for k in acc:
    a = acc[k]
    o = np.argsort(a[1:])[:6] + 1
    print(k, 'lag0', round(a[0], 3), 'most negative lags', o, a[o].round(4))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(14, 8))
for k in acc: ax[0].plot(np.arange(1, L + 1), acc[k][1:], lw=.7, label=k)
ax[0].set_xscale('log'); ax[0].legend(); ax[0].axhline(0, c='k', lw=.5); ax[0].set_title('xcorr of target increments (in ticks) vs lag, 40 seqs')
for k in acc: ax[1].plot(np.arange(1, 400), acc[k][1:400], lw=.7, label=k)
ax[1].axhline(0, c='k', lw=.5); ax[1].legend()
fig.tight_layout(); fig.savefig('research/campaign_2026-10-03/eda/statphys/f06_dacf.png', dpi=70)
print(sorted(set(np.round(units, 6))))
