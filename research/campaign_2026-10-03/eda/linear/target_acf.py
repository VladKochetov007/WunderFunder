# ACF of target first differences: if t(t)=P(t+h)-P(t) then corr(dt(t), dt(t+h)) has a negative spike at lag h.
import pyarrow.parquet as pq, numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=220)
f = pq.ParquetFile('datasets/valid.parquet')
L = 1200
acc = {k: np.zeros(L + 1) for k in ['d0d0', 'd1d1', 'd0d1', 'd1d0', 'y0y0', 'y1y1', 'y0y1']}
nseq = 60
rng = np.random.default_rng(0)
rgs = rng.choice(f.metadata.num_row_groups, nseq, replace=False)
def xcorr(a, b, L):
    # corr(a(t), b(t+k)) for k=0..L via FFT
    a = a - a.mean(); b = b - b.mean()
    n = len(a); nf = 1 << (2 * n - 1).bit_length()
    A = np.fft.rfft(a, nf); B = np.fft.rfft(b, nf)
    c = np.fft.irfft(np.conj(A) * B, nf)[:L + 1]
    return c / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12)
for rg in rgs:
    tb = f.read_row_group(int(rg), columns=['t0', 't1'])
    y0 = tb.column('t0').to_numpy().astype(np.float64); y1 = tb.column('t1').to_numpy().astype(np.float64)
    d0 = np.diff(y0); d1 = np.diff(y1)
    acc['d0d0'] += xcorr(d0, d0, L); acc['d1d1'] += xcorr(d1, d1, L)
    acc['d0d1'] += xcorr(d0, d1, L); acc['d1d0'] += xcorr(d1, d0, L)
    acc['y0y0'] += xcorr(y0, y0, L); acc['y1y1'] += xcorr(y1, y1, L); acc['y0y1'] += xcorr(y0, y1, L)
for k in acc: acc[k] /= nseq
np.savez('research/campaign_2026-10-03/eda/linear/target_acf.npz', **acc)
for k in ['d0d0', 'd1d1', 'd0d1', 'd1d0']:
    a = acc[k]
    idx = np.argsort(np.abs(a[1:]))[::-1][:12] + 1
    print(k, 'lag0', a[0], 'top lags', [(int(i), round(float(a[i]), 4)) for i in idx])
print('y0y0', acc['y0y0'][[1, 10, 50, 100, 150, 200, 300, 500, 1000]])
print('y1y1', acc['y1y1'][[1, 10, 50, 100, 150, 200, 300, 500, 1000]])
print('y0y1', acc['y0y1'][[0, 1, 10, 50, 100, 150, 200, 300, 500, 1000]])
