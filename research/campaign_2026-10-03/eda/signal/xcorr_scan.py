"""Lead-lag cross-correlation of targets with increments of every input column (pooled over sequences)."""
import numpy as np, sys
import sio

cols = [c for c in sio.columns() if c not in ('seq_ix', 'step_in_seq', 'need_prediction')]
inp = [c for c in cols if c not in ('t0', 't1')]
L = 400
lags = np.arange(-L, L + 1)
rng = np.random.default_rng(1)
idx = np.sort(rng.choice(sio.n_seq(), int(sys.argv[1]) if len(sys.argv) > 1 else 20, replace=False))
acc = {t: np.zeros((len(inp), len(lags))) for t in ('t0', 't1')}
acc_lvl = {t: np.zeros((len(inp), len(lags))) for t in ('t0', 't1')}
for rg in idx:
    d = sio.load(rg, cols)
    X = np.stack([d[c] for c in inp], 1).astype(np.float64)
    dX = np.diff(X, axis=0, prepend=X[:1])
    dX = (dX - dX.mean(0)) / (dX.std(0) + 1e-12)
    Xl = (X - X.mean(0)) / (X.std(0) + 1e-12)
    n = len(X)
    for t in ('t0', 't1'):
        y = d[t].astype(np.float64)
        y = (y - y.mean()) / y.std()
        # corr(y(t), dx(t+k)) via FFT
        m = 1 << int(np.ceil(np.log2(2 * n)))
        Fy = np.fft.rfft(y, m)
        for M, a in ((dX, acc), (Xl, acc_lvl)):
            F = np.fft.rfft(M, m, axis=0)
            cc = np.fft.irfft(np.conj(Fy)[:, None] * F, m, axis=0) / n  # cc[k] = sum y(t) x(t+k)
            a[t] += np.concatenate([cc[-L:], cc[:L + 1]], 0).T
for t in acc:
    acc[t] /= len(idx); acc_lvl[t] /= len(idx)
np.savez('xcorr_scan.npz', lags=lags, inp=np.array(inp), **{f'd_{t}': acc[t] for t in acc}, **{f'l_{t}': acc_lvl[t] for t in acc})
for t in ('t0', 't1'):
    a = acc[t]
    fut = a[:, L + 1:].sum(1)  # sum of corr over future increments
    past = a[:, :L + 1].sum(1)
    o = np.argsort(-np.abs(fut))
    print(t, 'top by sum future-increment corr  [name fut past maxlag]')
    for i in o[:20]:
        print(f'  {inp[i]:8s} fut={fut[i]:+.3f} past={past[i]:+.3f} argmax_lag={lags[np.argmax(np.abs(a[i]))]:+d} peak={a[i][np.argmax(np.abs(a[i]))]:+.4f}')
    b = acc_lvl[t][:, L]
    o = np.argsort(-np.abs(b))
    print(t, 'top contemporaneous level corr')
    for i in o[:12]:
        print(f'  {inp[i]:8s} {b[i]:+.3f}')
