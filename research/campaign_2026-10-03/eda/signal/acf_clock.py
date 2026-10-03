"""Target ACF in row time vs i0-price-event time; the triangle width gives the horizon in that clock."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sio

def acf(x, L):
    x = x - x.mean(); n = len(x)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    F = np.fft.rfft(x, m); r = np.fft.irfft(F * np.conj(F), m)[:L + 1]
    return r / r[0]

rng = np.random.default_rng(4)
idx = np.sort(rng.choice(sio.n_seq(), 40, replace=False))
pc = [f'i0_p{k}' for k in range(22)]
L_row, L_ev = 600, 200
A = {k: [] for k in ('row0', 'row1', 'ev0', 'ev1', 'ev01')}
for rg in idx:
    d = sio.load(rg, pc + ['t0', 't1'])
    P = np.stack([d[c] for c in pc], 1)
    ev = np.r_[True, np.any(np.diff(P, axis=0) != 0, 1)]
    ev[:99] = False
    y0, y1 = d['t0'][99:].astype(float), d['t1'][99:].astype(float)
    if y0.std() == 0: continue
    A['row0'].append(acf(y0, L_row)); A['row1'].append(acf(y1, L_row))
    e0, e1 = d['t0'][ev].astype(float), d['t1'][ev].astype(float)
    if len(e0) < 3 * L_ev: continue
    A['ev0'].append(acf(e0, L_ev)); A['ev1'].append(acf(e1, L_ev))
    a, b = e0 - e0.mean(), e1 - e1.mean()
    A['ev01'].append(np.array([np.mean(a[:len(a) - k] * b[k:]) for k in range(L_ev + 1)]) / (a.std() * b.std()))
M = {k: np.mean(v, 0) for k, v in A.items()}
np.savez('acf_clock.npz', **M)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(M['row0'], label='t0'); ax[0].plot(M['row1'], label='t1'); ax[0].set_title('target ACF, row lags'); ax[0].legend(); ax[0].grid()
ax[1].plot(M['ev0'], label='t0'); ax[1].plot(M['ev1'], label='t1'); ax[1].plot(M['ev01'], label='xcorr t0(n),t1(n+k)'); ax[1].set_title('target ACF, i0-price-event lags'); ax[1].legend(); ax[1].grid()
fig.tight_layout(); fig.savefig('fig_acf_clock.png', dpi=80)
for k in ('ev0', 'ev1'):
    m = M[k]; print(k, ' '.join(f'{i}:{m[i]:.3f}' for i in [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 200]))
for k in ('row0', 'row1'):
    m = M[k]; print(k, ' '.join(f'{i}:{m[i]:.3f}' for i in [0, 1, 5, 10, 50, 100, 200, 300, 400, 500, 600]))
print('ev01', ' '.join(f'{i}:{M["ev01"][i]:.3f}' for i in [0, 1, 5, 10, 20, 40, 60, 100]))
