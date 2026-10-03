"""Event-time spectra: targets, target increments, input increments; inter-event interval statistics."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import sio, feat

rng = np.random.default_rng(8)
idx = rng.choice(sio.n_seq(), 40, replace=False)
nper = 512
acc = {}
gaps = []
def add(k, f, p):
    acc.setdefault(k, [f, 0, 0]); acc[k][1] = acc[k][1] + p; acc[k][2] += 1
for rg in idx:
    d = sio.load(int(rg))
    e0p, e0, e1 = feat.clocks(d)
    ev = np.where(e0p[99:])[0] + 99
    if len(ev) < 2 * nper: continue
    gaps.append(np.diff(ev))
    for t in ('t0', 't1'):
        y = d[t][ev].astype(float)
        f, p = signal.welch(y - y.mean(), nperseg=nper); add(f'{t}_ev', f, p / y.var())
        f, p = signal.welch(np.diff(y), nperseg=nper); add(f'd{t}_ev', f, p / np.diff(y).var())
        yr = d[t][99:].astype(float)
        f, p = signal.welch(yr - yr.mean(), nperseg=4096); add(f'{t}_row', f, p / yr.var())
    for c in ('i0_p0', 'i0_p11', 'i0_v0', 'i0_dp0', 'a1', 'a0'):
        x = d[c][ev].astype(float); dx = np.diff(x)
        if dx.std() == 0: continue
        f, p = signal.welch(dx, nperseg=nper); add(f'd{c}_ev', f, p / dx.var())
        # coherence target vs input level (two-sided relation; upper bound on linear coupling)
        for t in ('t0',):
            f, C = signal.coherence(d[t][ev].astype(float), x, nperseg=nper); add(f'coh_{t}_{c}', f, C)
            f, C = signal.coherence(np.diff(d[t][ev].astype(float)), dx, nperseg=nper); add(f'cohd_{t}_{c}', f, C)
G = np.concatenate(gaps)
print('inter-i0-price-event gap (rows): mean %.2f median %d p90 %d p99 %d max %d; CV %.2f' % (G.mean(), np.median(G), np.percentile(G, 90), np.percentile(G, 99), G.max(), G.std() / G.mean()))
u, c = np.unique(G, return_counts=True); print('gap pmf', ' '.join(f'{a}:{b / len(G):.3f}' for a, b in zip(u[:12], c[:12])))
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
for k in ('t0_ev', 't1_ev', 'dt0_ev', 'dt1_ev'):
    f, p, n = acc[k]; ax[0, 0].loglog(f[1:], p[1:] / n, label=k)
ax[0, 0].set_xlabel('cycles / i0-price-event'); ax[0, 0].legend(); ax[0, 0].set_title('target PSD (event clock)'); ax[0, 0].grid(which='both', alpha=.3)
for k in ('t0_row', 't1_row'):
    f, p, n = acc[k]; ax[0, 1].loglog(f[1:], p[1:] / n, label=k)
ax[0, 1].set_xlabel('cycles / row'); ax[0, 1].legend(); ax[0, 1].set_title('target PSD (row clock)'); ax[0, 1].grid(which='both', alpha=.3)
for k in [k for k in acc if k.startswith('di')] + ['da1_ev', 'da0_ev']:
    if k in acc:
        f, p, n = acc[k]; ax[0, 2].semilogy(f, p / n, label=k, lw=.8)
ax[0, 2].legend(); ax[0, 2].set_title('input increment PSD (event clock)'); ax[0, 2].set_xlabel('cycles / event')
for k in [k for k in acc if k.startswith('coh_')]:
    f, p, n = acc[k]; ax[1, 0].semilogx(f[1:], p[1:] / n, label=k)
ax[1, 0].legend(); ax[1, 0].set_title('coherence |C|^2: t0 level vs input level (event clock)')
for k in [k for k in acc if k.startswith('cohd_')]:
    f, p, n = acc[k]; ax[1, 1].plot(f, p / n, label=k)
ax[1, 1].legend(); ax[1, 1].set_title('coherence: dt0 vs d input (event clock)')
ax[1, 2].hist(np.clip(G, 0, 40), bins=40, log=True); ax[1, 2].set_title('rows between i0 price events')
fig.tight_layout(); fig.savefig('fig_spectra_event.png', dpi=70)
np.savez('spectra.npz', **{k: np.c_[v[0], v[1] / v[2]] for k, v in acc.items()})
# periodic ripple: peak in dt0 PSD
f, p, n = acc['dt0_ev']; p = p / n
o = np.argsort(-p)[:5]; print('dt0 PSD top freqs (cyc/event):', [(round(f[i], 4), round(1 / f[i], 2) if f[i] > 0 else None, round(p[i], 3)) for i in o])
f, p, n = acc['t0_ev']; p = p / n
print('t0 event PSD slope low-f:', np.polyfit(np.log(f[2:30]), np.log(p[2:30]), 1)[0], ' high-f:', np.polyfit(np.log(f[60:250]), np.log(p[60:250]), 1)[0])
