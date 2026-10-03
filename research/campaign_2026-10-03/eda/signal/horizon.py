"""Find the target horizon: xcorr between target jumps dt(n) and the observable 'price-change' proxies at n+j
(event clock and row clock). If t(n) = X(n+K) - X(n) then dt(n) = dX(n+K) - dX(n): negative spike at 0, positive at K."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sio, feat
rng = np.random.default_rng(12)
idx = rng.choice(sio.n_seq(), 60, replace=False)
proxies = ['i0_dp1', 'i0_dp2', 'i0_p8', 'i0_p20', 'i0_p11', 'i0_p5']
LE, LR = 300, 1500
accE = {p: np.zeros(2 * LE + 1) for p in proxies}; accR = {p: np.zeros(2 * LR + 1) for p in proxies}
nE = nR = 0
def xc(y, x, L):
    y = (y - y.mean()) / (y.std() + 1e-12); x = (x - x.mean()) / (x.std() + 1e-12); n = len(y)
    m = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.conj(np.fft.rfft(y, m)) * np.fft.rfft(x, m), m) / n
    return np.r_[cc[-L:], cc[:L + 1]]
for rg in idx:
    d = sio.load(int(rg))
    e0p, _, _ = feat.clocks(d)
    ev = np.where(e0p)[0]; ev = ev[ev >= 99]
    if len(ev) < 3 * LE: continue
    dy = np.diff(d['t0'][ev].astype(float))
    for p in proxies:
        accE[p] += xc(dy, d[p][ev][1:].astype(float), LE)
    nE += 1
    # row clock: dt(row) vs proxy-at-event-rows (zero elsewhere): use increments of proxy
    dyr = np.diff(d['t0'][99:].astype(float))
    for p in proxies:
        x = d[p][99:].astype(float); dx = np.diff(x)
        accR[p] += xc(dyr, dx, LR)
    nR += 1
fig, ax = plt.subplots(2, 1, figsize=(16, 9))
for p in proxies:
    ax[0].plot(np.arange(-LE, LE + 1), accE[p] / nE, lw=.8, label=p)
    ax[1].plot(np.arange(-LR, LR + 1), accR[p] / nR, lw=.8, label=p)
ax[0].set_title('xcorr dt0(n) vs proxy level at event n+j (event clock)'); ax[0].legend(); ax[0].set_xlabel('j events'); ax[0].grid(alpha=.3)
ax[1].set_title('xcorr dt0(t) vs d proxy at row t+j (row clock)'); ax[1].legend(); ax[1].set_xlabel('j rows'); ax[1].grid(alpha=.3)
ax[0].set_ylim(-0.1, 0.1); ax[1].set_ylim(-0.05, 0.05)
fig.tight_layout(); fig.savefig('fig_horizon_xcorr.png', dpi=70)
for p in proxies:
    a = accE[p] / nE; j = np.arange(-LE, LE + 1)
    pos = j > 3
    print(p, 'event: c(0)=%.3f' % a[LE], ' max|c| for j>3 at', j[pos][np.argmax(np.abs(a[pos]))], '%.4f' % a[pos][np.argmax(np.abs(a[pos]))])
    a = accR[p] / nR; j = np.arange(-LR, LR + 1); pos = j > 3
    print('    row: c(0)=%.3f' % a[LR], ' max|c| for j>3 at', j[pos][np.argmax(np.abs(a[pos]))], '%.4f' % a[pos][np.argmax(np.abs(a[pos]))])
