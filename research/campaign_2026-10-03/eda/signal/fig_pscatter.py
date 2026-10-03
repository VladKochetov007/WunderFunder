import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sio
d = sio.load(0)
P = np.stack([d[f'i0_p{k}'] for k in range(22)], 1)
ev = np.r_[True, np.any(np.diff(P, axis=0) != 0, 1)]
Pe = P[ev]
fig, ax = plt.subplots(4, 6, figsize=(24, 16))
pairs = [(0, 3), (0, 1), (0, 10), (0, 5), (0, 4), (0, 11), (11, 21), (11, 16), (11, 13), (11, 17), (1, 2), (7, 8),
         (0, 14), (10, 19), (5, 17), (4, 13), (6, 9), (12, 18), (0, 21), (3, 14), (8, 15), (2, 20), (1, 12), (9, 16)]
for a, (i, j) in zip(ax.ravel(), pairs):
    a.scatter(Pe[:, i], Pe[:, j], s=1, alpha=.3); a.set_xlabel(f'p{i}'); a.set_ylabel(f'p{j}')
fig.tight_layout(); fig.savefig('fig_pscatter_rg0.png', dpi=60)
# event-resolution time series zoom
s = 3000
fig, ax = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
for k in range(11): ax[0].step(np.arange(200), Pe[s:s + 200, k], lw=.8, label=f'p{k}')
for k in range(11, 22): ax[1].step(np.arange(200), Pe[s:s + 200, k], lw=.8, label=f'p{k}')
ax[0].legend(ncol=11, fontsize=7); ax[1].legend(ncol=11, fontsize=7)
ax[2].step(np.arange(200), d['t0'][ev][s:s + 200], label='t0'); ax[2].step(np.arange(200), d['t1'][ev][s:s + 200], label='t1'); ax[2].legend(); ax[2].grid()
fig.tight_layout(); fig.savefig('fig_pevents_rg0.png', dpi=60)
