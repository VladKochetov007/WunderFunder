"""Horizon from plateaus: a future jump J of the underlying enters the window (target jumps +J at row r1) and
leaves it when it happens (target jumps -J at r2). Distribution of r2-r1 in rows and in i0-price events."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sio, feat
z = np.load('catalogue.npz'); A = z['train']; q = np.round(A[:, 2], 4)
rng = np.random.default_rng(15)
out = {}
for qq in (0.1254, 0.2413):
    rgs = rng.choice(A[q == qq, 0].astype(int), 40, replace=False)
    DR, DE = [], []
    for rg in rgs:
        d = sio.load(int(rg), [f'i0_p{k}' for k in range(22)] + ['t0', 't1'])
        e0p, _, _ = feat.clocks(d) if False else (np.r_[True, np.any(np.diff(np.stack([d[f'i0_p{k}'] for k in range(22)], 1), axis=0) != 0, 1)], None, None)
        evcount = np.cumsum(e0p)
        for t in ('t0', 't1'):
            k = np.round(d[t] / qq).astype(int)
            dk = np.diff(k); r = np.where(dk != 0)[0] + 1
            big = r[np.abs(dk[r - 1]) >= (6 if qq < 0.2 else 4)]
            for r1 in big:
                J = dk[r1 - 1]
                later = r[(r > r1)]
                cand = later[(np.sign(dk[later - 1]) == -np.sign(J)) & (np.abs(dk[later - 1] + J) <= max(1, abs(J) // 4))]
                if len(cand) == 0: continue
                r2 = cand[0]
                DR.append(r2 - r1); DE.append(evcount[r2] - evcount[r1])
    out[qq] = (np.array(DR), np.array(DE))
    DR, DE = out[qq]
    print(f'q={qq}: n pairs {len(DR)}; rows: median {np.median(DR):.0f} IQR {np.percentile(DR, 25):.0f}-{np.percentile(DR, 75):.0f} CV {DR.std() / DR.mean():.2f}; events: median {np.median(DE):.0f} IQR {np.percentile(DE, 25):.0f}-{np.percentile(DE, 75):.0f} CV {DE.std() / DE.mean():.2f}')
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for qq, (DR, DE) in out.items():
    ax[0].hist(np.clip(DR, 0, 2000), bins=100, histtype='step', label=f'q={qq}', density=True)
    ax[1].hist(np.clip(DE, 0, 500), bins=100, histtype='step', label=f'q={qq}', density=True)
ax[0].set_title('rows between matched +J / -J target jumps'); ax[1].set_title('i0-price events between matched jumps')
ax[0].legend(); ax[1].legend(); fig.tight_layout(); fig.savefig('fig_plateau.png', dpi=70)
