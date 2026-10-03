import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sio, feat
z = np.load('catalogue.npz'); A = z['train']; q = np.round(A[:, 2], 4)
rgs = [int(A[q == 0.6569, 0][3]), int(A[q == 0.1254, 0][3])]
fig, ax = plt.subplots(2, 3, figsize=(22, 10), gridspec_kw=dict(width_ratios=[2, 2, 1.2]))
for r, rg in enumerate(rgs):
    d = sio.load(rg)
    P = np.stack([d[f'i0_p{k}'] for k in range(22)], 1)
    e0p, _, _ = feat.clocks(d)
    ev = np.where(e0p)[0]
    _, sid = np.unique(P[ev], axis=0, return_inverse=True); sid = sid.ravel()
    Q = np.diff(np.unique(np.r_[d['t0'], d['t1']])).min()
    s = 300 if r == 0 else 1000
    sl = slice(s, s + 150)
    ax[r, 0].step(np.arange(150), sid[sl], where='post', lw=.8); ax[r, 0].set_ylabel('i0 price-state id')
    ax[r, 0].set_title(f'rg {rg} (quantum {Q:.4f}): price state per i0 event'); ax[r, 0].set_xlabel('event index')
    ax[r, 1].step(np.arange(150), np.round(d['t0'][ev][sl] / Q), where='post', label='t0 ticks')
    ax[r, 1].step(np.arange(150), np.round(d['t1'][ev][sl] / Q), where='post', label='t1 ticks')
    ax2 = ax[r, 1].twinx(); ax2.bar(np.arange(150), np.diff(ev)[sl], alpha=.25, color='k'); ax2.set_ylabel('rows to next event')
    ax[r, 1].legend(); ax[r, 1].set_title('targets (ticks) on the same events')
    # recurrence plot of states over 400 events
    W = 400; ss = sid[s:s + W]
    ax[r, 2].imshow(ss[:, None] == ss[None, :], cmap='Greys', interpolation='nearest'); ax[r, 2].set_title('recurrence plot (state_i == state_j)')
fig.tight_layout(); fig.savefig('fig_cycle_example.png', dpi=65)
