import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sio, sys
rg = int(sys.argv[1]) if len(sys.argv) > 1 else 0
a, b = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 20000)
d = sio.load(rg)
x = np.arange(20000)[a:b]
fig, ax = plt.subplots(7, 1, figsize=(16, 20), sharex=True)
for k in range(11):
    ax[0].plot(x, d[f'i0_p{k}'][a:b], lw=.6, label=f'p{k}')
for k in range(11, 22):
    ax[1].plot(x, d[f'i0_p{k}'][a:b], lw=.6, label=f'p{k}')
ax[0].legend(ncol=11, fontsize=6); ax[1].legend(ncol=11, fontsize=6)
ax[0].set_ylabel('i0 p0..p10'); ax[1].set_ylabel('i0 p11..p21')
for k in range(11):
    ax[2].plot(x, d[f'i1_p{k}'][a:b], lw=.6)
ax[2].set_ylabel('i1 p0..p10')
for k in range(4):
    ax[3].plot(x, d[f'i0_dp{k}'][a:b], lw=.6, label=f'dp{k}')
ax[3].legend(fontsize=6); ax[3].set_ylabel('i0 dp')
for k in range(8):
    ax[4].plot(x, d[f'a{k}'][a:b] + 0 * k, lw=.4, label=f'a{k}')
ax[4].legend(ncol=8, fontsize=6); ax[4].set_ylabel('a'); ax[4].set_ylim(-6, 4)
ax[5].plot(x, d['t0'][a:b], lw=.7, label='t0'); ax[5].plot(x, d['t1'][a:b], lw=.7, label='t1'); ax[5].legend()
ax[6].plot(x, np.cumsum(d['t0'])[a:b] / 100, lw=.7, label='cumsum t0/100'); ax[6].plot(x, np.cumsum(d['t1'])[a:b] / 100, lw=.7, label='cumsum t1/100'); ax[6].legend()
fig.suptitle(f'rg {rg} seq {d["seq_ix"][0]}')
fig.tight_layout()
fig.savefig(f'fig_overview_rg{rg}_{a}_{b}.png', dpi=70)
