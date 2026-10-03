import sys
sys.path.insert(0, '/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/puzzle_b/scripts')
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rg = int(sys.argv[1]); lo = int(sys.argv[2]); hi = int(sys.argv[3]); tag = sys.argv[4]
d = seq(rg)
x = np.arange(lo, hi); s = slice(lo, hi)
fig, ax = plt.subplots(7, 1, figsize=(16, 18), sharex=True)
for k in [0, 3, 6, 8, 9]: ax[0].step(x, d[f'i0_p{k}'].values[s], lw=.8, label=f'i0_p{k}', where='post')
ax[0].legend(fontsize=7, ncol=5); ax[0].set_title(f'rg {rg} seq {d.seq_ix.iloc[0]}  i0 price block A (selected)')
for k in [11, 14, 15, 16, 19]: ax[1].step(x, d[f'i0_p{k}'].values[s], lw=.8, label=f'i0_p{k}', where='post')
ax[1].legend(fontsize=7, ncol=5); ax[1].set_title('i0 price block B')
for k in [0, 3, 11, 16]: ax[2].step(x, d[f'i1_p{k}'].values[s], lw=.8, label=f'i1_p{k}', where='post')
ax[2].legend(fontsize=7); ax[2].set_title('i1 prices')
for k in range(4): ax[3].step(x, d[f'i0_dp{k}'].values[s], lw=.8, label=f'i0_dp{k}', where='post')
ax[3].legend(fontsize=7); ax[3].set_title('i0 dp')
for k in [0, 1, 11, 12]: ax[4].plot(x, d[f'i0_v{k}'].values[s], lw=.6, label=f'i0_v{k}')
ax[4].legend(fontsize=7); ax[4].set_title('i0 volumes')
for k in range(5): ax[5].plot(x, d[f'a{k}'].values[s], lw=.5, label=f'a{k}')
for k in range(5, 8): ax[5].step(x, d[f'a{k}'].values[s], lw=.8, label=f'a{k}', where='post')
ax[5].legend(fontsize=7, ncol=8); ax[5].set_title('a columns'); ax[5].set_ylim(-6, 4)
ax[6].step(x, d.t0.values[s], lw=1, label='t0', where='post'); ax[6].step(x, d.t1.values[s], lw=1, label='t1', where='post')
ax[6].legend(); ax[6].set_title('targets')
plt.tight_layout(); plt.savefig(f'{OUT}/{tag}.png', dpi=70)
