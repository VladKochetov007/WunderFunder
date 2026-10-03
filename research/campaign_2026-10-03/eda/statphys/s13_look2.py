import sys; sys.path.insert(0, 'research/campaign_2026-10-03/eda/statphys')
from common import *
import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pf = pq.ParquetFile(TRAIN); fn = feat_names(pf)
rg = int(sys.argv[1]); a, b = int(sys.argv[2]), int(sys.argv[3])
X, Y, u, _ = load(pf, rg)
T = np.round(Y / u).astype(int)
st = pd.factorize(pd.util.hash_pandas_object(pd.DataFrame(X[:, :22]), index=False).values)[0]
st1 = pd.factorize(pd.util.hash_pandas_object(pd.DataFrame(X[:, 52:74]), index=False).values)[0]
sl = slice(a, b); x = np.arange(a, b)
fig, ax = plt.subplots(6, 1, figsize=(16, 16), sharex=True)
ax[0].step(x, T[sl, 0], lw=.8, label='T0'); ax[0].step(x, T[sl, 1], lw=.8, label='T1'); ax[0].step(x, T[sl].sum(1), lw=.8, label='T0+T1'); ax[0].legend(); ax[0].set_title(f'rg {rg} unit {u:.4f}: targets in ticks')
ax[1].step(x, st[sl], lw=.6, label='i0 price-state id'); ax[1].step(x, st1[sl] % 50, lw=.6, label='i1 price-state id %50'); ax[1].legend()
for j in range(4): ax[2].plot(x, X[sl, 44 + j], lw=.6, label=fn[44 + j])
ax[2].legend(); 
for j in range(4): ax[3].plot(x, X[sl, 48 + j], lw=.6, label=fn[48 + j])
ax[3].legend()
for j in range(5): ax[4].plot(x, X[sl, 104 + j], lw=.6, label=fn[104 + j])
ax[4].legend()
for j in range(5, 8): ax[5].plot(x, X[sl, 104 + j], lw=.6, label=fn[104 + j])
ax[5].legend()
fig.tight_layout(); fig.savefig(OUT + f'f13_look_rg{rg}_{a}_{b}.png', dpi=65)
