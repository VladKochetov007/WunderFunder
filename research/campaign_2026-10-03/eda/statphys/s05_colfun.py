import pyarrow.parquet as pq, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pf = pq.ParquetFile('datasets/train.parquet')
OUT = 'research/campaign_2026-10-03/eda/statphys/'
t = pf.read_row_group(4178).to_pandas()
# functional dependence test: for each pair (p0, pj) number of distinct pj per distinct p0
for pre, rngj in [('i0_p', range(22)), ('i1_p', range(22))]:
    base = t[pre + '0'].values
    res = []
    for j in rngj:
        g = pd.DataFrame({'a': base, 'b': t[f'{pre}{j}'].values}).groupby('a').b.nunique()
        res.append(round(g.mean(), 2))
    print(pre, 'mean #distinct pj per p0 value', res)
fig, ax = plt.subplots(3, 8, figsize=(24, 9))
cols = [f'i0_p{j}' for j in range(1, 22)] + ['i0_dp0', 'i1_p0', 'a0']
for a, c in zip(ax.flat, cols):
    a.plot(t['i0_p0'].values, t[c].values, ',', alpha=.5); a.set_title(f'p0 vs {c}', fontsize=8)
fig.tight_layout(); fig.savefig(OUT + 'f05_colfun_rg4178.png', dpi=60)
