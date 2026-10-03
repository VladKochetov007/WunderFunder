import pyarrow.parquet as pq, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pf = pq.ParquetFile('datasets/train.parquet')
names = pf.schema_arrow.names; feat = names[3:115]
OUT = 'research/campaign_2026-10-03/eda/statphys/'
for rg in [4178, 6698]:
    t = pf.read_row_group(rg).to_pandas()
    sl = slice(0, 3000)
    fig, ax = plt.subplots(6, 1, figsize=(16, 20), sharex=True)
    for j in range(11): ax[0].plot(t[f'i0_p{j}'].values[sl], lw=.6, label=f'p{j}')
    ax[0].legend(ncol=11, fontsize=7); ax[0].set_title(f'rg {rg} i0 bid-index price-like')
    for j in range(11, 22): ax[1].plot(t[f'i0_p{j}'].values[sl], lw=.6, label=f'p{j}')
    ax[1].legend(ncol=11, fontsize=7); ax[1].set_title('i0 ask-index price-like')
    for j in range(4): ax[2].plot(t[f'i0_dp{j}'].values[sl], lw=.6, label=f'dp{j}')
    ax[2].legend(); ax[2].set_title('i0 trade price')
    for j in range(8): ax[3].plot(t[f'a{j}'].values[sl], lw=.6, label=f'a{j}')
    ax[3].legend(ncol=8); ax[3].set_title('a*')
    ax[4].plot(t.t0.values[sl], lw=.8, label='t0'); ax[4].plot(t.t1.values[sl], lw=.8, label='t1'); ax[4].legend(); ax[4].set_title('targets')
    for j in range(4): ax[5].plot(t[f'i0_dv{j}'].values[sl], lw=.6, label=f'dv{j}')
    ax[5].legend(); ax[5].set_title('i0 trade vol')
    fig.tight_layout(); fig.savefig(OUT + f'f03_look_rg{rg}.png', dpi=70); plt.close(fig)
    # distinct-values per column
    print(rg, {c: len(np.unique(t[c].values)) for c in feat[:22] + ['a0','a5','a6','a7','i0_dp0','i0_dv0']})
