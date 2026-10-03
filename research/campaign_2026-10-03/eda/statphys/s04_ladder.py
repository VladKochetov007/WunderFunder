import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
pf = pq.ParquetFile('datasets/train.parquet')
for rg in [6698, 4178]:
    t = pf.read_row_group(rg).to_pandas()
    B = t[[f'i0_p{j}' for j in range(11)]].values; A = t[[f'i0_p{j}' for j in range(11, 22)]].values
    vb = np.unique(B); va = np.unique(A)
    print('rg', rg, 'bid distinct', len(vb), 'ask distinct', len(va))
    print(' bid vals', vb[:14].round(4), vb[-5:].round(4))
    print(' ask vals', va[:14].round(4), va[-5:].round(4))
    # per-row: rank of each column value in the sorted distinct list
    rb = np.searchsorted(vb, B); ra = np.searchsorted(va, A)
    print(' rows 500..506 bid ranks\n', rb[500:507]); print(' ask ranks\n', ra[500:507])
    print(' t0,t1', t.t0.values[500:507], t.t1.values[500:507])
    # column mean values (identifier ordering)
    print(' bid col mean', B.mean(0).round(3)); print(' ask col mean', A.mean(0).round(3))
    print(' dp distinct', [len(np.unique(t[f"i0_dp{j}"])) for j in range(4)], 'dv', [len(np.unique(t[f"i0_dv{j}"])) for j in range(4)])
    # are dp values in bid/ask value sets?
    dp = t['i0_dp0'].values
    print(' frac dp0 in bid set', np.isin(dp, vb).mean(), 'in ask set', np.isin(dp, va).mean())
