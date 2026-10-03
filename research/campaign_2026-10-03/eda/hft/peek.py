import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_columns',60)
pf=pq.ParquetFile('datasets/valid.parquet')
t=pf.read_row_group(0).to_pandas()
print(t.shape, t.seq_ix.iloc[0])
cols=[c for c in t.columns]
i0p=[f'i0_p{i}' for i in range(22)]; i0v=[f'i0_v{i}' for i in range(22)]
print(t[i0p].iloc[:5].T)
print(t[i0v].iloc[:5].T)
print(t[[f'i0_dp{i}' for i in range(4)]+[f'i0_dv{i}' for i in range(4)]].iloc[:12])
print(t[[f'a{i}' for i in range(8)]+['t0','t1','is_scored']].iloc[:12])
print(t[[f'a{i}' for i in range(8)]+['t0','t1']].describe().T)
print(t[i0p+i0v].describe().T)
