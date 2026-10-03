import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option('display.width',250)
pf=pq.ParquetFile('datasets/valid.parquet')
P=[f'i0_p{i}' for i in range(22)]; V=[f'i0_v{i}' for i in range(22)]
for g in [0,2]:
    t=pf.read_row_group(g).to_pandas()
    X=t[P].values
    ch=(np.diff(X,axis=0)!=0)
    print('seq',g,'changes per price col', ch.sum(0))
    # co-change matrix
    print('frac rows where any price changes', ch.any(1).mean(), 'all change', ch.all(1).mean())
    nch=ch.sum(1); print(pd.Series(nch).value_counts().sort_index().to_dict())
    Xv=t[V].values; chv=(np.diff(Xv,axis=0)!=0); print('vol changes per col', chv.sum(0))
    D=[f'i0_dp{i}' for i in range(4)]+[f'i0_dv{i}' for i in range(4)]
    print('trade col changes', (np.diff(t[D].values,axis=0)!=0).sum(0))
    A=[f'a{i}' for i in range(8)]
    print('a col changes', (np.diff(t[A].values,axis=0)!=0).sum(0))
