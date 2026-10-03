import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option('display.width',250)
pf=pq.ParquetFile('datasets/valid.parquet')
P=[f'i0_p{i}' for i in range(22)]
for g in [0,2]:
    t=pf.read_row_group(g).to_pandas()
    for side,cols in [('bid',P[:11]),('ask',P[11:])]:
        X=t[cols].values.astype(np.float64)
        D=np.diff(X,axis=0); m=(D!=0).any(1)
        S=np.sign(D[m])
        # patterns
        pat=pd.Series([''.join('+' if v>0 else '-' for v in r) for r in S]).value_counts()
        print(g,side,'n',m.sum(),'top patterns'); print(pat.head(8).to_string())
