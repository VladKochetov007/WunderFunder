import pyarrow.parquet as pq, numpy as np, pandas as pd
np.set_printoptions(linewidth=200, precision=3, suppress=True)
pf=pq.ParquetFile('datasets/valid.parquet')
P=[f'i0_p{i}' for i in range(22)]
for g in [0,2]:
    t=pf.read_row_group(g).to_pandas()
    for side,cols in [('bid',P[:11]),('ask',P[11:])]:
        X=t[cols].values.astype(np.float64)
        D=np.diff(X,axis=0); m=(D!=0).any(1); D=D[m]
        u,s,vt=np.linalg.svd(D,full_matrices=False)
        print(g,side,'sv',s/s[0])
        # level matrix SVD (centered)
        Xc=X-X.mean(0)
        u2,s2,vt2=np.linalg.svd(Xc,full_matrices=False)
        print('   level sv',s2/s2[0])
        # distinct values per column
        print('   nunique per col',[len(np.unique(X[:,j])) for j in range(11)])
