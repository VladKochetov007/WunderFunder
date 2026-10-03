import pyarrow.parquet as pq, numpy as np, pandas as pd
pf=pq.ParquetFile('datasets/valid.parquet')
P=[f'i0_p{i}' for i in range(22)]
for g in [74,111,0]:
    t=pf.read_row_group(g).to_pandas()
    X=t[P].values
    y0=t.t0.values; y1=t.t1.values
    v=np.unique(np.abs(np.r_[np.diff(y0),np.diff(y1)])); q=v[v>1e-6].min()
    d0=np.r_[0,np.diff(y0)]/q; d1=np.r_[0,np.diff(y1)]/q
    cb=np.r_[False,(np.diff(X[:,:11],axis=0)!=0).any(1)]; ca=np.r_[False,(np.diff(X[:,11:],axis=0)!=0).any(1)]
    for name,m in [('bid only',cb&~ca),('ask only',ca&~cb),('both',ca&cb),('none',~ca&~cb)]:
        vc=pd.Series(list(zip(np.round(d0[m]).astype(int),np.round(d1[m]).astype(int)))).value_counts()
        print(g,name,m.sum(),'top (d0,d1):',vc.head(10).to_dict())
