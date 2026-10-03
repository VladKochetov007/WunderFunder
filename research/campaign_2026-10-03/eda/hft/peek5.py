import pyarrow.parquet as pq, numpy as np, pandas as pd
pf=pq.ParquetFile('datasets/valid.parquet')
P=[f'i0_p{i}' for i in range(22)]
for g in [0,1,2]:
    t=pf.read_row_group(g).to_pandas()
    X=t[P].values
    E=np.r_[False,(np.diff(X,axis=0)!=0).any(1)]
    Eb=np.r_[False,(np.diff(X[:,:11],axis=0)!=0).any(1)]
    for c in ['t0','t1']:
        y=t[c].values; d=np.r_[False,np.diff(y)!=0]
        base=d.mean()
        lags=np.arange(-50,3001)
        res=[]
        n=len(y)
        for k in lags:
            # P(d(t)!=0 | E(t+k))
            if k>=0: a=d[:n-k]; b=E[k:]
            else: a=d[-k:]; b=E[:n+k]
            res.append(a[b].mean())
        res=np.array(res)
        top=np.argsort(-res)[:8]
        print(g,c,'base',round(base,3),'P(d|E same row)',round(res[lags==0][0],3),'top lags',list(zip(lags[top],np.round(res[top],3))))
