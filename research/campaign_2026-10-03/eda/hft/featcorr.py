import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
pf=pq.ParquetFile('datasets/valid.parquet')
sc=pd.read_csv('research/campaign_2026-10-03/eda/hft/scan_seq.csv')
F=pf.schema_arrow.names[4:116]
P0=[f'i0_p{i}' for i in range(22)]
out=[]
for g in sc.g.values[::2]:
    t=pf.read_row_group(int(g)).to_pandas()
    y0=t.t0.values.astype(np.float64); y1=t.t1.values.astype(np.float64)
    v=np.unique(np.abs(np.r_[np.diff(y0),np.diff(y1),y0,y1])); q=v[v>1e-6].min()
    X=t[F].values.astype(np.float64)
    bk=np.r_[False,(np.diff(t[P0].values,axis=0)!=0).any(1)]
    dm=np.r_[0,np.diff(y1-y0)]/2/q   # = dm - dFbar (ticks)
    sp=-(y0+y1)/q
    dX=np.r_[np.zeros((1,X.shape[1])),np.diff(X,axis=0)]
    r={}
    for j,f in enumerate(F):
        a=dX[bk,j]; 
        r[('dm_vs_dX',f)]=np.corrcoef(a,dm[bk])[0,1] if a.std()>0 else np.nan
        r[('sp_vs_X',f)]=np.corrcoef(X[:,j],sp)[0,1] if X[:,j].std()>0 else np.nan
    out.append(pd.Series(r))
R=pd.concat(out,axis=1)
m=R.mean(1).unstack(0)
print(m.round(3).to_string())
