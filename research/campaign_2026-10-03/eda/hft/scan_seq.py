import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option('display.width',250)
pf=pq.ParquetFile('datasets/valid.parquet')
P0=[f'i0_p{i}' for i in range(22)]; P1=[f'i1_p{i}' for i in range(22)]
DV0=[f'i0_dv{i}' for i in range(4)]; DV1=[f'i1_dv{i}' for i in range(4)]
rows=[]
for g in range(0,1873,23):
    t=pf.read_row_group(g).to_pandas()
    y0=t.t0.values; y1=t.t1.values
    v=np.unique(np.abs(np.r_[np.diff(y0),np.diff(y1),y0,y1])); q=v[v>1e-6].min()
    bk=np.r_[False,(np.diff(t[P0].values,axis=0)!=0).any(1)]
    bk1=np.r_[False,(np.diff(t[P1].values,axis=0)!=0).any(1)]
    tr=np.r_[False,(np.diff(t[DV0].values,axis=0)!=0).any(1)]
    tr1=np.r_[False,(np.diff(t[DV1].values,axis=0)!=0).any(1)]
    s=t.is_scored.values.astype(bool); npred=t.need_prediction.values.astype(bool)
    sp=np.round(-(y0+y1)/q)
    d0=np.r_[False,np.diff(y0)!=0]
    rows.append(dict(g=g,q=round(q,4),bk=bk.mean(),bk1=bk1.mean(),tr=tr.mean(),tr1=tr1.mean(),sc=s.mean(),
        sc_bk=s[bk].mean(), sc_nobk=s[~bk&npred].mean(), sc_tr=s[tr].mean() if tr.any() else np.nan, sc_d0=s[d0].mean(),
        sc_y0nz=s[(y0!=0)&npred].mean(), sc_y0z=s[(y0==0)&npred].mean(),
        sp_mode=pd.Series(sp).mode()[0], sp_mean=sp.mean(), absy=np.abs(np.clip(y0,-2,2)).mean(), sd0=y0.std()))
df=pd.DataFrame(rows)
print(df.round(3).to_string())
print(df.groupby('q').mean().round(3).to_string())
print(df.groupby('q').size())
df.to_csv('research/campaign_2026-10-03/eda/hft/scan_seq.csv',index=False)
