import pyarrow.parquet as pq, numpy as np, pandas as pd
pf=pq.ParquetFile('datasets/valid.parquet')
rows=[]
for g in range(0,1873,37):
    t=pf.read_row_group(g,columns=['t0','t1','is_scored']).to_pandas()
    y0=t.t0.values; y1=t.t1.values
    d0=np.diff(y0); d1=np.diff(y1); nz=d0[d0!=0]; nz1=d1[d1!=0]
    q=np.min(np.abs(np.r_[nz,nz1])) if len(nz) else np.nan
    rows.append(dict(g=g,corr=np.corrcoef(y0,y1)[0,1],corr_d=np.corrcoef(d0,d1)[0,1],sd0=y0.std(),sd1=y1.std(),m0=y0.mean(),m1=y1.mean(),q=q,
       zero0=(y0==0).mean(), ch0=(d0!=0).mean(), sc=t.is_scored.mean()))
df=pd.DataFrame(rows); pd.set_option('display.width',200)
print(df.round(3).to_string())
print(df.describe().round(3))
