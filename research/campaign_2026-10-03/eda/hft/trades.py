import pyarrow.parquet as pq, numpy as np, pandas as pd, sys
sys.path.insert(0,'research/campaign_2026-10-03/eda/hft')
from decode_lib import decode, P0
pd.set_option('display.width',250)
pf=pq.ParquetFile('datasets/valid.parquet')
sc=pd.read_csv('research/campaign_2026-10-03/eda/hft/scan_seq.csv')
DP0=[f'i0_dp{i}' for i in range(4)]; DV0=[f'i0_dv{i}' for i in range(4)]
res=[]
allv=[]
for g in sc.g.values[::4]:
    t=pf.read_row_group(int(g),columns=P0+DP0+DV0+['t0','t1']).to_pandas()
    A,B,sp,bk=decode(t[P0].values)
    da=np.r_[0,np.diff(A)]; db=np.r_[0,np.diff(B)]
    DV=t[DV0].values; DP=t[DP0].values
    base=pd.Series(DV[:,0]).mode()[0]
    tr=np.r_[False,(np.diff(DV,axis=0)!=0).any(1)]
    nz=(DV!=base)
    allv.append(DV[tr][:200])
    r=dict(g=g,base=round(base,3),trfrac=tr.mean(),slots_used=nz[tr].sum(1).mean())
    x=(DV-base)
    for k in range(4):
        r[f'c_dv{k}_dm']=np.corrcoef(x[tr,k],(da+db)[tr])[0,1]
        r[f'c_dv{k}_da']=np.corrcoef(x[tr,k],da[tr])[0,1]
        r[f'c_dv{k}_db']=np.corrcoef(x[tr,k],db[tr])[0,1]
    res.append(r)
df=pd.DataFrame(res); print(df.round(3).to_string())
V=np.vstack(allv); print('dv value sample quantiles', np.round(np.quantile(V[:,0],[0,.05,.25,.5,.75,.95,1]),3))
print('frac dv0 above base', (V[:,0]>0.08).mean(), 'below', (V[:,0]<0.07).mean())
