import pyarrow.parquet as pq, numpy as np, pandas as pd, sys
sys.path.insert(0,'research/campaign_2026-10-03/eda/hft')
pd.set_option('display.width',250)
pf=pq.ParquetFile('datasets/valid.parquet')
sc=pd.read_csv('research/campaign_2026-10-03/eda/hft/scan_seq.csv')
P0=[f'i0_p{i}' for i in range(22)]; V0=[f'i0_v{i}' for i in range(22)]
DP0=[f'i0_dp{i}' for i in range(4)]; DV0=[f'i0_dv{i}' for i in range(4)]
P1=[f'i1_p{i}' for i in range(22)]; V1=[f'i1_v{i}' for i in range(22)]; DV1=[f'i1_dv{i}' for i in range(4)]
A=[f'a{i}' for i in range(8)]
out=[]
for g in sc.g.values[::3]:
    t=pf.read_row_group(int(g)).to_pandas()
    y0=t.t0.values; y1=t.t1.values
    v=np.unique(np.abs(np.r_[np.diff(y0),np.diff(y1),y0,y1])); q=v[v>1e-6].min()
    ch=lambda cols: np.r_[False,(np.diff(t[cols].values,axis=0)!=0).any(1)]
    ev=dict(bk0=ch(P0),vol0=ch(V0),tr0=ch(DV0),bk1=ch(P1),vol1=ch(V1),tr1=ch(DV1),
            tgt=np.r_[False,(np.diff(y0)!=0)|(np.diff(y1)!=0)])
    r=dict(g=g,q=round(q,3))
    for a in A:
        x=t[a].values; fl=x<-5.19
        r[f'{a}_fl']=fl.mean()
        r[f'{a}_ac1']=np.corrcoef(x[1:],x[:-1])[0,1]
        for k,e in ev.items():
            r[f'{a}|{k}']= fl[e].mean() if fl.any() else np.nan   # P(floor | event)
        r[f'{a}_mean']=x.mean()
    out.append(r)
df=pd.DataFrame(out)
for a in A:
    cols=[c for c in df.columns if c.startswith(a+'_') or c.startswith(a+'|')]
    print(a, df[cols].median().round(3).to_dict())
print(df.groupby('q')[[f'a{i}_mean' for i in range(8)]+[f'a{i}_fl' for i in range(8)]].mean().round(2).to_string())
