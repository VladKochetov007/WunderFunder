import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option('display.width',250)
pf=pq.ParquetFile('datasets/valid.parquet')
sc=pd.read_csv('research/campaign_2026-10-03/eda/hft/scan_seq.csv')
P0=[f'i0_p{i}' for i in range(22)]; V0=[f'i0_v{i}' for i in range(22)]
DP0=[f'i0_dp{i}' for i in range(4)]; DV0=[f'i0_dv{i}' for i in range(4)]
P1=[f'i1_p{i}' for i in range(22)]; DV1=[f'i1_dv{i}' for i in range(4)]
A=[f'a{i}' for i in range(8)]
rows=[]; runs=[]; gaps=[]
for g in sc.g.values[::2]:
    t=pf.read_row_group(int(g)).to_pandas()
    s=t.is_scored.values.astype(bool)
    ch=lambda cols: np.r_[False,(np.diff(t[cols].values,axis=0)!=0).any(1)]
    ev=dict(bk0=ch(P0),vol0=ch(V0),tr0=ch(DV0),dp0=ch(DP0),bk1=ch(P1),tr1=ch(DV1))
    for j,a in enumerate(A): ev[f'{a}_floor']=t[a].values<-5.19
    for k,e in ev.items():
        rows.append(dict(g=g,ev=k,p_ev=e[99:].mean(),p_sc_given=s[99:][e[99:]].mean() if e[99:].any() else np.nan,p_sc=s[99:].mean()))
    # run lengths of scored
    d=np.diff(np.r_[0,s.astype(int),0]); st=np.where(d==1)[0]; en=np.where(d==-1)[0]
    runs+=list(en-st); gaps+=list(st[1:]-en[:-1])
    # position
df=pd.DataFrame(rows); df['lift']=df.p_sc_given/df.p_sc
print(df.groupby('ev')[['p_ev','p_sc_given','p_sc','lift']].median().round(3))
print('scored run lengths',pd.Series(runs).describe().round(1).to_dict())
print(pd.Series(runs).value_counts().head(10).to_dict())
print('gaps',pd.Series(gaps).describe().round(1).to_dict())
