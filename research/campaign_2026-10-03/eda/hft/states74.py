import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option('display.width',250)
pf=pq.ParquetFile('datasets/valid.parquet')
P0=[f'i0_p{i}' for i in range(22)]
g=74
t=pf.read_row_group(g,columns=P0+['t0','t1']).to_pandas()
q=0.65687
X=t[P0].values
U,sid=np.unique(X,axis=0,return_inverse=True); sid=sid.ravel()
k0=np.round(t.t0.values/q).astype(int); k1=np.round(t.t1.values/q).astype(int)
df=pd.DataFrame(dict(s=sid,k0=k0,k1=k1,ks=k0+k1))
print(df.groupby('s').size())
for s in range(len(U)):
    sub=df[df.s==s]
    print('state',s,'n',len(sub),'p0,p3,p11,p19=',np.round(U[s][[0,3,11,19]],3),' k0+k1 vc',sub.ks.value_counts().to_dict())
    print('     (k0,k1) vc', pd.Series(list(zip(sub.k0,sub.k1))).value_counts().head(8).to_dict())
# transitions matrix
tr=pd.crosstab(sid[:-1][sid[1:]!=sid[:-1]], sid[1:][sid[1:]!=sid[:-1]])
print(tr)
