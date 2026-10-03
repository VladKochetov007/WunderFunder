import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option('display.width',250)
pf=pq.ParquetFile('datasets/valid.parquet')
P=[f'i0_p{i}' for i in range(22)]
def pat(D):
    return np.array([''.join('+' if v>0 else ('-' if v<0 else '0') for v in r) for r in D])
for g in [74,0]:
    t=pf.read_row_group(g).to_pandas()
    X=t[P].values
    y0=t.t0.values; y1=t.t1.values
    v=np.unique(np.abs(np.r_[np.diff(y0),np.diff(y1)])); q=v[v>1e-6].min()
    d0=np.round(np.diff(y0)/q).astype(int); d1=np.round(np.diff(y1)/q).astype(int)
    D=np.diff(X,axis=0); m=(D!=0).any(1)
    pb=pat(D[m,:11]); pa=pat(D[m,11:])
    df=pd.DataFrame(dict(pb=pb,pa=pa,d0=d0[m],d1=d1[m]))
    top=df.groupby(['pb','pa']).size().sort_values(ascending=False).head(14)
    print('seq',g)
    for (b,a),n in top.items():
        sub=df[(df.pb==b)&(df.pa==a)]
        print(b,a,n,'d0 mean %.2f d1 mean %.2f'%(sub.d0.mean(),sub.d1.mean()),'top',pd.Series(list(zip(sub.d0,sub.d1))).value_counts().head(4).to_dict())
