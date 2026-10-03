import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option('display.width',250)
pf=pq.ParquetFile('datasets/valid.parquet')
P0=[f'i0_p{i}' for i in range(22)]
for g in [74,111,0,555]:
    t=pf.read_row_group(g,columns=P0+['t0','t1']).to_pandas()
    y0=t.t0.values.astype(np.float64); y1=t.t1.values.astype(np.float64)
    v=np.unique(np.abs(np.r_[np.diff(y0),np.diff(y1),y0,y1])); q=v[v>1e-6].min()
    X=t[P0].values
    # state ids
    _,sid=np.unique(X,axis=0,return_inverse=True); sid=sid.ravel()
    bk=np.r_[False,sid[1:]!=sid[:-1]]
    d0=np.round(np.r_[0,np.diff(y0)]/q).astype(int); d1=np.round(np.r_[0,np.diff(y1)]/q).astype(int)
    idx=np.where(bk)[0]
    df=pd.DataFrame(dict(a=sid[idx-1],b=sid[idx],d0=d0[idx],d1=d1[idx]))
    df['o']=list(zip(df.d0,df.d1))
    gg=df.groupby(['a','b'])
    tot=len(df); dom=gg.o.agg(lambda s:s.value_counts().iloc[0]).sum()
    single=(gg.size()==1).sum()
    print(g,'q',round(q,3),'nstates',sid.max()+1,'book events',tot,'distinct transitions',gg.ngroups,'singletons',single,'dominant-outcome share',round(dom/tot,3))
    # spread determinism given state
    sp=np.round(-(y0+y1)/q).astype(int)
    ds=pd.DataFrame(dict(s=sid,sp=sp)); dd=ds.groupby('s').sp.agg(lambda s:s.value_counts().iloc[0]).sum()/len(ds)
    print('   spread(t0+t1) determined by current state share', round(dd,3))
