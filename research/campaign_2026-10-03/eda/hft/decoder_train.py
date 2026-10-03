# Train a causal "book-transition decoder": from (x_{t-1}, x_t) of i0 price cols predict
# d_ask, d_bid (ticks) and the current spread (ticks). Labels come from target jumps in TRAIN only.
import pyarrow.parquet as pq, numpy as np, pandas as pd, pickle, time
from sklearn.ensemble import HistGradientBoostingClassifier
pf=pq.ParquetFile('datasets/train.parquet')
P0=[f'i0_p{i}' for i in range(22)]
rng=np.random.default_rng(0); gs=rng.choice(pf.metadata.num_row_groups,60,replace=False)
Xe,Ya,Yb,Xs,Ys=[],[],[],[],[]
for g in gs:
    t=pf.read_row_group(int(g),columns=P0+['t0','t1']).to_pandas()
    y0=t.t0.values.astype(np.float64); y1=t.t1.values.astype(np.float64)
    v=np.unique(np.abs(np.r_[np.diff(y0),np.diff(y1),y0,y1])); q=v[v>1e-6].min()
    X=t[P0].values.astype(np.float32)
    bk=np.r_[False,(np.diff(X,axis=0)!=0).any(1)]
    i=np.where(bk)[0]
    dk0=np.round((y0[i]-y0[i-1])/q); dk1=np.round((y1[i]-y1[i-1])/q)
    Xe.append(np.c_[X[i-1],X[i],X[i]-X[i-1]]); Ya.append(np.clip(-dk0,-3,3)); Yb.append(np.clip(dk1,-3,3))
    j=rng.choice(len(X),2000,replace=False)
    Xs.append(X[j]); Ys.append(np.clip(np.round(-(y0[j]+y1[j])/q),-1,6))
Xe=np.vstack(Xe); Ya=np.concatenate(Ya).astype(int); Yb=np.concatenate(Yb).astype(int)
Xs=np.vstack(Xs); Ys=np.concatenate(Ys).astype(int)
print('events',len(Xe),'da dist',pd.Series(Ya).value_counts().to_dict(),'db dist',pd.Series(Yb).value_counts().to_dict())
print('spread dist',pd.Series(Ys).value_counts().to_dict())
m={}
for name,(X,y) in {'da':(Xe,Ya),'db':(Xe,Yb),'sp':(Xs,Ys)}.items():
    t0=time.time(); c=HistGradientBoostingClassifier(max_iter=150,learning_rate=0.15,max_leaf_nodes=63,early_stopping=True,random_state=0)
    c.fit(X,y); m[name]=c; print(name,'train acc',round(c.score(X,y),4),'iters',c.n_iter_,'sec',round(time.time()-t0))
pickle.dump(m,open('research/campaign_2026-10-03/eda/hft/decoder.pkl','wb'))
