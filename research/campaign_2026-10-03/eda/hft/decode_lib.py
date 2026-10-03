import numpy as np, pickle
P0=[f'i0_p{i}' for i in range(22)]
_m=None
def load():
    global _m
    if _m is None: _m=pickle.load(open('research/campaign_2026-10-03/eda/hft/decoder.pkl','rb'))
    return _m
def decode(X):
    """X: (T,22) i0 price cols. Returns a,b (ticks, cumulative from 0), spread_hat (ticks). Causal."""
    m=load()
    X=X.astype(np.float32)
    bk=np.r_[False,(np.diff(X,axis=0)!=0).any(1)]; i=np.where(bk)[0]
    F=np.c_[X[i-1],X[i],X[i]-X[i-1]]
    da=np.zeros(len(X)); db=np.zeros(len(X))
    if len(i):
        da[i]=m['da'].classes_[m['da'].predict_proba(F).argmax(1)]
        db[i]=m['db'].classes_[m['db'].predict_proba(F).argmax(1)]
    pr=m['sp'].predict_proba(X); sp=(pr*m['sp'].classes_).sum(1)
    return np.cumsum(da),np.cumsum(db),sp,bk
def ema(x,span):
    a=2/(span+1); out=np.empty_like(x,dtype=np.float64); s=x[0]
    for k in range(len(x)):
        s=s+a*(x[k]-s); out[k]=s
    return out
