import numpy as np, sys
sys.path.insert(0, "research/campaign_2026-10-03/eda/nonlinear")
from nlcommon import *
from collections import Counter
ids = seq_ids("valid", 40, seed=1)
res=[]
for rg in ids:
    d = load("valid", rg, ["t0","t1","is_scored"])
    t0,t1,s=d["t0"],d["t1"],d["is_scored"]
    u=unit_of(np.concatenate([t0,t1]))
    k0=np.round(t0/u).astype(int); k1=np.round(t1/u).astype(int)
    assert np.allclose(k0*u,t0,atol=1e-3) and np.allclose(k1*u,t1,atol=1e-3)
    m0=Counter(k0[99:]).most_common(3); m1=Counter(k1[99:]).most_common(3)
    c=np.corrcoef(t0[99:],t1[99:])[0,1]; cs=np.corrcoef(t0[s],t1[s])[0,1] if s.sum()>10 else np.nan
    ac0=[np.corrcoef(t0[99:-L],t0[99+L:])[0,1] for L in (1,10,50,200)]
    ch0=(np.diff(t0)!=0).mean(); ch1=(np.diff(t1)!=0).mean()
    # scored rows: target at mode vs not
    res.append((rg,u,m0[0],m1[0],round(c,3),round(cs,3),np.round(ac0,3),round(ch0,3),round(ch1,3),round(s.mean(),3),
                round(np.mean(k0[s]!=m0[0][0]) if s.sum() else np.nan,3), round(np.mean(k0[99:]!=m0[0][0]),3)))
for r in res: print(r)
