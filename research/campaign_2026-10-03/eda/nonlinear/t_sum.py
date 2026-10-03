import numpy as np, sys
sys.path.insert(0, "research/campaign_2026-10-03/eda/nonlinear")
from nlcommon import *
from collections import Counter
ids = seq_ids("valid", 40, seed=1)
S=[];Xs=[];Ss=[]
for rg in ids[:12]:
    d = load("valid", rg)
    t0,t1=d["t0"],d["t1"]; u=unit_of(np.concatenate([t0,t1]))
    k0=np.round(t0/u).astype(int); k1=np.round(t1/u).astype(int); s=k0+k1
    print(rg, round(u,4), 'sum dist', Counter(s[99:]).most_common(6), ' scored:', Counter(s[d['is_scored']]).most_common(4))
    x=np.stack([d[c] for c in FEATS],1)
    cc=[np.corrcoef(x[99:,j], s[99:])[0,1] if x[99:,j].std()>0 else 0 for j in range(len(FEATS))]
    top=np.argsort(-np.abs(cc))[:5]; print('   top corr with sum', [(FEATS[j], round(cc[j],3)) for j in top])
