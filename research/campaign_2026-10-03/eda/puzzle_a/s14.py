import numpy as np, pandas as pd
from util import *
d = np.load("scan1_train.npz"); qa = np.round(np.minimum(d["q0"], d["q1"]), 5)
for rg in [0,1,2,3,4,5,12,500,5000]:
    qq = qa[rg]
    df = load("train", rg)
    k0 = np.round(df.t0.values/qq).astype(int); k1 = np.round(df.t1.values/qq).astype(int); s = k0 + k1
    P = df[[f"i0_p{i}" for i in range(22)]].values
    sid = pd.factorize(pd.Series([r.tobytes() for r in P]))[0]
    g = pd.DataFrame({"sid": sid, "s": s, "k0": k0})
    mode = g.groupby("sid").s.agg(lambda x: x.value_counts().index[0])
    hit = (g.s.values == mode.reindex(g.sid).values).mean()
    # also check if bid-side state alone or ask-side alone determines
    print(f"rg {rg} q {qq} nstates {sid.max()+1} frac(sum k0+k1 == mode per price-state) {hit:.4f}  var(s)/var(k0)={s.var()/k0.var():.3f}  corr(s,k0)={np.corrcoef(s,k0)[0,1]:.3f}")
