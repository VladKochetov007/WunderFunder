import numpy as np, pandas as pd
pd.set_option("display.width", 250)
from util import *
d = np.load("scan1_train.npz"); q = np.round(np.minimum(d["q0"], d["q1"]), 5)
for qq in [0.65687, 0.12543]:
    rgs = np.flatnonzero(q == qq)[:3]
    for rg in rgs:
        df = load("train", rg, ["t0","t1"])
        k0 = np.round(df.t0.values/qq).astype(int); k1 = np.round(df.t1.values/qq).astype(int)
        s = k0 + k1
        print("q", qq, "rg", rg, "sum k0+k1 dist:", dict(zip(*np.unique(s, return_counts=True))) if qq>0.5 else np.quantile(s,[.01,.1,.5,.9,.99]))
        if qq > 0.5: print(pd.crosstab(k0, k1))
