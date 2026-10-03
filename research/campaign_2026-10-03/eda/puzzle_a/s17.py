import numpy as np, pandas as pd
pd.set_option("display.width", 250)
from util import *
d = np.load("scan1_train.npz"); qa = np.round(np.minimum(d["q0"], d["q1"]), 5)
for cls in [0.65687, 0.46854, 0.12543]:
    rgs = np.flatnonzero(qa == cls)[:10]
    rows = []
    for rg in rgs:
        df = load("train", rg)
        P = df[[f"i0_p{i}" for i in range(22)]].values
        pchg = np.r_[True, np.any(P[1:] != P[:-1], axis=1)]
        k0 = np.round(df.t0.values/cls).astype(int); k1 = np.round(df.t1.values/cls).astype(int)
        d0 = np.r_[0, np.diff(k0)]; d1 = np.r_[0, np.diff(k1)]
        m = (~pchg) & ((d0 != 0) | (d1 != 0))
        rows.append(pd.DataFrame({"d0": d0[m], "d1": d1[m]}))
        m2 = pchg & ((d0 != 0) | (d1 != 0))
    r = pd.concat(rows)
    print(cls, "target changes WITHOUT price-state change:", len(r))
    print(pd.crosstab(r.d0.clip(-4,4), r.d1.clip(-4,4)))
