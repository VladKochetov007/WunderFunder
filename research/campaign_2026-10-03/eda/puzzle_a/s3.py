import numpy as np, pandas as pd
from util import *
for name, rgs in [("train",[0,1,2,500,5000]),("valid",[0,1,2])]:
    for rg in rgs:
        df = load(name, rg, ["t0","t1"] + (["is_scored"] if name=="valid" else []))
        for t in ["t0","t1"]:
            x = df[t].values.astype(np.float64)
            u = np.unique(np.round(np.abs(x[x!=0]),6))
            q = u.min()
            k = x / q
            print(name, rg, t, "min|x|", q, "nuniq", len(np.unique(x)), "max int err", np.abs(k-np.round(k)).max(), "kmax", np.abs(k).max(), "zero frac", (x==0).mean(), "std", x.std())
        if name=="valid":
            s = df.is_scored.values
            idx = np.flatnonzero(s)
            runs = np.split(idx, np.flatnonzero(np.diff(idx)!=1)+1)
            print("  scored n", s.sum(), "runs", len(runs), "run lens", [len(r) for r in runs][:20], "starts", [r[0] for r in runs][:20])
            print("  |t0| scored vs not", np.abs(df.t0[s]).mean(), np.abs(df.t0[~s]).mean())
