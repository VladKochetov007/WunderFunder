import numpy as np, pandas as pd
from util import *
for rg in [0,1,2,500]:
    df = load("train", rg)
    def chg(cols):
        X = df[cols].values
        return np.r_[True, np.any(X[1:] != X[:-1], axis=1)]
    c = {}
    c["i0p"] = chg([f"i0_p{i}" for i in range(22)]); c["i0v"] = chg([f"i0_v{i}" for i in range(22)])
    c["i0d"] = chg([f"i0_dp{i}" for i in range(4)]+[f"i0_dv{i}" for i in range(4)])
    c["i1p"] = chg([f"i1_p{i}" for i in range(22)]); c["i1v"] = chg([f"i1_v{i}" for i in range(22)])
    c["i1d"] = chg([f"i1_dp{i}" for i in range(4)]+[f"i1_dv{i}" for i in range(4)])
    c["t0"] = chg(["t0"]); c["t1"] = chg(["t1"])
    for i in range(8): c[f"a{i}"] = chg([f"a{i}"])
    c["i0any"] = c["i0p"]|c["i0v"]|c["i0d"]; c["i1any"] = c["i1p"]|c["i1v"]|c["i1d"]
    print("seq", rg, {k: round(v.mean(),4) for k,v in c.items()})
    # does t0 change only when i0 changes?
    for t in ["t0","t1"]:
        for s in ["i0any","i1any"]:
            print(f"   P({t} chg | {s} chg)={c[t][c[s]].mean():.3f}  P({t} chg | no {s} chg)={c[t][~c[s]].mean():.3f}")
    print("   P(t0chg&t1chg)", (c["t0"]&c["t1"]).mean(), "P(i0any & i1any)", (c["i0any"]&c["i1any"]).mean(), "P(i0any|i1any)", (c["i0any"]|c["i1any"]).mean())
