import numpy as np, pandas as pd
np.set_printoptions(linewidth=200, precision=6, suppress=True)
from util import *
for rg in [0, 2]:
    df = load("train", rg)
    for c in ["i0_p17","i0_p4","i0_p19","i0_p3","i1_p17","i0_dp0","i0_dv0","i0_v17","i1_v17","a7","a5"]:
        u, cnt = np.unique(df[c].values, return_counts=True)
        print(rg, c, "nuniq", len(u), "top vals", u[np.argsort(-cnt)[:8]], "cnt", np.sort(cnt)[::-1][:8])
    # shared value sets
    s17 = set(np.unique(df["i0_p17"])); s4 = set(np.unique(df["i0_p4"])); s1_17 = set(np.unique(df["i1_p17"]))
    print(" shared i0_p17 & i0_p4", len(s17 & s4), " i0_p17 & i1_p17", len(s17 & s1_17), len(s17))
