import numpy as np, pandas as pd
from util import *
np.set_printoptions(linewidth=220, precision=3, suppress=True)
d = np.load("scan1_train.npz"); qa = np.round(np.minimum(d["q0"], d["q1"]), 5)
cols = [f"i0_p{i}" for i in range(22)] + [f"i1_p{i}" for i in range(22)] + [f"i0_dp{i}" for i in range(4)] + [f"i1_dp{i}" for i in range(4)] + [f"a{i}" for i in range(8)]
for cls in [0.12543, 0.65687]:
    rgs = np.flatnonzero(qa == cls)[:25]
    C = []
    for rg in rgs:
        df = load("train", rg, cols + ["t0", "t1"])
        dd = (df.t0 - df.t1).values; ss = (df.t0 + df.t1).values
        X = df[cols].values
        C.append([[np.corrcoef(X[:, j], dd)[0, 1], np.corrcoef(X[:, j], ss)[0, 1]] for j in range(len(cols))])
    C = np.nanmean(np.array(C), 0)
    print("class", cls)
    for j in np.argsort(-np.abs(C[:, 0]))[:14]:
        print(f"   {cols[j]:8s} corr(diff)={C[j,0]:+.3f} corr(sum)={C[j,1]:+.3f}")
    for j in np.argsort(-np.abs(C[:, 1]))[:6]:
        print(f"   SUM {cols[j]:8s} corr(diff)={C[j,0]:+.3f} corr(sum)={C[j,1]:+.3f}")
