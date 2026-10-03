import numpy as np, pandas as pd
from scipy.stats import spearmanr
from util import *
np.set_printoptions(linewidth=200, precision=3, suppress=True)
for rg in [0, 2, 500]:
    df = load("train", rg)
    for inst in ["i0","i1"]:
        for base, grp in [(0, range(0,11)), (11, range(11,22))]:
            x = df[f"{inst}_p{base}"].values
            r = [spearmanr(x, df[f"{inst}_p{k}"].values)[0] for k in grp]
            print(rg, inst, f"base p{base}", np.array(r))
        x = df[f"{inst}_dp0"].values
        print(rg, inst, "dp0 vs dp1..3", [round(spearmanr(x, df[f"{inst}_dp{k}"].values)[0],3) for k in range(1,4)],
              "dv0 vs dv1..3", [round(spearmanr(df[f"{inst}_dv0"].values, df[f"{inst}_dv{k}"].values)[0],3) for k in range(1,4)])
        print(rg, inst, "p0 vs p11 spearman", round(spearmanr(df[f"{inst}_p0"].values, df[f"{inst}_p11"].values)[0],3))
