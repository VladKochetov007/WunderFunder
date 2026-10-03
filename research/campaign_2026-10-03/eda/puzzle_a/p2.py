import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from util import *
fig, ax = plt.subplots(4, 11, figsize=(30,11))
for r, rg in enumerate([0, 2]):
    df = load("train", rg)
    for j, (base, grp) in enumerate([(0, range(0,11)), (11, range(11,22))]):
        x = df[f"i0_p{base}"].values
        for k_i, k in enumerate(grp):
            a = ax[r*2+j, k_i]
            a.scatter(x, df[f"i0_p{k}"].values, s=2)
            a.set_title(f"seq{rg} i0_p{k} vs p{base}", fontsize=8)
plt.tight_layout(); plt.savefig("fig03_price_cols_vs_base.png", dpi=50)
