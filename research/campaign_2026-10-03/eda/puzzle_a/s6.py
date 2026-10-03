import numpy as np, pandas as pd
np.set_printoptions(linewidth=250, precision=2, suppress=True)
from util import *
df = load("train", 0)
P = df[[f"i0_p{i}" for i in range(22)]].values.astype(np.float64)
order = np.argsort(P.mean(0))[::-1]
Ps = P[:, order]
M = (Ps[:,:,None] > Ps[:,None,:]).mean(0)
print("order", order)
print(M)
# all distinct values pooled
u = np.unique(np.r_[P.ravel(), df[[f"i1_p{i}" for i in range(22)]].values.ravel(), df[[f"i0_dp{i}" for i in range(4)]].values.ravel()])
print("n unique price vals", len(u))
print(u[(u>-0.6)&(u<0.6)])
