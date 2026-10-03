import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.precision", 3)
from util import *
df = load("train", 0)
P = df[[f"i0_p{i}" for i in range(22)]].values
order = np.argsort(P.mean(0))[::-1]
print("order by mean desc:", order)
Ps = P[:, order]
print("frac rows monotone desc:", np.mean(np.all(np.diff(Ps,axis=1) <= 0, axis=1)), np.mean(np.all(np.diff(Ps,axis=1) < 0, axis=1)))
sub = df.iloc[1000:1030]
cols = [f"i0_p{i}" for i in order]
print(sub[cols].to_string())
vcols = [f"i0_v{i}" for i in order]
print(sub[vcols].to_string())
print(sub[[f"i0_dp{i}" for i in range(4)]+[f"i0_dv{i}" for i in range(4)]+["t0","t1"]+[f"a{i}" for i in range(8)]].to_string())
