import numpy as np, pandas as pd
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.precision", 3)
from util import *
qq = 0.65687; rg = 3
df = load("train", rg)
cols = [f"i0_p{i}" for i in range(22)]
P = df[cols].values
sid = pd.factorize(pd.Series([r.tobytes() for r in P]))[0]
df["sid"] = sid; df["k0"] = np.round(df.t0/qq).astype(int); df["k1"] = np.round(df.t1/qq).astype(int); df["s"] = df.k0 + df.k1
st = df.groupby("sid")[cols].first()
st["n"] = df.groupby("sid").size()
st["s_mode"] = df.groupby("sid").s.agg(lambda x: x.value_counts().index[0])
st["s_frac"] = df.groupby("sid").s.agg(lambda x: x.value_counts(normalize=True).iloc[0])
print(st.sort_values("i0_p7").to_string())
