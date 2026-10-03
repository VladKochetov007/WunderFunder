import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)
from util import *
qq = 0.65687; rg = 3
df = load("train", rg)
k0 = np.round(df.t0.values/qq).astype(int); k1 = np.round(df.t1.values/qq).astype(int)
P = df[[f"i0_p{i}" for i in range(22)]].values
sid = pd.factorize(pd.Series([r.tobytes() for r in P]))[0]
D = df[[f"i0_dp0", "i0_dv0"]].values
did = pd.factorize(pd.Series([r.tobytes() for r in D]))[0]
V = df[[f"i0_v{i}" for i in range(22)]].values
vid = pd.factorize(pd.Series([r.tobytes() for r in V]))[0]
i0chg = np.r_[True, (np.diff(sid)!=0)|(np.diff(did)!=0)|(np.diff(vid)!=0)]
ev = np.flatnonzero(i0chg)
print("n i0 events", len(ev), "of", len(df))
out = pd.DataFrame({"row": ev, "sid": sid[ev], "p7": df.i0_p7.values[ev].round(3), "p11": df.i0_p11.values[ev].round(3), "dp0": df.i0_dp0.values[ev].round(3), "dv0": df.i0_dv0.values[ev].round(3), "k0": k0[ev], "k1": k1[ev]})
out["pchg"] = np.r_[True, np.diff(out.sid.values)!=0]
out["dchg"] = np.r_[True, np.diff(did[ev])!=0]
print(out.iloc[300:420].to_string())
