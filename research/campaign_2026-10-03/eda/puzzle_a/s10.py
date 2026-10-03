import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.precision", 3)
from util import *
df = load("train", 2)
for side, grp in [("bid", range(0,11)), ("ask", range(11,22))]:
    cols = [f"i0_p{k}" for k in grp]
    st = df[cols].drop_duplicates().copy()
    st["count"] = df.groupby(cols).size().reindex(pd.MultiIndex.from_frame(st[cols])).values
    print(side); print(st.sort_values(cols[0]).to_string())
# transitions in time: state id sequence
cb = [f"i0_p{k}" for k in range(11)]; ca = [f"i0_p{k}" for k in range(11,22)]
bid_id = pd.factorize(df[cb].apply(tuple, axis=1))[0]; ask_id = pd.factorize(df[ca].apply(tuple, axis=1))[0]
chg = np.flatnonzero((np.diff(bid_id)!=0) | (np.diff(ask_id)!=0)) + 1
print("n side-state changes", len(chg))
print(pd.DataFrame({"row": chg[:60], "bid": bid_id[chg[:60]], "ask": ask_id[chg[:60]], "t0": df.t0.values[chg[:60]]/0.468536, "t1": df.t1.values[chg[:60]]/0.468536}).T.to_string())
