import pyarrow.parquet as pq, numpy as np, pandas as pd
from scipy.stats import spearmanr
pf = pq.ParquetFile("datasets/train.parquet")
df = pf.read_row_group(0).to_pandas()
def fdep(x, y):
    # is y a function of x? fraction of rows where y differs from the y of the first occurrence of same x
    s = pd.DataFrame({"x":x,"y":y}).groupby("x")["y"].nunique()
    return (s>1).mean()
def groups(prefix, rng):
    return [f"{prefix}{k}" for k in rng]
G = {"i0_pA":groups("i0_p",range(0,11)),"i0_pB":groups("i0_p",range(11,22)),"i1_pA":groups("i1_p",range(0,11)),"i1_pB":groups("i1_p",range(11,22)),
     "i0_dp":groups("i0_dp",range(4)),"i0_dv":groups("i0_dv",range(4)),"i1_dp":groups("i1_dp",range(4)),"i1_dv":groups("i1_dv",range(4)),
     "i0_vA":groups("i0_v",range(0,11)),"i0_vB":groups("i0_v",range(11,22))}
for g, cs in G.items():
    base = cs[0]
    res = []
    for c in cs[1:]:
        res.append((c, round(fdep(df[base].values, df[c].values),4), round(fdep(df[c].values, df[base].values),4), round(spearmanr(df[base], df[c]).correlation,4)))
    print(g, "base", base, res)
# change-time coincidence
for g, cs in G.items():
    ch = np.stack([np.diff(df[c].values)!=0 for c in cs])
    print(g, "rows any change", ch.any(0).sum(), "all change", ch.all(0).sum())
