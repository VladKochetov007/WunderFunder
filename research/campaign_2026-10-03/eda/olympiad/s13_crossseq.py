import pyarrow.parquet as pq, numpy as np, pandas as pd
pf = pq.ParquetFile("datasets/train.parquet")
rgs = list(range(0, 10607, 270))[:40]
cols = [f"i0_p{k}" for k in range(22)] + [f"i0_dp{k}" for k in range(4)] + [f"i0_dv{k}" for k in range(4)]+["seq_ix"]
parts = [pf.read_row_group(r, columns=cols).to_pandas() for r in rgs]
df = pd.concat(parts, ignore_index=True)
def fd(x, y):
    g = pd.DataFrame({"x":x,"y":y}).groupby("x")["y"].nunique()
    return int((g>1).sum()), len(g)
for base, others in [("i0_p0",["i0_p1","i0_p3","i0_p10"]),("i0_p11",["i0_p12","i0_p19"]),("i0_dp0",["i0_dp1","i0_dp2","i0_dp3"]),("i0_dv0",["i0_dv1","i0_dv2","i0_dv3"])]:
    for o in others:
        print(base, "->", o, "conflicting x values / distinct x:", fd(df[base].values, df[o].values))
# how many sequences share values
for c in ["i0_p0","i0_dp0","i0_dv0"]:
    g = df.groupby(c)["seq_ix"].nunique()
    print(c, "distinct values", len(g), "values seen in >1 sequence", (g>1).sum())
