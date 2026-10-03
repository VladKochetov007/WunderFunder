import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option("display.width",250)
pf = pq.ParquetFile("datasets/train.parquet")
Q = 0.65686768
for rg in [500, 7000]:
    df = pf.read_row_group(rg).to_pandas()
    k0 = np.round(df.t0/Q).astype(int); k1 = np.round(df.t1/Q).astype(int)
    s = (df.a6.values < -5.19); s[0]=True
    dpc = np.r_[True, df.i0_dp0.values[1:]!=df.i0_dp0.values[:-1]]
    print("rg", rg, "anchors", s.sum(), "dp changes", dpc.sum(), "anchor&dpchange", (s&dpc).sum())
    a = df[s]
    g = a.groupby(["i0_dp0","i0_dp1","i0_dp2","i0_dp3"]).size().sort_values(ascending=False)
    print("distinct dp states at anchors", len(g)); print(g.head(12).to_string())
    # does the dp state at the anchor + AB state predict next-anchor change? look at k by dp state
    a = a.assign(k0=k0[s].values, k1=k1[s].values)
    print(a.groupby("i0_dp0").agg(n=("k0","size"), k0=("k0","mean"), k1=("k1","mean")).sort_values("n", ascending=False).head(10).round(3).to_string())
    # dp at non-anchor dp-change rows
    o = df[dpc & ~s]
    print("non-anchor dp changes", len(o), "distinct", o.i0_dp0.nunique(), "share of rows where dp2==0:", (df.i0_dp2==0).mean().round(3))
