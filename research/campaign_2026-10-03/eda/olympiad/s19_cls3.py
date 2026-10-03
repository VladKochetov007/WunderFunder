import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option("display.width",250); pd.set_option("display.max_columns",30)
pf = pq.ParquetFile("datasets/train.parquet")
Q = 0.65686768
for rg in [481, 555]:
    df = pf.read_row_group(rg).to_pandas()
    df["k0"]=np.round(df.t0/Q).astype(int); df["k1"]=np.round(df.t1/Q).astype(int)
    g = df.groupby(["i0_p0","i0_p11"]).agg(n=("k0","size"), k0=("k0","mean"), k1=("k1","mean"), first=("step_in_seq","min"), last=("step_in_seq","max"),
        p3=("i0_p3","first"), p8=("i0_p8","first"), p14=("i0_p14","first"), p21=("i0_p21","first")).reset_index().sort_values("first")
    print("rg", rg, "nstates", len(g)); print(g.round(3).to_string())
    # state sequence of first 60 changes
    s = df.groupby(["i0_p0","i0_p11"]).ngroup().values
    ch = np.r_[True, s[1:]!=s[:-1]]
    print("state seq:", s[ch][:80].tolist())
