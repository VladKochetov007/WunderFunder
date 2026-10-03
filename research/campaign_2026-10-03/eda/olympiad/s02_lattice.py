import pyarrow.parquet as pq, numpy as np, pandas as pd
pf = pq.ParquetFile("datasets/train.parquet")
names = pf.schema_arrow.names
for rg in [0, 1, 500, 5000, 10606]:
    df = pf.read_row_group(rg, columns=["seq_ix","t0","t1","a5","a6","a7","i0_dp2","i0_dv0"]).to_pandas()
    print("== rg", rg, "seq", df.seq_ix.iloc[0])
    for c in ["t0","t1"]:
        x = df[c].values.astype(np.float64)
        u = np.unique(x)
        pos = u[u>0]
        q = pos.min()
        k = x/q
        print(c, "nuniq", len(u), "minpos", q, "max|k-round|", np.abs(k-np.round(k)).max(), "krange", k.min().round(2), k.max().round(2),
              "frac0", np.mean(x==0), "nruns", 1+np.sum(np.diff(x)!=0))
    print("corr t0 t1", np.corrcoef(df.t0, df.t1)[0,1])
