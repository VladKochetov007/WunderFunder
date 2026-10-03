import pyarrow.parquet as pq, numpy as np, pandas as pd
pd.set_option("display.width",250); pd.set_option("display.max_columns",40)
pf = pq.ParquetFile("datasets/valid.parquet")
print(pf.schema_arrow.names[-14:])
pm = pq.ParquetFile("datasets/valid_mask.parquet")
print("mask", pm.metadata.num_row_groups, pm.metadata.num_rows, pm.schema_arrow)
pf = pq.ParquetFile("datasets/train.parquet")
df = pf.read_row_group(0).to_pandas()
print(df.shape, df.seq_ix.unique()[:5], df.step_in_seq.min(), df.step_in_seq.max())
cols0 = [c for c in df.columns if c.startswith("i0_")]
print(df[["step_in_seq","need_prediction"]+[f"i0_p{k}" for k in range(22)]].head(5).T)
print(df[[f"i0_v{k}" for k in range(22)]].head(5).T)
print(df[[f"i0_dp{k}" for k in range(4)]+[f"i0_dv{k}" for k in range(4)]+[f"a{k}" for k in range(8)]+["t0","t1"]].head(12))
print(df.describe().T.to_string())
