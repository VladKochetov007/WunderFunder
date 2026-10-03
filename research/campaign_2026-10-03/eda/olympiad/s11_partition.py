import pyarrow.parquet as pq, numpy as np, pandas as pd
pf = pq.ParquetFile("datasets/train.parquet")
names = [c for c in pf.schema_arrow.names if c not in ("seq_ix","step_in_seq","need_prediction")]
for rg in [0, 1, 500, 3000, 7000]:
    df = pf.read_row_group(rg, columns=names).to_pandas()
    C = (np.diff(df.values, axis=0)!=0)
    keys = {}
    for j,c in enumerate(names):
        k = C[:,j].tobytes()
        keys.setdefault(k, []).append(c)
    blocks = sorted(keys.values(), key=lambda b: names.index(b[0]))
    print("rg", rg, "nblocks", len(blocks))
    for b in blocks:
        if len(b)>1 or b[0] in ("t0","t1"): print("   ", len(b), b[0], "..", b[-1])
    print("   singletons:", [b[0] for b in blocks if len(b)==1])
