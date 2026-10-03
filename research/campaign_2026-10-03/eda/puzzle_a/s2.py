import numpy as np, pyarrow.parquet as pq
from util import *
out = {}
for name in ["train","valid","valid_mask"]:
    p = pf(name); n = p.metadata.num_row_groups
    ids = np.empty(n, np.int64); nun = 0
    sc = []
    for rg in range(n):
        cols = ["seq_ix"] + (["is_scored"] if name != "train" else [])
        t = p.read_row_group(rg, columns=cols)
        s = t.column("seq_ix").to_numpy()
        if s.min() != s.max(): nun += 1
        ids[rg] = s[0]
        if name != "train":
            sc.append(t.column("is_scored").to_numpy())
    out[name] = ids
    print(name, n, "nonuniform rgs", nun, "min", ids.min(), "max", ids.max(), "sorted?", np.all(np.diff(ids) > 0), "unique", len(np.unique(ids)))
    if name != "train":
        sc = np.stack(sc)
        np.save(f"scored_{name}.npy", sc)
        print(" scored frac", sc.mean(), "per-seq frac quantiles", np.quantile(sc.mean(1), [0,.1,.5,.9,1]))
    np.save(f"ids_{name}.npy", ids)
tr, va = out["train"], out["valid"]
print("overlap train/valid ids", len(np.intersect1d(tr, va)))
print("valid==mask ids", np.array_equal(out["valid"], out["valid_mask"]))
allids = np.sort(np.concatenate([tr, va]))
print("all ids range", allids.min(), allids.max(), "count", len(allids), "gaps stats", np.quantile(np.diff(allids), [0,.5,.9,.99,1]))
print("first 30 train ids", tr[:30]); print("first 30 valid ids", va[:30])
