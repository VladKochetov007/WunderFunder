# Exhaustive invariant checks on all sequences of a split (targets only + i0_p0 for change rows).
import pyarrow.parquet as pq, numpy as np, pandas as pd, sys, collections
split = sys.argv[1]
pf = pq.ParquetFile(f"datasets/{split}.parquet")
Q = np.array([0.12543313, 0.24132973, 0.46853608, 0.65686768])
cnt = collections.Counter(); viol_pos = []; kmin = collections.defaultdict(lambda: 999)
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=["t0","t1","i0_p0","i0_p11","step_in_seq"])
    t0 = t.column("t0").to_numpy(); t1 = t.column("t1").to_numpy(); p = t.column("i0_p0").to_numpy(); pb = t.column("i0_p11").to_numpy()
    a = np.abs(np.r_[t0,t1]); q = a[a>0].min(); c = int(np.argmin(np.abs(Q-q)))
    k0 = np.round(t0/q).astype(int); k1 = np.round(t1/q).astype(int)
    cnt["rows"] += len(k0); cnt[f"rows_c{c}"] += len(k0)
    cnt["both>=0"] += int(np.sum((k0>=0)&(k1>=0)))
    cnt["k0>=1&k1>=-2"] += int(np.sum((k0>=1)&(k1>=-2)))
    cnt["k1>=1&k0>=-2"] += int(np.sum((k1>=1)&(k0>=-2)))
    cnt[f"c{c}_k0>=1&k1>=-2"] += int(np.sum((k0>=1)&(k1>=-2)))
    cnt["k0+k1==0"] += int(np.sum(k0+k1==0))
    kmin[c] = min(kmin[c], k0.min(), k1.min())
    T = np.r_[False, (t0[1:]!=t0[:-1])|(t1[1:]!=t1[:-1])]
    A = np.r_[False, (p[1:]!=p[:-1])|(pb[1:]!=pb[:-1])]
    v = np.where(T & ~A)[0]
    cnt["T"] += int(T.sum()); cnt["T&~(A|B)"] += len(v)
    for i in v[:5]: viol_pos.append((rg, int(i)))
print(split, dict(cnt)); print("min k per class", dict(kmin))
vp = np.array([x[1] for x in viol_pos]); 
if len(vp): print("violation step positions: quantiles", np.percentile(vp,[0,10,50,90,100]), "n", len(vp), "examples", viol_pos[:10])
