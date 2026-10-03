import numpy as np, sys
from util import *
name = sys.argv[1]
p = pf(name); n = p.metadata.num_row_groups
cols = ["t0","t1"] + [f"a{i}" for i in range(8)]
rec = {k: [] for k in ["q0","q1","sd0","sd1","z0","z1","kmax0","kmax1","m0","m1","c01","amean","astd","amin","anu"]}
for rg in range(n):
    t = p.read_row_group(rg, columns=cols)
    t0 = t.column("t0").to_numpy().astype(np.float64); t1 = t.column("t1").to_numpy().astype(np.float64)
    for j,x in enumerate([t0,t1]):
        nz = np.abs(x[x!=0]); q = nz.min() if len(nz) else np.nan
        rec[f"q{j}"].append(q); rec[f"sd{j}"].append(x.std()); rec[f"z{j}"].append((x==0).mean())
        rec[f"kmax{j}"].append(np.abs(x).max()/q); rec[f"m{j}"].append(x.mean())
    rec["c01"].append(np.corrcoef(t0,t1)[0,1])
    A = np.stack([t.column(f"a{i}").to_numpy() for i in range(8)],1).astype(np.float64)
    rec["amean"].append(A.mean(0)); rec["astd"].append(A.std(0)); rec["amin"].append((A<=-5.19).mean(0))
    rec["anu"].append([len(np.unique(A[:,i])) for i in range(8)])
    if rg % 1000 == 0: print(rg, flush=True)
np.savez(f"scan1_{name}.npz", **{k: np.array(v) for k,v in rec.items()})
print("done")
