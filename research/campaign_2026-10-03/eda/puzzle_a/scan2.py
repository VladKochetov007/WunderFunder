import numpy as np, sys
from util import *
name = sys.argv[1]; nmax = int(sys.argv[2])
p = pf(name); n = min(p.metadata.num_row_groups, nmax)
cols = [f"i0_p{i}" for i in range(22)] + [f"i1_p{i}" for i in range(22)] + ["i0_dv0", "i1_dv0", "a7", "a6"]
feats = []
for rg in range(n):
    t = p.read_row_group(rg, columns=cols)
    X = np.stack([t.column(c).to_numpy() for c in cols], 1)
    f = []
    for W in [99, 20000]:
        Xw = X[:W]
        P0 = Xw[:, :22]; P1 = Xw[:, 22:44]
        # modal value of each price column
        for blk in [P0, P1]:
            for j in range(22):
                u, c = np.unique(blk[:, j], return_counts=True); f.append(u[np.argmax(c)])
        f.append(np.mean(np.any(P0[1:] != P0[:-1], 1)) if W > 1 else 0)
        f.append(np.mean(np.any(P1[1:] != P1[:-1], 1)))
        f.append(len(np.unique(P0[:, 7]))); f.append(len(np.unique(P1[:, 7])))
        f.append(np.mean(Xw[:, 44] != np.median(Xw[:, 44]))); f.append(np.mean(Xw[:, 45] != np.median(Xw[:, 45])))
        f.append(np.mean(Xw[:, 46] <= -5.19)); f.append(np.mean(Xw[:, 47] <= -5.19))
    feats.append(f)
    if rg % 1000 == 0: print(rg, flush=True)
np.save(f"scan2_{name}.npy", np.array(feats, np.float32)); print("done")
