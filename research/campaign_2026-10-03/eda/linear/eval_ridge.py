"""Solve ridge models from saved Grams (block subsets, lambdas, weightings, oracle per-class) and score pooled WP on
valid scored rows (300 sequences, same subset as the univariate scan)."""
import sys, time, json, numpy as np
sys.path.insert(0, "/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-10-03/eda/linear")
from common import *
from ridge import blocks, qclass, BLOCKS, NB, QS

tag = sys.argv[1]; NV = int(sys.argv[2]) if len(sys.argv) > 2 else 300
g = np.load(f"{OUT}/gram_{tag}.npz")
G, b = g["G"], g["b"]
P = G.shape[-1]

def cols(bl):
    idx = [j * NB + i for j, nm in enumerate(BLOCKS) if nm in bl for i in range(NB)]
    return np.array(idx + [P - 1])

def solve(Gm, bm, idx, lam):
    Gs = Gm[np.ix_(idx, idx)]; bs = bm[idx]
    sw = Gs[-1, -1]
    d = np.sqrt(np.maximum(np.diag(Gs) / sw, 1e-12)); d[-1] = 1.0
    Gn = Gs / np.outer(d, d); bn = bs / d
    R = lam * sw * np.eye(len(idx)); R[-1, -1] = 0
    beta = np.linalg.solve(Gn + R, bn) / d
    full = np.zeros(P); full[idx] = beta
    return full

models = {}  # name -> (P,2) or per-class (4,P,2)
SETS = {"L": ["L"], "L+D1": ["L", "D1"], "L+E": ["L", "E32", "E256"], "L+CM": ["L", "CM"], "ALL": BLOCKS}
for sn, bl in SETS.items():
    idx = cols(bl)
    for lam in [1e-4, 1e-3, 1e-2, 1e-1]:
        models[f"{sn}|lam{lam}|w"] = np.stack([solve(G[0].sum(0)[k], b[0].sum(0)[k], idx, lam) for k in range(2)], 1)
idx = cols(BLOCKS)
for lam in [1e-3, 1e-2]:
    models[f"ALL|lam{lam}|wpi"] = np.stack([solve(G[1].sum(0)[k], b[1].sum(0)[k], idx, lam) for k in range(2)], 1)
    models[f"ALL|lam{lam}|w|oracle-class"] = np.stack([np.stack([solve(G[0, c, k], b[0, c, k], idx, lam) for k in range(2)], 1) for c in range(4)], 0)
names = list(models)
pf = pfile("valid")
rgs = np.random.default_rng(42).choice(pf.metadata.num_row_groups, NV, replace=False)
preds = {n: [] for n in names}; ys = []; cls = []
t0 = time.time()
for i, rg in enumerate(rgs):
    X, Y, ex = load(pf, rg, extra=["is_scored"])
    sc = ex["is_scored"]
    c = qclass(Y)
    T = np.c_[blocks(X)[sc], np.ones(sc.sum(), np.float32)].astype(np.float64)
    for n in names:
        B = models[n]
        preds[n].append(T @ (B[c] if B.ndim == 3 else B))
    ys.append(Y[sc]); cls.append(np.full(sc.sum(), c))
    if i % 50 == 0: print(i, round(time.time() - t0, 1), flush=True)
ys = np.concatenate(ys); cls = np.concatenate(cls)
res = {}
for n in names:
    p = np.concatenate(preds[n])
    r = [wcorr(p[:, k], ys[:, k]) for k in range(2)]
    rc = [[wcorr(p[cls == c, k], ys[cls == c, k]) for k in range(2)] for c in range(4)]
    res[n] = {"wp": float(np.mean(r)), "t0": float(r[0]), "t1": float(r[1]), "per_class_mean": [float(np.mean(x)) for x in rc]}
    print(f"{n:32s} WP={np.mean(r):.4f} (t0 {r[0]:.4f} t1 {r[1]:.4f})  per-class {[round(float(np.mean(x)), 3) for x in rc]}", flush=True)
json.dump(res, open(f"{OUT}/eval_{tag}.json", "w"), indent=1)
np.savez(f"{OUT}/models_{tag}.npz", **{n.replace("|", "__"): models[n] for n in names})
