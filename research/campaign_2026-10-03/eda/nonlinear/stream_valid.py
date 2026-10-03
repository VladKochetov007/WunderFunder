# Fit weighted ridge on the train cache (t0, t1, and d=t0-t1, s=t0+t1), then stream many valid sequences
# (disjoint from the 40-seq cache) and keep scored rows: ridge preds + causal states + targets + class unit.
import numpy as np, sys, gc
sys.path.insert(0, "research/campaign_2026-10-03/eda/nonlinear")
from nlcommon import *
from build_state import state, SN
NSEQ = int(sys.argv[1]) if len(sys.argv) > 1 else 600
TR = dict(np.load(OUT + "/cache_train48.npz"))
def design(X, DX, S):
    S = S.copy(); S[:, -1] /= 20000.0
    return np.hstack([X, DX, S]).astype(np.float32)
Xtr = design(TR["X"], TR["DX"], TR["S"]); Ytr = TR["Y"].copy(); del TR; gc.collect()
mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
Xtr -= mu; Xtr /= sd
nf = Xtr.shape[1]
betas = []
for k in range(2):
    y = np.clip(Ytr[:, k], -2, 2).astype(np.float64); w = np.abs(y) + 1e-3
    G = np.zeros((nf + 1, nf + 1)); r = np.zeros(nf + 1)
    for i in range(0, len(y), 20000):
        A = np.hstack([Xtr[i:i + 20000], np.ones((min(20000, len(y) - i), 1), np.float32)]).astype(np.float64)
        ww = w[i:i + 20000]
        G += (A * ww[:, None]).T @ A; r += (A * ww[:, None]).T @ y[i:i + 20000]
    lam = 10.0 * len(y) / 1000; R = lam * np.eye(nf + 1); R[-1, -1] = 0
    betas.append(np.linalg.solve(G + R, r))
B = np.stack(betas, 1).astype(np.float32)
np.savez(OUT + "/ridge_beta.npz", B=B, mu=mu, sd=sd)
del Xtr; gc.collect()
pf = pfile("valid"); allg = np.arange(pf.metadata.num_row_groups)
excl = set(seq_ids("valid", 40, seed=1).tolist())
rng = np.random.default_rng(7); pick = np.sort(rng.choice([g for g in allg if g not in excl], NSEQ, replace=False))
keepcols = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7"]
out = {k: [] for k in ["P", "S", "A", "Y", "U", "G"]}
for j, rg in enumerate(pick):
    d = load("valid", rg)
    x = np.stack([d[c] for c in FEATS], 1); st, dx = state(x)
    sc = d["is_scored"]
    Z = (design(x, dx, st)[sc] - mu) / sd
    p = np.hstack([Z, np.ones((len(Z), 1), np.float32)]) @ B
    u = unit_of(np.concatenate([d["t0"], d["t1"]]))
    out["P"].append(p.astype(np.float32)); out["S"].append(st[sc]); out["A"].append(np.stack([d[c] for c in keepcols], 1)[sc])
    out["Y"].append(np.stack([d["t0"], d["t1"]], 1)[sc]); out["U"].append(np.full(sc.sum(), u, np.float32)); out["G"].append(np.full(sc.sum(), rg, np.int32))
    if j % 100 == 0: print(j, flush=True)
np.savez(OUT + f"/stream_valid{NSEQ}.npz", **{k: np.concatenate(v) for k, v in out.items()}, SN=np.array(SN))
print("done", sum(len(v) for v in out["G"]))
