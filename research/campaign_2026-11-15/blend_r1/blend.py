"""Fast blend/error-structure analysis on exposed valid with chain-fold cross-fitting (fit weights on one
fold, score on the other). Train-only members only. Metric: official global WP pieces via raw sums."""
import json, sys
from pathlib import Path
import numpy as np
C = Path("/home/vlad/development/wunder/research_tmp/chrono_root/cache")
MEM = ["q14t_fbc_s5", "q14t_tlead_s5", "q14t_tlead_s6", "tt29rep_fbc1k_s5", "q14s_local_kd90_s3", "q14s_local_kd100_s3",
       "l14_kd50_s3", "l14_kd50_s4", "l14_kd90_s3", "tt29rep_fin256_s3", "tt29rep_fin256_s4", "q14s_swa23_kd90_s3"]
with np.load(C / "valid_targets.npz") as t:
    groups, mask, Y = t["row_group"], t["mask"], t["y"]
cf = np.load(C / "chain_folds.npz"); v = cf["split"] == "valid"
fmap = dict(zip(cf["row_group"][v].tolist(), (cf["fold"][v] % 2).tolist()))
seqfold = np.array([fmap[int(g)] for g in groups])
rowfold = np.repeat(seqfold, mask.sum(1))
step = np.nonzero(mask)[1]
y = np.clip(Y[mask].astype(np.float64), -2, 2)
P = {m: np.load(C / f"pred_{m}_valid.npy", mmap_mode="r")[mask].astype(np.float64) for m in MEM}
w = np.abs(y)
def wp(p, sel=None):
    r = []
    for j in range(2):
        yy, pp, ww = y[:, j], np.clip(p[:, j], -2, 2), w[:, j]
        if sel is not None: yy, pp, ww = yy[sel], pp[sel], ww[sel]
        W = ww.sum(); my, mp = (ww * yy).sum() / W, (ww * pp).sum() / W
        cov = (ww * (yy - my) * (pp - mp)).sum() / W
        r.append(cov / np.sqrt((ww * (yy - my) ** 2).sum() / W * (ww * (pp - mp) ** 2).sum() / W))
    return float(np.mean(r)), r
out = {"single": {m: wp(P[m])[0] for m in MEM}}
# residual (error) correlation: corr of (p - E[y|p] ~ scaled) -> use standardized pred minus y-projection
E = {}
for m in MEM:
    e = np.empty_like(y)
    for j in range(2):
        b = (w[:, j] * P[m][:, j] * y[:, j]).sum() / (w[:, j] * P[m][:, j] ** 2).sum()
        e[:, j] = y[:, j] - b * P[m][:, j]
    E[m] = e
out["error_corr"] = {f"{a}|{b}": float(np.mean([np.corrcoef(E[a][:, j], E[b][:, j])[0, 1] for j in range(2)]))
                     for i, a in enumerate(MEM) for b in MEM[i + 1:]}
out["pred_corr"] = {f"{a}|{b}": float(np.mean([np.corrcoef(P[a][:, j], P[b][:, j])[0, 1] for j in range(2)]))
                    for i, a in enumerate(MEM) for b in MEM[i + 1:]}
# cross-fitted weighted least squares stacking (per target, nonneg via clipping iterations), fit on fold k, score on 1-k
def wls(sel, members, j, ridge=0.0, nonneg=True):
    X = np.stack([P[m][sel, j] for m in members], 1); yy = y[sel, j]; ww = w[sel, j]
    A = (X * ww[:, None]).T @ X + ridge * np.eye(len(members)); b = (X * ww[:, None]).T @ yy
    c = np.linalg.solve(A, b)
    if nonneg:
        act = list(range(len(members)))
        while (c < 0).any() and act:
            act = [i for i in act if c[act.index(i)] > 0] if False else [act[i] for i in range(len(act)) if c[i] > 0]
            Xa = X[:, act]; c2 = np.linalg.solve((Xa * ww[:, None]).T @ Xa, (Xa * ww[:, None]).T @ yy)
            c = c2
        full = np.zeros(len(members)); full[act] = c; c = full
    return c
def crossfit(members, ridge=0.0, per_target=True):
    res, coefs = [], []
    for k in (0, 1):
        tr, te = rowfold == k, rowfold != k
        p = np.zeros((te.sum(), 2)); cc = []
        for j in range(2):
            c = wls(tr, members, j if per_target else j, ridge); cc.append(c.tolist())
            p[:, j] = np.stack([P[m][te, j] for m in members], 1) @ c
        sc = wp(np.where(te[:, None], 0, 0)[:0] if False else None) if False else None
        # score on test fold rows
        yy = y[te]; ww = w[te]
        r = []
        for j in range(2):
            W = ww[:, j].sum(); my = (ww[:, j] * yy[:, j]).sum() / W; pp = np.clip(p[:, j], -2, 2); mp = (ww[:, j] * pp).sum() / W
            r.append(((ww[:, j] * (yy[:, j] - my) * (pp - mp)).sum() / W) / np.sqrt(((ww[:, j] * (yy[:, j] - my) ** 2).sum() / W) * ((ww[:, j] * (pp - mp) ** 2).sum() / W)))
        res.append(float(np.mean(r))); coefs.append(cc)
    return res, coefs
def fixed(members, coef):
    p = sum(c * P[m] for m, c in zip(members, coef)); return [wp(p, rowfold != k)[0] for k in (0, 1)]
TL3C = (["q14t_fbc_s5", "q14t_tlead_s5", "q14t_tlead_s6"], [.34, .33, .33])
out["fold_scores"] = {"tl3c_fixed": fixed(*TL3C), "tl3c_kd90q": fixed(TL3C[0] + ["q14s_local_kd90_s3"], [.255, .2475, .2475, .25]),
                      "equal_all": fixed(MEM, [1 / len(MEM)] * len(MEM)),
                      "tlead_s5_single": fixed(["q14t_tlead_s5"], [1.0])}
for name, mem in {"stack_tl3c": TL3C[0], "stack_tl3c_kd90": TL3C[0] + ["q14s_local_kd90_s3"], "stack_all": MEM}.items():
    for ridge in (0.0,):
        r, c = crossfit(mem, ridge); out["fold_scores"][name] = r; out.setdefault("stack_coefs", {})[name] = {"members": mem, "coef_by_fitfold_target": c}
print(json.dumps({k: out[k] for k in ("single", "fold_scores")}, indent=1))
ec = sorted(out["error_corr"].items(), key=lambda kv: kv[1])
print("lowest error corr:", ec[:8]); print("highest:", ec[-4:])
print(json.dumps(out["stack_coefs"]["stack_all"], indent=0)[:1500])
Path(__file__).with_name("BLEND_R1.json").write_text(json.dumps(out, indent=1))
