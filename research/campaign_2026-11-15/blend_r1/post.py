"""Causal post-processing on TL3C-torch (train-only, exposed valid), chain-fold cross-check.
1) EMA smoothing over steps within sequence; 2) blend weight by step bucket; 3) per-target shrink toward seq mean."""
import json
from pathlib import Path
import numpy as np
C = Path("/home/vlad/development/wunder/research_tmp/chrono_root/cache")
with np.load(C / "valid_targets.npz") as t:
    groups, mask, Y = t["row_group"], t["mask"], t["y"]
cf = np.load(C / "chain_folds.npz"); v = cf["split"] == "valid"
fmap = dict(zip(cf["row_group"][v].tolist(), (cf["fold"][v] % 2).tolist()))
seqfold = np.array([fmap[int(g)] for g in groups])
L = lambda m: np.load(C / f"pred_{m}_valid.npy", mmap_mode="r")
full = .34 * L("q14t_fbc_s5")[:] + .33 * L("q14t_tlead_s5")[:] + .33 * L("q14t_tlead_s6")[:]   # (S,T,2)
def score(p, fold):
    sel = mask & (seqfold == fold)[:, None]
    y = np.clip(Y[sel].astype(np.float64), -2, 2); q = np.clip(p[sel].astype(np.float64), -2, 2); w = np.abs(y); r = []
    for j in range(2):
        W = w[:, j].sum(); my = (w[:, j] * y[:, j]).sum() / W; mp = (w[:, j] * q[:, j]).sum() / W
        r.append((w[:, j] * (y[:, j] - my) * (q[:, j] - mp)).sum() / np.sqrt((w[:, j] * (y[:, j] - my) ** 2).sum() * (w[:, j] * (q[:, j] - mp) ** 2).sum()))
    return float(np.mean(r))
out = {"base": [score(full, f) for f in (0, 1)], "ema": {}}
for a in (0.9, 0.7, 0.5):
    s = full.copy()
    for t in range(1, s.shape[1]): s[:, t] = a * full[:, t] + (1 - a) * s[:, t - 1]
    out["ema"][a] = [score(s, f) for f in (0, 1)]; print("ema", a, out["ema"][a], flush=True)
# step-dependent: difference TLead pair vs fbc by step bucket
pair = (L("q14t_tlead_s5")[:] + L("q14t_tlead_s6")[:]) / 2; fbc = L("q14t_fbc_s5")[:]
steps = np.arange(full.shape[1]); out["by_step"] = {}
for lo, hi in ((0, 2000), (2000, 6000), (6000, 12000), (12000, 20000)):
    m2 = mask.copy(); m2[:, :lo] = False; m2[:, hi:] = False
    def sc(p):
        y = np.clip(Y[m2].astype(np.float64), -2, 2); q = np.clip(p[m2].astype(np.float64), -2, 2); w = np.abs(y); r = []
        for j in range(2):
            W = w[:, j].sum(); my = (w[:, j] * y[:, j]).sum() / W; mp = (w[:, j] * q[:, j]).sum() / W
            r.append((w[:, j] * (y[:, j] - my) * (q[:, j] - mp)).sum() / np.sqrt((w[:, j] * (y[:, j] - my) ** 2).sum() * (w[:, j] * (q[:, j] - mp) ** 2).sum()))
        return float(np.mean(r))
    out["by_step"][f"{lo}-{hi}"] = {"rows": int(m2.sum()), "fbc": sc(fbc), "pair": sc(pair), "tl3c": sc(full)}
    print(lo, hi, out["by_step"][f"{lo}-{hi}"], flush=True)
print(json.dumps(out["base"])); Path(__file__).with_name("POST_R1.json").write_text(json.dumps(out, indent=1))
