"""Cross-fitted per-class output scale under pooled Global WP (reference since pair, 4 clean folds).

Fits multipliers a_c (class 0 fixed at 1) on three folds' OOF rows, applies them to the fourth, scores exactly.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import load_campaign_split  # noqa: E402

z = np.load(ROOT / "cache/valid_targets.npz")
Y, M, cls = z["y"], z["mask"].copy(), z["market_class"]
M[:, :99] = False
pos = {int(g): i for i, g in enumerate(z["row_group"])}
fold_rows = {}
for f in range(4):
    split = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json", ROOT)
    fold_rows[f] = np.array([pos[g] for g in split.evaluation_groups])
P = {}
for f in range(4):
    a = np.load(ROOT / f"cache/pred_c20260929_ft_since_s3_f{f}_valid.npy", mmap_mode="r")
    b = np.load(ROOT / f"cache/pred_c20260929_ft_since_s4_f{f}_valid.npy", mmap_mode="r")
    for i in fold_rows[f]:
        P[i] = 0.5 * (np.asarray(a[i], np.float64) + np.asarray(b[i], np.float64))

# per-class, per-target uncentred sums over rows: [w, wy, wp, wyy, wpp, wyp]
def class_sums(rows):
    s = np.zeros((4, 2, 6))
    for i in rows:
        m = M[i]
        if not m.any():
            continue
        y = np.clip(Y[i][m].astype(np.float64), -2, 2); p = np.clip(P[i][m], -2, 2); w = np.abs(y)
        s[cls[i]] += np.stack([w.sum(0), (w*y).sum(0), (w*p).sum(0), (w*y*y).sum(0), (w*p*p).sum(0), (w*y*p).sum(0)], -1)
    return s

def wp(s, a):
    a = np.asarray(a)[:, None]
    n = s[..., 0].sum(0); sy = s[..., 1].sum(0); sp = (a*s[..., 2]).sum(0)
    syy = s[..., 3].sum(0); spp = (a*a*s[..., 4]).sum(0); syp = (a*s[..., 5]).sum(0)
    cov = syp/n - sy*sp/n**2; vy = syy/n - (sy/n)**2; vp = spp/n - (sp/n)**2
    return float(np.mean(cov/np.sqrt(vy*vp)))

S = {f: class_sums(fold_rows[f]) for f in range(4)}
out = {}
tot_base, tot_fit = np.zeros((4, 2, 6)), np.zeros((4, 2, 6))
for f in range(4):
    other = sum(S[g] for g in range(4) if g != f)
    res = minimize(lambda v: -wp(other, np.r_[1.0, np.exp(v)]), np.zeros(3), method="Nelder-Mead")
    a = np.r_[1.0, np.exp(res.x)]
    out[f] = {"a": a.round(4).tolist(), "base": wp(S[f], np.ones(4)), "scaled": wp(S[f], a)}
    print(f, out[f], "delta", out[f]["scaled"] - out[f]["base"])
allS = sum(S.values())
res = minimize(lambda v: -wp(allS, np.r_[1.0, np.exp(v)]), np.zeros(3), method="Nelder-Mead")
print("in-sample all-folds a", np.r_[1, np.exp(res.x)].round(4), "gain", -res.fun - wp(allS, np.ones(4)))
json.dump(out, open(ROOT / "research/campaign_2026-10-01/class_scale_probe.json", "w"), indent=1)
