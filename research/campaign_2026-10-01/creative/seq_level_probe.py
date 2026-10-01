"""Move 1 probe: decompose pooled |y|-weighted covariance into between-/within-sequence parts and sweep a causal
sequence-level correction p' = p - lam * causal_running_mean(p) (mean over rows 99..t of the sequence), lam per class,
cross-fitted over the 4 clean folds (fit lam on 3 folds, score the 4th)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import load_campaign_split  # noqa: E402
z = np.load(ROOT / "cache/valid_targets.npz"); Y, M, C = z["y"], z["mask"].copy(), z["market_class"]; M[:, :99] = False
pos = {int(g): i for i, g in enumerate(z["row_group"])}
lams = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
# sums[fold][class][lam] -> (2 targets, 6 raw sums)
S = np.zeros((4, 4, len(lams), 2, 6)); dec = np.zeros((4, 2, 4))  # between/within cov & var parts
for f in range(4):
    split = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json", ROOT)
    A = [np.load(ROOT / f"cache/pred_c20260929_ft_since_s{s}_f{f}_valid.npy", mmap_mode="r") for s in (3, 4)]
    for g in split.evaluation_groups:
        i = pos[g]; m = M[i]
        if not m.any(): continue
        p = 0.5 * (np.asarray(A[0][i], np.float64) + np.asarray(A[1][i], np.float64))
        run = np.zeros_like(p); cnt = np.arange(1, 20000 - 99 + 1)[:, None]
        run[99:] = np.cumsum(p[99:], 0) / cnt  # causal mean of predictions from row 99 to t
        y = np.clip(Y[i][m].astype(np.float64), -2, 2); w = np.abs(y)
        for k, lam in enumerate(lams):
            q = np.clip(p[m] - lam * run[m], -2, 2)
            S[f, C[i], k] += np.stack([w.sum(0), (w*y).sum(0), (w*q).sum(0), (w*y*y).sum(0), (w*q*q).sum(0), (w*y*q).sum(0)], -1)
def wp(s):
    n, sy, sp, syy, spp, syp = np.moveaxis(s, -1, 0)
    return float(np.mean((syp/n - sy*sp/n**2) / np.sqrt((syy/n - (sy/n)**2) * (spp/n - (sp/n)**2))))
base = lams.tolist().index(0.0)
tot = {}
for f in range(4):
    others = [g for g in range(4) if g != f]
    best = []
    for c in range(4):
        # pick lam_c on other folds with other classes at lam=0
        scores = []
        for k in range(len(lams)):
            s = sum(S[g, cc, base] for g in others for cc in range(4) if cc != c) + sum(S[g, c, k] for g in others)
            scores.append(wp(s))
        best.append(int(np.argmax(scores)))
    s_new = sum(S[f, c, best[c]] for c in range(4)); s_old = sum(S[f, c, base] for c in range(4))
    tot.setdefault("new", []).append(s_new); tot.setdefault("old", []).append(s_old)
    print(f"fold {f} lam per class {[lams[b] for b in best]} base {wp(s_old):.6f} new {wp(s_new):.6f} delta {wp(s_new)-wp(s_old):+.6f}")
print("pooled delta", wp(sum(tot['new'])) - wp(sum(tot['old'])))
for k, lam in enumerate(lams):
    print(f"in-sample all folds lam {lam:+.1f} (all classes) wp {wp(S[:, :, k].sum((0,1))):.6f}  class3-only lam: {wp(sum(S[:, c, base if c != 3 else k].sum(0) for c in range(4))):.6f}")
