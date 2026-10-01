"""Causal EMA of member-average predictions per sequence (reset at row 0); pooled Global WP on clean folds."""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import load_campaign_split  # noqa: E402

z = np.load(ROOT / "cache/valid_targets.npz")
Y, M = z["y"], z["mask"].copy()
M[:, :99] = False
pos = {int(g): i for i, g in enumerate(z["row_group"])}
folds = [int(f) for f in sys.argv[1].split(",")]
alphas = [(0.0,0.0),(0.5,0.1),(0.5,0.25),(0.9,0.05),(0.9,0.15),(0.97,0.05)]
acc = {a: np.zeros((2, 6)) for a in alphas}
for f in folds:
    split = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json", ROOT)
    names = [f"c20260929_ft_since_s3_f{f}h", f"c20260929_ft_since_s4_f{f}h", f"c20260929_since_s3_f{f}", f"c20260929_since_s4_f{f}"]
    arrs = [np.load(ROOT / f"cache/pred_{n}_valid.npy", mmap_mode="r") for n in names]
    for g in split.evaluation_groups:
        i = pos[g]
        m = M[i]
        if not m.any():
            continue
        p = sum(np.asarray(a[i], np.float64) for a in arrs) / len(arrs)
        y = np.clip(Y[i][m].astype(np.float64), -2, 2)
        w = np.abs(y)
        for a in alphas:
            al, beta = a
            ema = p if al == 0 else lfilter([1 - al], [1, -al], p, axis=0, zi=(al * p[:1]))[0]
            q = p + beta * (p - ema)
            q = np.clip(q[m], -2, 2)
            acc[a] += np.stack([w.sum(0), (w*y).sum(0), (w*q).sum(0), (w*y*y).sum(0), (w*q*q).sum(0), (w*y*q).sum(0)], -1)
for a, s in acc.items():
    n, sy, sp, syy, spp, syp = s.T
    r = (syp/n - sy*sp/n**2) / np.sqrt((syy/n - (sy/n)**2) * (spp/n - (sp/n)**2))
    print(f"alpha,beta {a} wp {r.mean():.6f}")
