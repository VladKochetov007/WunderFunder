import json
from pathlib import Path
import numpy as np
S = Path("/tmp/claude-1000/-home-vlad-development-wunder-wnn-connectome-starterpack/53411689-5c90-4ea0-a054-137837af8779/scratchpad/flat")

def load(fold):
    d = dict(np.load(S / f"flat_f{fold}.npz"))
    d["members"] = json.loads((S / f"members_f{fold}.json").read_text())
    return d

def P(fold, stem):
    return np.load(S / f"p_f{fold}_{stem}.npy")

def wp(y, p, sel=None):
    """official metric on flat scored rows: mean over targets of |y|-weighted Pearson."""
    if sel is not None:
        y, p = y[sel], p[sel]
    out = []
    for k in range(2):
        w = np.abs(y[:, k].astype(np.float64)); yy = y[:, k].astype(np.float64); pp = np.clip(p[:, k].astype(np.float64), -2, 2)
        W = w.sum(); my = (w * yy).sum() / W; mp = (w * pp).sum() / W
        c = (w * yy * pp).sum() / W - my * mp
        vy = (w * yy * yy).sum() / W - my * my; vp = (w * pp * pp).sum() / W - mp * mp
        out.append(c / np.sqrt(vy * vp))
    return float(np.mean(out))

def halves(comp, seed=0):
    u = np.unique(comp); rng = np.random.default_rng(seed); rng.shuffle(u)
    a = set(u[: len(u) // 2].tolist())
    return np.array([c in a for c in comp])  # per sequence: True=half A
