"""Per-class in-sample (train rows the teacher fitted) vs held-out (fold evaluation rows) WP, rows >= 99."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import load_campaign_split


def wp(y, p):
    r = []
    for t in range(2):
        yy = np.clip(y[..., t], -2, 2).astype(np.float64).ravel(); pp = np.clip(p[..., t], -2, 2).astype(np.float64).ravel()
        w = np.abs(yy); sw = w.sum(); my = (w * yy).sum() / sw; mp = (w * pp).sum() / sw
        r.append((w * (yy - my) * (pp - mp)).sum() / np.sqrt((w * (yy - my) ** 2).sum() * (w * (pp - mp) ** 2).sum()))
    return round(float(np.mean(r)), 4)


teacher, n = sys.argv[1], int(sys.argv[2])
tt = np.load(ROOT / "cache/train_targets.npz", mmap_mode="r"); pr = np.load(ROOT / f"cache/pred_{teacher}_train.npy", mmap_mode="r")
vt = np.load(ROOT / "cache/valid_targets.npz", mmap_mode="r"); vp = np.load(ROOT / f"cache/pred_{teacher}_valid.npy", mmap_mode="r")
split = load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{teacher.rsplit('_f', 1)[1]}.json", ROOT)
ev = np.isin(np.asarray(vt["row_group"]), np.array(sorted(split.evaluation_groups)))
out = {"teacher": teacher}
for c in range(4):
    ti = np.flatnonzero(np.asarray(tt["market_class"]) == c)[:n]; vi = np.flatnonzero(ev & (np.asarray(vt["market_class"]) == c))[:n]
    out[f"c{c}"] = {"train": wp(np.asarray(tt["y"][ti, 99:]), np.asarray(pr[ti, 99:])), "valid": wp(np.asarray(vt["y"][vi, 99:]), np.asarray(vp[vi, 99:])),
                    "n_train": int(len(ti)), "n_valid": int(len(vi))}
print(json.dumps(out))
