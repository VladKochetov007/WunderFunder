"""Same-distribution memorisation check: a fold-k teacher's WP on valid sequences it was fitted on (other folds'
evaluation groups) vs on fold k's evaluation groups it never saw. Scored mask, per class."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from wnn.splits import load_campaign_split


def wp(y, p, m):
    r = []
    for t in range(2):
        yy = np.clip(y[..., t][m], -2, 2).astype(np.float64); pp = np.clip(p[..., t][m], -2, 2).astype(np.float64)
        w = np.abs(yy); sw = w.sum(); my = (w * yy).sum() / sw; mp = (w * pp).sum() / sw
        r.append((w * (yy - my) * (pp - mp)).sum() / np.sqrt((w * (yy - my) ** 2).sum() * (w * (pp - mp) ** 2).sum()))
    return round(float(np.mean(r)), 4)


teacher = sys.argv[1]; k = int(teacher.rsplit("_f", 1)[1])
vt = np.load(ROOT / "cache/valid_targets.npz", mmap_mode="r"); vp = np.load(ROOT / f"cache/pred_{teacher}_valid.npy", mmap_mode="r")
rg = np.asarray(vt["row_group"]); cls = np.asarray(vt["market_class"])
sp = {f: load_campaign_split(ROOT / f"research/campaign_2026-09-29/artifacts/splits/fold{f}.json", ROOT) for f in range(4)}
held = np.isin(rg, np.array(sorted(sp[k].evaluation_groups)))
seen = np.isin(rg, np.array(sorted(set().union(*[sp[f].evaluation_groups for f in range(4) if f != k]))))
out = {"teacher": teacher}
for c in range(4):
    r = {}
    for name, sel in (("fitted", seen), ("held_out", held)):
        i = np.flatnonzero(sel & (cls == c))
        r[name] = wp(np.asarray(vt["y"][i]), np.asarray(vp[i]), np.asarray(vt["mask"][i]))
        r["n_" + name] = int(len(i))
    out[f"c{c}"] = r
print(json.dumps(out))
