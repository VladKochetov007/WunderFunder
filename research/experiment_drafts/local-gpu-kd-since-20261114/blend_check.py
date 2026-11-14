"""Zero-fit: full-valid WP and blend grid of the distilled student (Torch export) with the served TL3C ZIP predictions."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CACHE = Path("/home/vlad/development/wunder/research_tmp/chrono_root/cache")
sys.path.insert(0, str(ROOT / "src"))
from wnn.gmetric import global_wp  # noqa: E402

with np.load(CACHE / "valid_targets.npz") as z:
    y, mask, groups = z["y"].astype(np.float32), z["mask"], z["row_group"]


def torch_pred(run):
    side = json.loads((CACHE / f"pred_{run}_valid.json").read_text())
    rg = side["row_groups"]
    pos = {int(g): i for i, g in enumerate(json.loads(rg) if isinstance(rg, str) else rg)}
    arr = np.load(CACHE / f"pred_{run}_valid.npy", mmap_mode="r")
    return np.stack([np.asarray(arr[pos[int(g)]], np.float32) for g in groups])


def served(name):
    arr = np.load(ROOT / f"submissions/tl3c_pr49_20261029/{name}/predictions.npy", mmap_mode="r")
    return np.stack([np.asarray(arr[int(g)], np.float32) for g in groups])


ones = np.ones(int(mask.sum()), bool)
wp = lambda p: float(global_wp(y[mask], p[mask], ones)["wp"])  # noqa: E731
w = np.abs(np.clip(y[mask], -2, 2))


def rho(a, b):
    a, b = a[mask].astype(np.float64), b[mask].astype(np.float64)
    out = []
    for t in range(2):
        x, v, ww = a[:, t], b[:, t], w[:, t]
        x, v = x - np.average(x, weights=ww), v - np.average(v, weights=ww)
        out.append((ww * x * v).sum() / np.sqrt((ww * x * x).sum() * (ww * v * v).sum()))
    return float(np.mean(out))


kd = torch_pred(sys.argv[1])
blend, single = served("blend-int8-full-v3"), served("single-int8-full-v3")
print("kd", wp(kd), "blend", wp(blend), "tlead_s5", wp(single))
print("rho kd-blend", rho(kd, blend), "rho kd-tlead", rho(kd, single))
for c in (0.1, 0.2, 0.25, 0.3, 0.4, 0.5):
    print("blend*(1-c)+kd*c", c, round(wp((1 - c) * blend + c * kd), 6), "| tlead*(1-c)+kd*c", round(wp((1 - c) * single + c * kd), 6))
