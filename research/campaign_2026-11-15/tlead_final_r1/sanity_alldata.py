"""Fitted-valid sanity for all-data runs: score (in-sample, inflated) and correlations; detects broken fits only."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/vlad/development/wunder/wnn_connectome_starterpack/src")
from wnn.gmetric import global_wp  # noqa: E402
C = Path("/home/vlad/development/wunder/research_tmp/chrono_root/cache")
with np.load(C / "valid_targets.npz") as t:
    mask, y = t["mask"], t["y"]
names = sys.argv[1:]
P = {n: np.load(C / f"pred_{n}_valid.npy", mmap_mode="r")[mask] for n in names}
out = {}
for n in names:
    out[n] = {"fitted_valid_wp": global_wp(y[mask], P[n], np.ones(len(P[n]), bool))}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        out[f"corr {a}~{b}"] = float(np.mean([np.corrcoef(P[a][:, j], P[b][:, j])[0, 1] for j in range(2)]))
print(json.dumps(out, indent=1))
