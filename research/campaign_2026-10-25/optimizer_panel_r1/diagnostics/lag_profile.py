"""Zero-fit: lag profile corr_w(y_t, p_{t+k}) for matched AdamW vs Muon F0 predictions (|y|-weighted)."""
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from wnn.gmetric import global_wp  # noqa: E402

z = np.load("cache/valid_targets.npz")
tpos = {int(g): i for i, g in enumerate(z["row_group"])}
split = json.load(open("research/campaign_2026-09-29/artifacts/splits/fold0.json"))
groups = [int(g) for g in split["evaluation"]["valid"]]
y = z["y"][[tpos[g] for g in groups]].astype(np.float32)
m = z["mask"][[tpos[g] for g in groups]].copy()
m[:, :99] = False
out = {}
for name in ("i1lead_f0_s1001_control_r1", "optim_f0_s1001_muon_r1"):
    side = json.load(open(f"cache/pred_{name}_valid.json"))
    pred = np.load(f"cache/pred_{name}_valid.npy", mmap_mode="r")
    pos = {int(g): i for i, g in enumerate(side["row_groups"])}
    p = np.stack([np.asarray(pred[pos[g]], np.float32) for g in groups])
    prof = {}
    for k in (-4, -2, -1, 0, 1, 2, 3, 4, 8, 16):
        mk = m.copy()
        if k > 0:
            mk[:, -k:] = False
            pk = np.concatenate([p[:, k:], p[:, -k:]], 1)
        elif k < 0:
            mk[:, :-k + 99] = False
            pk = np.concatenate([p[:, :-k], p[:, :k]], 1)
        else:
            pk = p
        prof[k] = float(global_wp(y[mk], pk[mk], np.ones(int(mk.sum()), bool))["wp"])
    out[name] = prof
    print(name, {k: round(v, 4) for k, v in prof.items()}, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
