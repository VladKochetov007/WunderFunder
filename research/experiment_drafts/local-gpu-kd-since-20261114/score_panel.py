"""Score a native replay (64-sequence valid panel) and compare it with the Torch FP32 valid export.

Usage: score_panel.py REPLAY_PREDICTIONS.npy TORCH_RUN [TORCH_RUN ...] (several runs = equal-weight mean).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CACHE = Path("/home/vlad/development/wunder/research_tmp/chrono_root/cache")
sys.path.insert(0, str(ROOT / "src"))
from wnn.gmetric import global_wp  # noqa: E402

plan = json.loads((ROOT / "research/campaign_2026-10-24/resume4h_r1/quota_submission_r1/REPLAY_PLAN_R1.json").read_text())
groups = [m["row_group"] for m in plan["members"]]
preds = np.load(sys.argv[1])
with np.load(CACHE / "valid_targets.npz") as z:
    pos = {int(g): i for i, g in enumerate(z["row_group"])}
    idx = [pos[g] for g in groups]
    y, mask = z["y"][idx].astype(np.float32), z["mask"][idx]
refs = []
for run in sys.argv[2:]:
    side = json.loads((CACHE / f"pred_{run}_valid.json").read_text())
    rg = side["row_groups"]
    rpos = {int(g): i for i, g in enumerate(json.loads(rg) if isinstance(rg, str) else rg)}
    arr = np.load(CACHE / f"pred_{run}_valid.npy", mmap_mode="r")
    refs.append(np.stack([np.asarray(arr[rpos[g]], np.float32) for g in groups]))
ref = np.mean(refs, axis=0)
ones = np.ones(int(mask.sum()), bool)
diff = (preds - ref)[mask]
print(json.dumps({
    "panel_sequences": len(groups), "scored_rows": int(mask.sum()),
    "native_wp": float(global_wp(y[mask], preds[mask], ones)["wp"]),
    "torch_wp": float(global_wp(y[mask], ref[mask], ones)["wp"]),
    "max_abs_diff": float(np.abs(diff).max()),
    "rms_diff_over_rms_pred": float(np.sqrt((diff**2).mean() / (ref[mask] ** 2).mean())),
}))
