"""Fixed 4-member screen on exposed full valid: .25 each of {fbc, tlead_s5, tlead_s6, X} vs TL3C .34/.33/.33.

X ranges over every train-only H256x2 endpoint with a full-valid cache; all candidates are reported (selection exposure = len(X)).
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-11-14/quality_r2/trajectory_mean_r1")
from grade_local import CACHE, predictions, score  # noqa: E402

with np.load(CACHE / "valid_targets.npz") as t:
    groups, mask, y = t["row_group"], t["mask"], t["y"]
y = y[mask]
base = {n: predictions(n, groups, mask) for n in ["q14t_fbc_s5", "q14t_tlead_s5", "q14t_tlead_s6"]}
tl3c = .34 * base["q14t_fbc_s5"] + .33 * base["q14t_tlead_s5"] + .33 * base["q14t_tlead_s6"]
ref = score(y, tl3c)["wp"]
X = ["tt29rep_fbc1k_s5", "tt29rep_fin256_s3", "tt29rep_fin256_s4", "l14_kd50_s3", "l14_kd50_s4",
     "l14_kd90_s3", "q14s_local_kd90_s3", "q14s_local_kd100_s3"]
out = {"tl3c_torch": ref, "candidates": {}}
for x in X:
    try:
        p = predictions(x, groups, mask)
    except Exception as e:
        out["candidates"][x] = {"error": repr(e)[:200]}; continue
    four = .25 * (base["q14t_fbc_s5"] + base["q14t_tlead_s5"] + base["q14t_tlead_s6"] + p)
    s = score(y, p)["wp"]; f = score(y, four)["wp"]
    rho = float(np.mean([np.corrcoef(p[:, j], tl3c[:, j])[0, 1] for j in range(2)]))
    out["candidates"][x] = {"single": s, "four_member": f, "delta_vs_tl3c": f - ref, "corr_with_tl3c": rho}
    print(x, out["candidates"][x], flush=True)
Path(__file__).with_name("FOURTH_MEMBER_R1.json").write_text(json.dumps(out, indent=1))
