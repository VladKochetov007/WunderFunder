"""Dependency map on exposed full valid (train-only twins), fixed grids, no selection for packaging.

1) TL3C weight curve: (1-a)*fbc + a*TLead pair.  2) Fourth-member weight b for strict KD90 added to TL3C.
3) Paired component bootstrap (2,000) of TL3C+.25 KD90 vs TL3C and per-target deltas.
"""
import json, sys
from pathlib import Path
import numpy as np
R = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
sys.path[:0] = [str(R / "src"), str(R / "research/campaign_2026-11-14/quality_r2/trajectory_mean_r1")]
from grade_local import CACHE, predictions, score  # noqa: E402
from wnn.gmetric import wp_sums, wp_from_sums  # noqa: E402

with np.load(CACHE / "valid_targets.npz") as t:
    groups, mask, y = t["row_group"], t["mask"], t["y"]
assert mask.ndim == 2 and mask.shape[0] == len(groups)
seq_of_row = np.repeat(np.arange(len(groups)), mask.sum(1))
y = y[mask]
man = json.loads((R / "research/campaign_2026-10-29/teacher_track_r1/chronological_primary.json").read_text())
comp_of_rg = {e["row_group"]: e["component"] for e in man["sequences"]["valid"]}
comp = np.array([comp_of_rg[int(g)] for g in groups]); ucomp, cidx = np.unique(comp, return_inverse=True)
P = {n: predictions(n, groups, mask) for n in ["q14t_fbc_s5", "q14t_tlead_s5", "q14t_tlead_s6", "q14s_local_kd90_s3"]}
pair = (P["q14t_tlead_s5"] + P["q14t_tlead_s6"]) / 2
wp = lambda p: score(y, p)["wp"]
out = {"weight_curve_tlead_share": {}, "kd90_fourth_weight": {}}
for a in (0, .2, .4, .5, .6, .66, .7, .8, .9, 1):
    out["weight_curve_tlead_share"][a] = wp((1 - a) * P["q14t_fbc_s5"] + a * pair)
tl3c = .34 * P["q14t_fbc_s5"] + .66 * pair
for b in (0, .1, .2, .25, .3, .4, .5):
    out["kd90_fourth_weight"][b] = wp((1 - b) * tl3c + b * P["q14s_local_kd90_s3"])
cand = .75 * tl3c + .25 * P["q14s_local_kd90_s3"]
def comp_sums(p):
    s = np.zeros((len(ucomp), 2, 6))
    yc, pc = np.clip(y, -2, 2).astype(np.float64), np.clip(p, -2, 2).astype(np.float64)
    w = np.abs(yc)
    cols = [w, w * yc, w * pc, w * yc * yc, w * pc * pc, w * yc * pc]
    rc = cidx[seq_of_row]
    for j in range(2):
        for k, c in enumerate(cols):
            s[:, j, k] = np.bincount(rc, weights=c[:, j], minlength=len(ucomp))
    return s
sb, sc = comp_sums(tl3c), comp_sums(cand)
full = lambda s: float(wp_from_sums(s.sum(0)).mean())
out["check_full_from_sums"] = {"tl3c": full(sb), "cand": full(sc)}
rng = np.random.default_rng(20261115); d = []
for _ in range(2000):
    i = rng.integers(0, len(ucomp), len(ucomp)); d.append(full(sc[i]) - full(sb[i]))
d = np.array(d)
out["kd90_quarter_bootstrap"] = {"delta": full(sc) - full(sb), "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
    "p_positive": float((d > 0).mean()), "components": int(len(ucomp))}
out["kd90_quarter_per_target_delta"] = (wp_from_sums(sc.sum(0)) - wp_from_sums(sb.sum(0))).tolist()
print(json.dumps(out, indent=1)); Path(__file__).with_name("DEPENDENCIES_R1.json").write_text(json.dumps(out, indent=1))
