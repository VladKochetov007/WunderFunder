"""Predict the value of extra TLead seeds from train-only s5/s6 full-valid predictions (exposed valid).

Fixed, pre-declared comparisons only: singles, equal TLead pair, TL3C-torch (.34 fbc/.33/.33), and the
measured pair gain versus the k-member law ensemble-mean ≈ c(1-1/k)(1-rho).
"""
import json, sys
from pathlib import Path
import numpy as np

TP = Path("/home/vlad/development/wunder/wnn_connectome_starterpack/research/campaign_2026-11-14/quality_r2/trajectory_mean_r1")
sys.path.insert(0, str(TP))
from grade_local import CACHE, predictions, score  # noqa: E402

with np.load(CACHE / "valid_targets.npz") as t:
    groups, mask, y = t["row_group"], t["mask"], t["y"]
y = y[mask]
names = ["q14t_tlead_s5", "q14t_tlead_s6", "q14t_fbc_s5"]
P = {n: predictions(n, groups, mask) for n in names}
S = {n: score(y, P[n])["wp"] for n in names}
pair = (P["q14t_tlead_s5"] + P["q14t_tlead_s6"]) / 2
S["tlead_pair"] = score(y, pair)["wp"]
S["tl3c_torch"] = score(y, .34 * P["q14t_fbc_s5"] + .33 * P["q14t_tlead_s5"] + .33 * P["q14t_tlead_s6"])["wp"]
rho = [float(np.corrcoef(P["q14t_tlead_s5"][:, j], P["q14t_tlead_s6"][:, j])[0, 1]) for j in range(2)]
gain2 = S["tlead_pair"] - (S["q14t_tlead_s5"] + S["q14t_tlead_s6"]) / 2
c = gain2 / (0.5 * (1 - np.mean(rho)))
pred = {k: float(c * (1 - 1 / k) * (1 - np.mean(rho))) for k in (2, 3, 4, 5)}
res = {"scores": S, "rho_t0_t1": rho, "pair_gain_vs_mean_single": gain2, "fitted_c": float(c),
       "predicted_gain_vs_mean_single_by_k": pred,
       "predicted_increment_k2_to_k3": pred[3] - pred[2], "predicted_increment_k3_to_k4": pred[4] - pred[3],
       "scope": "exposed full valid, train-only twins; transfer to all-data seeds assumed"}
print(json.dumps(res, indent=1)); Path(__file__).with_name("SEED_LAW_R1.json").write_text(json.dumps(res, indent=1))
