"""Critic: public-score predictive Monte Carlo for SP2, SP4, SP2+D and the folds-2/3 update table.

Inputs are numbers already on disk (cited in expectation_v1.md). Two routes for SP2's public delta over the 0.6622
incumbent are mixed 50/50 because neither bridge is validated for this mechanism:
  H  holdout route: public = b * holdout_delta + e, fitted on the 6 adjacent GRU package steps (expectation_bridge.out,
     "pair_no_tv/holdout_all"); e already contains holdout sampling noise and public paired noise.
  F  clean-fold route: public = tau * theta + e_LB, theta = public-like (pub143) local delta after empirical-Bayes
     shrinkage for post hoc selection; tau = transfer slope with wide uncertainty.

    .venv/bin/python research/campaign_2026-10-01/critic/expectation_mc.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
bridge = json.loads((HERE / "expectation_bridge.out").read_text())
fit = bridge["bridge_fits"]["pair_no_tv/holdout_all"]
steps = list(fit["residuals"])
x = np.array([bridge["pairs"][k]["holdout_all"] for k in steps])
y = np.array([bridge["pairs"][k]["public"] for k in steps])
jack = np.array([(x[m] * y[m]).sum() / (x[m] ** 2).sum() for m in (np.arange(len(x)) != i for i in range(len(x)))])
b_hat, b_sd, e_sd = fit["slope"], float(np.std(jack) * np.sqrt(len(x) - 1)), fit["resid_sd"]

N = 1_000_000
rng = np.random.default_rng(20261001)
INC, LB = 0.6622, 0.0006
PRIOR_MU, PRIOR_SD = 0.0003, 0.0006   # true effect of tonight's cheap ensemble/FT tweaks (expectation_v1.md section 3)
TAU_MU, TAU_SD = 0.75, 0.35           # clean-fold -> public transfer; S2 point 0.8 (0.63-0.92), history spans <0 to >2


def posterior(terms):
    """Normal prior + independent normal likelihood terms [(obs, sd, weight)] -> (mean, sd)."""
    prec = 1 / PRIOR_SD ** 2 + sum(w / s ** 2 for _, s, w in terms)
    mean = (PRIOR_MU / PRIOR_SD ** 2 + sum(w * o / s ** 2 for o, s, w in terms)) / prec
    return mean, prec ** -0.5


def summarise(s):
    return {"mean": round(float(s.mean()), 5), "p10": round(float(np.quantile(s, 0.1)), 5),
            "p90": round(float(np.quantile(s, 0.9)), 5), "P_gt_0.6622": round(float((s > 0.6622).mean()), 3),
            "P_gt_0.6641": round(float((s > 0.6641).mean()), 4), "P_ge_0.6660": round(float((s >= 0.6660).mean()), 5)}


def sp2_delta(f01_weight=1.0, f23=None):
    """Public delta of SP2 over the incumbent: 50/50 mixture of the holdout and clean-fold routes."""
    route_h = rng.normal(b_hat, b_sd, N) * 0.000379 + rng.normal(0, e_sd, N)
    terms = [(0.001326, 0.000714, f01_weight)]          # folds 0+1, pub143 (critic/sp2_pub143_f01_report.json)
    if f23 is not None:
        terms.append((f23[0], f23[1], 1.0))
    mu, sd = posterior(terms)
    route_f = rng.normal(TAU_MU, TAU_SD, N) * rng.normal(mu, sd, N) + rng.normal(0, LB, N)
    return np.where(rng.random(N) < 0.5, route_h, route_f), (mu, sd)


out = {"bridge": {"slope": b_hat, "slope_jackknife_sd": b_sd, "resid_sd": e_sd, "steps": steps}}
base = INC + rng.uniform(-0.00005, 0.00005, N)
d_sp2, post = sp2_delta()
out["SP2_now"] = {"theta_local_publike_posterior": post, **summarise(base + d_sp2)}

# SP4: two more same-recipe trunks with merged heads. Local increment from the ensemble-size law,
# gain(2->4) ~ 0.5 * gain(1->2); gain(1->2) = FT pair minus single, pub143 folds 0+1 = +0.00139 +- 0.00047.
g4 = rng.normal(0.5 * 0.00139, 0.00025, N)
d_sp4 = d_sp2 + rng.normal(0.8, 0.25, N) * g4 + rng.normal(0, 0.0003, N)
out["SP4"] = summarise(base + d_sp4)

# SP2+D: one distilled GRU256 trunk (merged head) added. Fold 0 only: +0.00020..0.00026 +- 0.00035 over the
# 4-member parent/FT mix (combo_f0_report.json), shrunk with the same prior.
mu_d, sd_d = posterior([(0.00025, 0.00035, 1.0)])
d_spd = d_sp2 + rng.normal(TAU_MU, TAU_SD, N) * rng.normal(mu_d, sd_d, N) + rng.normal(0, 0.0003, N)
out["SP2_plus_D"] = {"increment_posterior": (mu_d, sd_d), **summarise(base + d_spd)}
d_all = d_sp4 + rng.normal(TAU_MU, TAU_SD, N) * rng.normal(mu_d, sd_d, N)
out["SP4_plus_D_illustrative"] = summarise(base + d_all)

# Update table after the preregistered folds 2/3 confirmation (pooled pub143 delta, SD ~0.0007 expected).
table = {}
for o23 in (-0.0005, 0.0, 0.0005, 0.0010, 0.0015, 0.0020):
    for label, w in (("primary_f23_only", 0.0), ("sensitivity_f01_plus_f23", 1.0)):
        d, p = sp2_delta(w, (o23, 0.0007))
        table[f"{o23:+.4f}/{label}"] = {"theta": [round(v, 6) for v in p], **summarise(base + d)}
out["SP2_after_f23"] = table
print(json.dumps(out, indent=1))
