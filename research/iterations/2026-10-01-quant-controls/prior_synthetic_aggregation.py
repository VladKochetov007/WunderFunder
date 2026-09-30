"""Synthetic-only supplement; no feature reads or change to frozen PRIOR02 code."""
import importlib.util
from pathlib import Path

import numpy as np

path = Path(__file__).with_name("prior_control.py")
spec = importlib.util.spec_from_file_location("frozen_prior", path)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)

fake, expected_gain = [], []
for i in range(256):
    count = np.array([20+i % 23, 16, 4+i % 23])
    gain = .2+i*.001
    expected_gain.append(gain+.03)
    conditions = {"original": {"counts": count.tolist(),
        "loss_sums": (count[:, None]*np.array([1.1, .9, .7, .8, .85])).tolist()}}
    for j, seed in enumerate(prior.SEEDS):
        conditions[f"block128_{seed}"] = {"counts": count.tolist(),
            "loss_sums": (count[:, None]*np.array([1., 1.2+gain+j*.03, 1.2])).tolist()}
    fake.append({"half": i % 2, "class": i//64, "conditions": conditions})
report = prior.summarize(fake)["pooled"]["slices"]
for ci, cut in enumerate(prior.CUTS):
    contrast = report[cut]["contrasts"]["block128_mean_old_minus_matched"]
    count = np.array([f["conditions"]["original"]["counts"][ci] for f in fake])
    prior.require_close(contrast["macro_chain"]["point"], np.mean(expected_gain))
    prior.require_close(contrast["event_weighted"]["point"], np.dot(count, expected_gain)/count.sum())
    prior.require_close(report[cut]["contrasts"]["original_F_minus_T_original"]["macro_chain"]["ci95"], [.2, .2])
print("PASS: artificial 256-chain paired seed averaging, macro/event ratios and constant-effect bootstrap intervals")
