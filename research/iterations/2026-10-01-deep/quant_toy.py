"""Post-diagnostic deterministic positive control; no real data or score access."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

directory = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("quant_geometry", directory/"quant_geometry.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rng = np.random.default_rng(20261002)
mix = np.linalg.qr(rng.normal(size=(11, 11)))[0]
rows = []
for sparsity in (1, 2):
    samples = []
    for _ in range(2):
        latent = np.zeros((6000, 11))
        index = rng.integers(0, 11, 6000)
        latent[np.arange(6000), index] = rng.normal(size=6000)
        if sparsity == 2:
            other = (index + rng.integers(1, 11, 6000)) % 11
            latent[np.arange(6000), other] = rng.normal(size=6000)
        samples.append(latent @ mix.T)
    train, test = samples
    scale, learned, pca = module.fit_lines(train, np.ones(6000)/6000)
    truth = mix.T/scale
    truth /= np.linalg.norm(truth, axis=1)[:, None]
    unit = test/scale
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    entry = {"true_updates_per_row": sparsity, "n_fit": 6000, "n_eval": 6000,
             "true_mixing_condition": float(np.linalg.cond(mix)), "models": {}}
    for name, atoms in (("true", truth), ("learned", learned), ("pca", pca), ("raw", np.eye(11))):
        one, two = module.errors(unit, atoms)
        entry["models"][name] = {"best1_coverage_01": float(np.mean(one <= .1)),
                                 "best2_coverage_01": float(np.mean(two <= .1)),
                                 "best2_median_error": float(np.median(two))}
    match = np.abs(truth @ learned.T)
    i, j = linear_sum_assignment(-match)
    entry["learned_true_atom_matches_ge095"] = int((match[i, j] >= .95).sum())
    assert entry["models"]["true"]["best2_coverage_01"] == 1.0
    rows.append(entry)
result = {"seed": 20261002, "status": "post_diagnostic_algorithm_positive_control",
          "selection": "Two deterministic planted cases fixed before toy outputs; no real data rerun",
          "cases": rows}
module.exclusive(directory/"quant-toy-result.json", result)
print(json.dumps(result, indent=2))
