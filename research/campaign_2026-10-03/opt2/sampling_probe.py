from __future__ import annotations

import json

import numpy as np
import torch
from sampling import EpochOrderSampler, init_qmc_, stratified_order
from torch import nn


def main() -> None:
    """Measure positive/negative variance controls; these are not forecasting scores."""
    strata = np.repeat(np.arange(4), 16)
    signal = 10 * strata.astype(float)
    residual = np.random.default_rng(39).normal(size=64)
    measures: dict[str, list[float]] = {
        "random_signal": [],
        "stratified_signal": [],
        "random_residual": [],
        "stratified_residual": [],
    }
    for seed in range(32):
        random = np.random.default_rng(seed).permutation(64)
        stratified = stratified_order(strata, 8, seed)
        for kind, values in (("signal", signal), ("residual", residual)):
            for name, order in (("random", random), ("stratified", stratified)):
                measures[f"{name}_{kind}"].append(float(values[order].reshape(-1, 8).mean(1).var()))
    control = EpochOrderSampler(64, 3)
    reverse = EpochOrderSampler(64, 3, True)
    weights = np.linspace(-1, 1, 64)
    centered = signal - signal.mean()
    linear_control = float(weights @ centered[list(control)])
    linear_reverse = float(weights @ centered[list(reverse)])
    torch.manual_seed(11)
    model = nn.Linear(32, 32, bias=False)
    original_sv = torch.linalg.svdvals(model.weight.detach())
    init_qmc_(model, 11)
    qmc_sv = torch.linalg.svdvals(model.weight.detach())
    out = {
        "data": "64 synthetic sequences, 4 equal input strata; batch8; 32 seeded permutations",
        "mean_batch_mean_variance": {
            name: float(np.mean(values)) for name, values in measures.items()
        },
        "antithetic_linear_order_control": {
            "forward": linear_control,
            "reverse": linear_reverse,
            "pair_mean": (linear_control + linear_reverse) / 2,
            "limitation": "exact cancellation in linear artificial order statistic; nonlinear SGD need not be negatively correlated",
        },
        "initialization_spectrum_32x32": {
            "iid_min_max": [float(original_sv.min()), float(original_sv.max())],
            "qmc_min_max": [float(qmc_sv.min()), float(qmc_sv.max())],
            "limitation": "uniform scalar marginals do not imply iid spectrum or trainability",
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
