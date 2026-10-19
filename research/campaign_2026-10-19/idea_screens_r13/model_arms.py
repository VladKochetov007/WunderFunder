"""Shared 336-input R7-shaped models and exact initial parity checks."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from idea_screens_r7.model_math import HIDDEN_WIDTH, MeanVarianceMLP, state_fingerprint
from torch import Tensor, nn
from torch.nn import functional as F

INPUT_WIDTH = 336
BASE_WIDTH = 224
SIDE_WIDTH = 112
ARM_NAMES = ("zero", "current_bypass", "warmup99")


class R13MeanVarianceMLP(MeanVarianceMLP):
    """R7 mean/variance heads over additive 224-base and 112-side projections."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(INPUT_WIDTH, HIDDEN_WIDTH)

    def _hidden(self, features: Tensor) -> Tensor:
        """Add separate base and side projections for exact zero-side parity."""
        if features.ndim != 2 or features.shape[1] != INPUT_WIDTH:
            raise ValueError("R13 model inputs must have shape (N,336)")
        values = features.float()
        base = F.linear(values[:, :BASE_WIDTH], self.trunk.weight[:, :BASE_WIDTH], self.trunk.bias)
        side = F.linear(values[:, BASE_WIDTH:], self.trunk.weight[:, BASE_WIDTH:], None)
        return F.gelu(base + side)


def initialize_models(seed: int = 19_061) -> tuple[dict[str, R13MeanVarianceMLP], dict[str, str]]:
    """Clone one seeded R7 core into three expanded arms with zero side weights."""
    torch.manual_seed(seed)
    core = MeanVarianceMLP()
    with torch.no_grad():
        core.logvar_head.weight.zero_()
        core.logvar_head.bias.zero_()
    models = {name: R13MeanVarianceMLP() for name in ARM_NAMES}
    for model in models.values():
        with torch.no_grad():
            model.trunk.weight[:, :BASE_WIDTH].copy_(core.trunk.weight)
            model.trunk.weight[:, BASE_WIDTH:].zero_()
            model.trunk.bias.copy_(core.trunk.bias)
            model.mean_head.load_state_dict(core.mean_head.state_dict())
            model.logvar_head.load_state_dict(core.logvar_head.state_dict())
    fingerprints = {name: state_fingerprint(model) for name, model in models.items()}
    if len(set(fingerprints.values())) != 1:
        raise RuntimeError("R13 model states are not byte-identical at initialization")
    return models, fingerprints


def assert_initial_prediction_parity(
    models: Mapping[str, R13MeanVarianceMLP],
    base: np.ndarray,
    side_blocks: Mapping[str, np.ndarray],
) -> dict[str, float | str]:
    """Require exact equality of initial predictions for the three side variants."""
    if set(models) != set(ARM_NAMES) or set(side_blocks) != set(ARM_NAMES):
        raise ValueError("R13 parity requires exactly the frozen three arms")
    base_values = np.asarray(base, dtype=np.float32)
    if base_values.ndim != 2 or base_values.shape[1] != BASE_WIDTH:
        raise ValueError("R13 base parity inputs must have shape (N,224)")
    outputs = []
    with torch.no_grad():
        for name in ARM_NAMES:
            side = np.asarray(side_blocks[name], dtype=np.float32)
            if side.shape != (len(base_values), SIDE_WIDTH):
                raise ValueError("R13 parity side block has the wrong shape")
            features = torch.from_numpy(
                np.ascontiguousarray(np.concatenate((base_values, side), axis=1))
            )
            prediction = models[name].forward_mean(features).cpu().numpy()
            if not np.isfinite(prediction).all():
                raise FloatingPointError("nonfinite R13 initial prediction")
            outputs.append(prediction)
    if not np.array_equal(outputs[0], outputs[1]) or not np.array_equal(outputs[0], outputs[2]):
        raise RuntimeError("zero-initialized R13 side weights changed initial predictions")
    return {
        "initial_prediction_max_abs_difference": 0.0,
        "initial_model_state_sha256": next(
            iter(set(state_fingerprint(model) for model in models.values()))
        ),
    }
