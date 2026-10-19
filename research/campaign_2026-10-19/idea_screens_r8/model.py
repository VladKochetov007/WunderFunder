"""Constant-output logvariance head with the unchanged R7 model topology."""

from __future__ import annotations

import torch
from idea_screens_r7.model_math import MeanVarianceMLP, initialize_pair, state_fingerprint
from torch import Tensor


class ConstantVarianceMLP(MeanVarianceMLP):
    """Retain both R7 heads while making logvariance depend only on its biases."""

    def forward_all(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Return ordinary mean predictions and row-constant logvariance outputs."""
        hidden = self._hidden(features)
        return self.mean_head(hidden), self.logvar_head(hidden * 0.0)


def initialize_constant_variance_model(
    expected_initial_fingerprint: str,
) -> tuple[ConstantVarianceMLP, dict[str, str]]:
    """Copy the exact seed-19031 R7 initialization into the constant subclass."""
    _, r7_candidate, initialization = initialize_pair()
    model = ConstantVarianceMLP()
    model.load_state_dict(r7_candidate.state_dict(), strict=True)
    actual = state_fingerprint(model)
    if actual != expected_initial_fingerprint or actual != initialization["initial"]:
        raise RuntimeError("constant-variance initialization differs from stored R7 seed 19031")
    if torch.count_nonzero(model.logvar_head.weight).item() != 0:
        raise RuntimeError("R7 variance weights must start at zero")
    return model, initialization


def constant_head_state(model: ConstantVarianceMLP) -> dict[str, object]:
    """Return scalar checks and the two fitted constant logvariance biases."""
    weight = model.logvar_head.weight.detach()
    bias = model.logvar_head.bias.detach()
    return {
        "weight_is_exactly_zero": bool(torch.count_nonzero(weight).item() == 0),
        "weight_max_abs": float(weight.abs().max().item()),
        "bias": [float(value) for value in bias.cpu().tolist()],
        "bias_min": float(bias.min().item()),
        "bias_max": float(bias.max().item()),
    }
