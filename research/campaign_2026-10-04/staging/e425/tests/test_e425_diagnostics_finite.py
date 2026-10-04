"""Finite-aware E425 gradient accounting tests."""

from __future__ import annotations

import math

import torch
from wnn.specialist_diagnostics import SoftRouteDiagnostics

from wnn.specialist import SpecialistGRUModel


def test_nonfinite_expert_gradient_is_counted_without_poisoning_finite_mean() -> None:
    """Exclude only the nonfinite expert sample and preserve attempt/update facts."""
    torch.manual_seed(909)
    model = SpecialistGRUModel(
        n_in=3,
        hidden=8,
        expert_hidden=4,
        experts=4,
        layers=1,
        n_out=2,
        route="soft",
        prefix_rows=3,
    )
    values = torch.randn(2, 8, 3)
    prediction, _ = model(values)
    probabilities = model.last_route_probabilities
    assert probabilities is not None
    assert torch.isfinite(probabilities).all()
    prediction.square().sum().backward()

    injected_gradient = model.experts[0].weight_ih_l0.grad
    assert injected_gradient is not None
    injected_gradient[0, 0] = math.inf
    diagnostics = SoftRouteDiagnostics(model)
    snapshots = diagnostics.before_optimizer_step(model)
    assert torch.isinf(injected_gradient[0, 0])

    # Simulate GradScaler skipping this optimizer attempt after unscale detects
    # the nonfinite value. The trainer's EpochUpdateCounter owns the skip count.
    diagnostics.after_optimizer_step(model, snapshots)
    summary = diagnostics.finish()

    assert summary["optimizer_attempts"] == 1
    assert summary["expert_prediction_gradient_finite_steps"] == [0, 1, 1, 1]
    assert summary["expert_prediction_gradient_nonfinite_steps"] == [1, 0, 0, 0]
    assert summary["expert_prediction_gradient_nonzero_steps"] == [0, 1, 1, 1]
    gradient_means = summary["expert_prediction_gradient_l2_mean"]
    assert gradient_means[0] is None
    assert all(
        value is not None and math.isfinite(value) and value > 0 for value in gradient_means[1:]
    )
    assert summary["expert_parameter_nonzero_update_steps"] == [0, 0, 0, 0]
    assert summary["expert_parameter_update_l2_mean"] == [0.0, 0.0, 0.0, 0.0]
