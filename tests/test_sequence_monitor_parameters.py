from __future__ import annotations

import pytest
import torch
from torch import nn

from wnn.training.sequence_monitor_parameters import MAX_SAMPLED_PARAMETERS, SampledParameterProbe


def test_sampled_update_norm_is_bounded_and_does_not_consume_rng() -> None:
    """The observer measures actual parameter motion with a fixed scalar budget."""
    model = nn.Linear(128, 128)
    rng = torch.get_rng_state().clone()
    probe = SampledParameterProbe(model)
    with torch.no_grad():
        model.weight.add_(0.125)
    metrics = probe.metrics()
    assert metrics["optimizer/sampled_parameter_count"].item() == MAX_SAMPLED_PARAMETERS
    assert metrics["optimizer/sampled_update_l2_norm"].item() == pytest.approx(8)
    expected = model.weight.detach().reshape(-1)[:MAX_SAMPLED_PARAMETERS].norm().item()
    assert metrics["optimizer/sampled_weight_l2_norm"].item() == pytest.approx(expected)
    assert torch.equal(torch.get_rng_state(), rng)


def test_zero_weights_do_not_produce_nonfinite_ratios() -> None:
    """Zero and absent trainable weights remain finite or explicitly unobserved."""
    model = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    metrics = SampledParameterProbe(model).metrics()
    assert all(bool(torch.isfinite(value)) for value in metrics.values())
    model.weight.requires_grad_(False)
    assert SampledParameterProbe(model).metrics() == {}
