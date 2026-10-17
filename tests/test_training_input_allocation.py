from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import pytest
import torch

from wnn.data import N_FEATURES
from wnn.features import event_features_torch, online_zscore_torch
from wnn.privileged import privileged_inputs
from wnn.research_features import research_features_torch
from wnn.training.environment import InputConfiguration, TrainingEnvironment
from wnn.training.inputs import build_inputs


class InputBuilder(Protocol):
    """A feature assembler accepting an explicit training environment."""

    def __call__(self, x: torch.Tensor, *, environment: TrainingEnvironment) -> torch.Tensor:
        """Assemble model inputs for one batch."""
        ...


def _staged_reference(x: torch.Tensor, *, environment: TrainingEnvironment) -> torch.Tensor:
    """Retain the former staged concatenation as a small behavioral oracle."""
    raw, extra = x[..., :N_FEATURES], x[..., N_FEATURES:]
    event = event_features_torch(raw)
    if environment.inputs.raw_only:
        event = torch.zeros_like(event)
    inputs = torch.cat([raw, event], dim=-1)
    if environment.inputs.online_norm:
        inputs = torch.cat([inputs, online_zscore_torch(raw)], dim=-1)
    propensity_inputs = inputs
    if environment.inputs.research_features != "none":
        inputs = torch.cat(
            [
                inputs,
                research_features_torch(
                    raw,
                    environment.inputs.research_features,
                    environment.inputs.research_feature_prior,
                    control=environment.inputs.research_feature_control,
                ),
            ],
            dim=-1,
        )
    if environment.inputs.privileged_mode != "none":
        inputs = torch.cat(
            [
                inputs,
                privileged_inputs(
                    raw,
                    environment.inputs.privileged_mode,
                    environment.inputs.privileged_horizon,
                ),
            ],
            dim=-1,
        )
    if extra.shape[-1]:
        inputs = torch.cat([inputs, extra], dim=-1)
        if environment.inputs.propensity_model is not None:
            propensity_inputs = torch.cat([propensity_inputs, extra], dim=-1)
    if environment.inputs.propensity_model is not None:
        inputs = torch.cat(
            [
                inputs,
                environment.propensity(environment.inputs.propensity_model, propensity_inputs)[
                    ..., None
                ].to(inputs.dtype),
            ],
            dim=-1,
        )
    if environment.inputs.sentinel_preprocessor is not None:
        inputs = environment.inputs.sentinel_preprocessor.torch(inputs)
    return inputs


def _environment() -> TrainingEnvironment:
    """Build a CPU-only configuration that exercises every generic feature block."""

    def propensity(_model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
        """Produce deterministic weights while retaining a differentiable input path."""
        return torch.sigmoid(features[..., 0] + features[..., -1])

    return TrainingEnvironment(
        inputs=InputConfiguration(
            propensity_model=torch.nn.Identity(),
            online_norm=True,
            research_features="volume28",
            privileged_mode="current",
        ),
        propensity=propensity,
    )


def _outputs_and_input_gradient(
    builder: InputBuilder,
    values: torch.Tensor,
    environment: TrainingEnvironment,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a deterministic scalar objective and retain its input gradient."""
    input_values = values.detach().clone().requires_grad_(True)
    output = builder(input_values, environment=environment)
    weights = torch.arange(output.numel(), dtype=output.dtype).reshape_as(output) / output.numel()
    (output * weights).sum().backward()
    assert input_values.grad is not None
    return output.detach(), input_values.grad.detach()


def _model_parameter_gradient(
    builder: InputBuilder,
    values: torch.Tensor,
    environment: TrainingEnvironment,
    initial_weight: torch.Tensor,
) -> torch.Tensor:
    """Measure a downstream trainable layer's gradient on ordinary data inputs."""
    features = builder(values, environment=environment)
    weight = torch.nn.Parameter(initial_weight.clone())
    predictions = features @ weight
    (predictions.square() + 0.1 * predictions).mean().backward()
    assert weight.grad is not None
    return weight.grad.detach()


def test_generic_input_assembly_matches_staged_values_gradients_and_reduces_cats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep feature values and gradients while removing intermediate full-width joins."""
    generator = torch.Generator().manual_seed(20261017)
    values = torch.randn(2, 37, N_FEATURES + 3, generator=generator)
    environment = _environment()

    actual, actual_gradient = _outputs_and_input_gradient(build_inputs, values, environment)
    expected, expected_gradient = _outputs_and_input_gradient(
        _staged_reference, values, environment
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    initial_weight = torch.randn(expected.shape[-1], 3, generator=generator)
    actual_parameter_gradient = _model_parameter_gradient(
        build_inputs, values, environment, initial_weight
    )
    expected_parameter_gradient = _model_parameter_gradient(
        _staged_reference, values, environment, initial_weight
    )
    torch.testing.assert_close(
        actual_parameter_gradient, expected_parameter_gradient, rtol=0, atol=0
    )

    original_cat = torch.cat
    cat_count = [0]

    # Preserve torch.cat's overloaded argument form in the allocation counter.
    def counted_cat(tensors: Sequence[torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:
        cat_count[0] += 1
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", counted_cat)
    build_inputs(values, environment=environment)
    optimized_cat_count = cat_count[0]
    cat_count[0] = 0
    _staged_reference(values, environment=environment)
    staged_cat_count = cat_count[0]

    assert optimized_cat_count < staged_cat_count
