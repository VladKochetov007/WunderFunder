"""Server-only checks for the matched rank-32 input routing pair."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from bilinear_entmax.receiver_support import (
    INPUT_BRANCH_KEYS,
    assert_close,
    build_native_probe_model,
    build_probe_model,
    canonical_wmse_backward,
    cpu_state_dict,
    synthetic_tensor,
)

_INPUT_VARIANTS = ("input_linear", "input_bilinear")
_OUTPUT_WEIGHT = "input_projection.bilinear_output.weight"
_U_WEIGHT = "input_projection.bilinear_u.weight"
_V_WEIGHT = "input_projection.bilinear_v.weight"


def _parameter(model: nn.Module, name: str) -> nn.Parameter:
    """Return one required trainable branch matrix by its stable state name."""
    parameters = dict(model.named_parameters())
    value = parameters.get(name)
    if value is None:
        raise AssertionError(f"input branch is missing parameter {name}")
    return value


def _parameter_count(model: nn.Module) -> int:
    """Count all trainable scalar weights in one probe model."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _check_route_formula(
    linear: nn.Module, bilinear: nn.Module, native: nn.Module, device: torch.device
) -> float:
    """Check the declared additive and product formulas on controlled normalized inputs."""
    for model in (linear, bilinear):
        with torch.no_grad():
            _parameter(model, _U_WEIGHT).zero_()
            _parameter(model, _V_WEIGHT).zero_()
            _parameter(model, _OUTPUT_WEIGHT).zero_()
            _parameter(model, _U_WEIGHT)[0, 0] = 1.0
            _parameter(model, _V_WEIGHT)[0, 0] = 1.0
            _parameter(model, _OUTPUT_WEIGHT)[0, 0] = 1.0

    normalized = torch.zeros((1, 1, 112), dtype=torch.float32, device=device)
    normalized[..., 0] = 2.0
    normalized[..., 52] = 3.0
    native_projection = native.input_projection(normalized)
    linear_projection = linear.input_projection(normalized)
    bilinear_projection = bilinear.input_projection(normalized)
    linear_delta = linear_projection - native_projection
    bilinear_delta = bilinear_projection - native_projection
    expected_linear = torch.zeros_like(linear_delta)
    expected_bilinear = torch.zeros_like(bilinear_delta)
    expected_linear[..., 0] = 5.0 / math.sqrt(2.0)
    expected_bilinear[..., 0] = 6.0
    linear_gap = assert_close(
        linear_delta, expected_linear, "rank-32 additive/sqrt2 formula", atol=1e-6
    )
    bilinear_gap = assert_close(
        bilinear_delta, expected_bilinear, "rank-32 product formula", atol=1e-6
    )
    difference = float((linear_delta - bilinear_delta).abs().max().item())
    if difference <= 1.0:
        raise AssertionError("input additive and product branches did not separate on the fixture")
    return max(linear_gap, bilinear_gap)


def _check_output_weight_learning(
    model: nn.Module,
    variant: str,
    sample: Tensor,
    target: Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float | int]:
    """Check zero U/V gradients at zero W and learning after a canonical update."""
    from wnn.training.sequence_optimizer import build_sequence_muon_optimizer
    from wnn.training.sequence_predict import predict_sequence

    report = canonical_wmse_backward(model, sample[None, :16], target, "bf16_bf16x9")
    initial_branch_gradients: dict[str, float] = {}
    for name in (_U_WEIGHT, _V_WEIGHT):
        gradient = _parameter(model, name).grad
        if gradient is None or not bool(torch.isfinite(gradient).all()):
            raise AssertionError(f"{variant} {name} has a missing or nonfinite initial gradient")
        initial_l1 = float(gradient.float().abs().sum().item())
        if initial_l1 != 0.0:
            raise AssertionError(f"{variant} {name} must have zero gradient while output W is zero")
        initial_branch_gradients[name] = initial_l1

    output_weight = _parameter(model, _OUTPUT_WEIGHT)
    gradient = output_weight.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise AssertionError(f"{variant} output W has a missing or nonfinite main-loss gradient")
    gradient_norm = float(gradient.float().norm().item())
    if gradient_norm <= 1e-8:
        raise AssertionError(f"{variant} output W has no useful main-loss gradient")

    output_before = output_weight.detach().clone()
    model.eval()
    prediction_before = torch.from_numpy(predict_sequence(model, sample, 16, device))
    normalized = (sample - model.mean) / model.std
    embedding_before = model.input_projection(normalized[None]).detach().clone()
    optimizer = build_sequence_muon_optimizer(model, config)
    optimizer.step()
    update_norm = float((output_weight.detach() - output_before).float().norm().item())
    if update_norm <= 1e-8:
        raise AssertionError(f"{variant} canonical optimizer did not update output W")
    embedding_after = model.input_projection(normalized[None]).detach()
    prediction_after = torch.from_numpy(predict_sequence(model, sample, 16, device))
    embedding_gap = float((embedding_after - embedding_before).abs().max().item())
    prediction_gap = float(
        (prediction_after.float() - prediction_before.float()).abs().max().item()
    )
    if embedding_gap <= 1e-6 or prediction_gap <= 1e-8:
        raise AssertionError(f"{variant} output W update did not change the input-model function")
    optimizer.zero_grad(set_to_none=True)
    updated_report = canonical_wmse_backward(model, sample[None, :16], target, "bf16_bf16x9")
    updated_branch_gradients: dict[str, float] = {}
    for name in (_U_WEIGHT, _V_WEIGHT):
        updated_gradient = _parameter(model, name).grad
        if updated_gradient is None or not bool(torch.isfinite(updated_gradient).all()):
            raise AssertionError(
                f"{variant} {name} has a missing or nonfinite post-update gradient"
            )
        updated_l1 = float(updated_gradient.float().abs().sum().item())
        if updated_l1 <= 1e-8:
            raise AssertionError(f"{variant} {name} did not learn after the output-W update")
        updated_branch_gradients[name] = updated_l1
    return {
        "main_loss": float(report["loss"]),
        "post_update_main_loss": float(updated_report["loss"]),
        "main_loss_gradient_tensors": int(report["nonzero_gradient_tensors"]),
        "output_weight_gradient_l2": gradient_norm,
        "output_weight_update_l2": update_norm,
        "input_embedding_max_abs_after_update": embedding_gap,
        "prediction_max_abs_after_update": prediction_gap,
        "u_v_gradient_l1_at_zero_W": initial_branch_gradients,
        "u_v_gradient_l1_after_optimizer_update": updated_branch_gradients,
    }


def check_input_routing(
    config: dict[str, Any],
    artifact: Mapping[str, Any],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Validate equal parameters, shared initialization, formula, and learnability."""
    models = {
        variant: build_probe_model(config, dict(artifact), variant, seed, device)
        for variant in _INPUT_VARIANTS
    }
    linear, bilinear = (models[variant] for variant in _INPUT_VARIANTS)
    native = build_native_probe_model(config, artifact, seed, device)
    states = {variant: cpu_state_dict(model) for variant, model in models.items()}
    native_state = cpu_state_dict(native)
    for variant, state in states.items():
        extras = set(state) - set(native_state)
        if extras != set(INPUT_BRANCH_KEYS):
            raise AssertionError(f"{variant} adapter tensor keys differ: {sorted(extras)}")
        for name, value in native_state.items():
            if name not in state or not torch.equal(state[name], value):
                raise AssertionError(f"{variant} changed canonical native tensor {name}")
        for name in INPUT_BRANCH_KEYS:
            if not torch.equal(state[name], states["input_bilinear"][name]):
                raise AssertionError(f"input variants do not share the initial tensor {name}")
    if _parameter_count(linear) != _parameter_count(bilinear):
        raise AssertionError("input_linear and input_bilinear parameter counts differ")
    for variant, state in states.items():
        if bool(state[_OUTPUT_WEIGHT].count_nonzero()):
            raise AssertionError(f"{variant} output W must start at zero")

    sample = synthetic_tensor((32, 112), device, seed + 71, scale=0.8)
    from wnn.training.sequence_predict import predict_sequence

    native.eval()
    native_output = torch.from_numpy(predict_sequence(native, sample, 16, device))
    native_gaps: dict[str, float] = {}
    for variant, model in models.items():
        model.eval()
        output = torch.from_numpy(predict_sequence(model, sample, 16, device))
        native_gaps[variant] = assert_close(
            output, native_output, f"{variant} zero-W native output", atol=1e-5
        )

    formula_gap = _check_route_formula(linear, bilinear, native, device)
    for variant, model in models.items():
        model.load_state_dict(states[variant], strict=True)
    target = synthetic_tensor((1, 16, 2), device, seed + 73, scale=1.2)
    learning = {
        variant: _check_output_weight_learning(
            models[variant], variant, sample, target, config, device
        )
        for variant in _INPUT_VARIANTS
    }
    return {
        "parameter_count": _parameter_count(linear),
        "equal_parameter_count": True,
        "shared_initial_branch_tensors": sorted(INPUT_BRANCH_KEYS),
        "zero_output_W": True,
        "native_initial_output_max_abs": native_gaps,
        "controlled_formula_max_abs": formula_gap,
        "output_W_learning": learning,
    }
