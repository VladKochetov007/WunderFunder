"""Shared synthetic model fixtures for the Bilinear Entmax receiver gates."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

VARIANTS = ("attention_entmax", "attention_softmax", "input_bilinear", "input_linear")
ATOL = 3e-4
# Fixed before receiver execution: a 2% relative and 0.02 absolute bound allows
# roughly one BF16 output ULP near unit scale for explicit versus fused softmax.
BF16_PARITY_ATOL = 2e-2
BF16_PARITY_RTOL = 2e-2
INPUT_BRANCH_KEYS = frozenset(
    f"input_projection.bilinear_{name}.weight" for name in ("u", "v", "output")
)


def _mapping(value: object, name: str) -> dict[str, Any]:
    """Validate one JSON object at the receiver boundary."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def probe_config(
    config: dict[str, Any], variant: str, seed: int, *, compile_blocks: bool = False
) -> dict[str, Any]:
    """Create a bounded two-layer probe recipe while preserving the cell contract."""
    result = copy.deepcopy(config)
    model = _mapping(result.get("model"), "model")
    model.update(
        hidden=48,
        layers=2,
        heads=3,
        ff_multiplier=4,
        context_rows=32,
        dropout=0.0,
        checkpoint_blocks=False,
        hint_dim=0,
        attention_backend="efficient",
        compile_blocks=compile_blocks,
    )
    extension = _mapping(result.get("bilinear_entmax"), "bilinear_entmax")
    cell = _mapping(extension.get("cell"), "bilinear_entmax.cell")
    cell["variant"] = variant
    cell["seed"] = seed
    return result


def build_probe_model(
    config: dict[str, Any],
    artifact: Mapping[str, Any],
    variant: str,
    seed: int,
    device: torch.device,
    *,
    compile_blocks: bool = False,
) -> nn.Module:
    """Build one reproducible small variant on the receiver GPU."""
    from bilinear_entmax.modeling import build_model

    probe = probe_config(config, variant, seed, compile_blocks=compile_blocks)
    extension = _mapping(probe["bilinear_entmax"], "bilinear_entmax")
    extension["preprocessing"] = dict(artifact)
    caller_cpu_rng = torch.get_rng_state()
    caller_cuda_rng = torch.cuda.get_rng_state_all()
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = build_model(probe, device)
    finally:
        torch.set_rng_state(caller_cpu_rng)
        torch.cuda.set_rng_state_all(caller_cuda_rng)
    return model


def build_native_probe_model(
    config: dict[str, Any],
    artifact: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> nn.Module:
    """Build the pinned native Transformer as a same-seed receiver reference."""
    from wnn.training.sequence_models import build_sequence_model

    probe = probe_config(config, "attention_softmax", seed)
    model_config = _mapping(probe.get("model"), "model")
    caller_cpu_rng = torch.get_rng_state()
    caller_cuda_rng = torch.cuda.get_rng_state_all()
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = build_sequence_model(model_config)
    finally:
        torch.set_rng_state(caller_cpu_rng)
        torch.cuda.set_rng_state_all(caller_cuda_rng)
    mean = torch.as_tensor(artifact.get("mean"), dtype=torch.float32)
    std = torch.as_tensor(artifact.get("std"), dtype=torch.float32)
    with torch.no_grad():
        model.mean.copy_(mean)
        model.std.copy_(std)
    return model.to(device=device, dtype=torch.float32)


def synthetic_tensor(
    shape: tuple[int, ...], device: torch.device, seed: int, scale: float = 1.0
) -> Tensor:
    """Create deterministic FP32 synthetic values without consuming global RNG."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
    return values.to(device=device)


def assert_close(actual: Tensor, expected: Tensor, label: str, atol: float = ATOL) -> float:
    """Require finite tensors within tolerance and return their max absolute gap."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label} shape differs: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError(f"{label} contains nonfinite values")
    gap = float((actual.float() - expected.float()).abs().max().item()) if actual.numel() else 0.0
    if gap > atol:
        raise AssertionError(f"{label} gap {gap:.8g} exceeds {atol:.8g}")
    return gap


def assert_dtype_aware_close(
    actual: Tensor,
    expected: Tensor,
    label: str,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float]:
    """Check an explicit absolute/relative bound and return both error scales."""
    if actual.shape != expected.shape:
        raise AssertionError(
            f"{label} shape differs: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    left, right = actual.float(), expected.float()
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        raise AssertionError(f"{label} contains nonfinite values")
    difference = (left - right).abs()
    maximum_absolute = float(difference.max().item()) if difference.numel() else 0.0
    maximum_relative = (
        float((difference / right.abs().clamp_min(1e-6)).max().item())
        if difference.numel()
        else 0.0
    )
    if not torch.allclose(left, right, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{label} abs gap {maximum_absolute:.8g}, rel gap {maximum_relative:.8g} "
            f"exceeds atol={atol:.8g}, rtol={rtol:.8g}"
        )
    return {"max_abs": maximum_absolute, "max_relative": maximum_relative}


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Snapshot strict model tensors on CPU for matched-init and reload checks."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def assert_shared_initial_state(
    actual: Mapping[str, Tensor], reference: Mapping[str, Tensor], variant: str
) -> dict[str, int]:
    """Require native tensors to match and permit only the declared input branch."""
    missing = set(reference) - set(actual)
    if missing:
        raise AssertionError(f"{variant} lacks canonical state keys: {sorted(missing)}")
    extras = set(actual) - set(reference)
    expected = set(INPUT_BRANCH_KEYS) if variant in ("input_linear", "input_bilinear") else set()
    if extras != expected:
        raise AssertionError(f"{variant} extra state keys differ: {sorted(extras)}")
    for name, value in reference.items():
        candidate = actual[name]
        if candidate.shape != value.shape or candidate.dtype != value.dtype:
            raise AssertionError(f"{variant} canonical state shape/dtype differs at {name}")
        if not torch.equal(candidate, value):
            raise AssertionError(f"{variant} canonical initial state differs at {name}")
    return {"canonical_tensor_count": len(reference), "input_branch_tensor_count": len(extras)}


def canonical_wmse_backward(
    model: nn.Module, values: Tensor, target: Tensor, mode: str
) -> dict[str, float | int]:
    """Backpropagate one synthetic update through the unchanged clipped weighted loss."""
    from wnn.training.sequence_objectives import (
        objective_numerators,
        scaled_objective,
        window_denominators,
    )
    from wnn.training.sequence_precision import forward_context

    mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
    teacher = torch.zeros_like(target)
    hint_target = values.new_empty((*values.shape[:2], 0))
    model.train()
    with forward_context(mode, values.device.type):
        prediction, hints, _ = model(values, None)
    numerators = objective_numerators(
        prediction.float(), hints.float(), target, teacher, hint_target, mask
    )
    denominators = window_denominators(target, mask, hint_target)
    loss = scaled_objective(numerators, denominators, 1, 0.0, 0.0)
    if not bool(torch.isfinite(loss.detach())):
        raise AssertionError("canonical synthetic WMSE is nonfinite")
    loss.backward()
    named_gradients = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_gradients:
        raise AssertionError("synthetic WMSE found no trainable model parameters")
    missing = [name for name, gradient in named_gradients if gradient is None]
    if missing:
        raise AssertionError(f"synthetic WMSE left trainable parameters disconnected: {missing}")
    gradients = [gradient for _, gradient in named_gradients]
    if any(
        not bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None
    ):
        raise AssertionError("synthetic WMSE produced nonfinite model gradients")
    nonzero = [
        gradient for gradient in gradients if gradient is not None and bool((gradient != 0).any())
    ]
    if not nonzero:
        raise AssertionError("synthetic WMSE produced no nonzero model gradient")
    return {
        "loss": float(loss.detach().item()),
        "parameters_with_gradients": sum(gradient is not None for gradient in gradients),
        "nonzero_gradient_tensors": len(nonzero),
    }
