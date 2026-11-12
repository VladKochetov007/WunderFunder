"""Shared synthetic model fixtures for the Signal Routes receiver gates."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

VARIANTS = (
    "attention_additive",
    "attention_differential",
    "ffn_conditional",
    "ffn_static",
    "plain",
)
ATOL = 3e-4


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
    extension = _mapping(result.get("signal_routes"), "signal_routes")
    cell = _mapping(extension.get("cell"), "signal_routes.cell")
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
    from signal_routes.modeling import build_model

    probe = probe_config(config, variant, seed, compile_blocks=compile_blocks)
    extension = _mapping(probe["signal_routes"], "signal_routes")
    extension["preprocessing"] = dict(artifact)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(probe, device)
    return model


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


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Snapshot strict model tensors on CPU for matched-init and reload checks."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def assert_shared_initial_state(
    actual: Mapping[str, Tensor], reference: Mapping[str, Tensor], variant: str
) -> dict[str, int]:
    """Require canonical state bytes to match and permit only declared router tensors."""
    missing = set(reference) - set(actual)
    if missing:
        raise AssertionError(f"{variant} lacks canonical state keys: {sorted(missing)}")
    extras = set(actual) - set(reference)
    expected = (
        {
            f"blocks.{layer}.{name}"
            for layer in range(2)
            for name in ("routing_matrix", "routing_bias")
        }
        if variant.startswith("ffn_")
        else set()
    )
    if extras != expected:
        raise AssertionError(f"{variant} extra state keys differ: {sorted(extras)}")
    for name, value in reference.items():
        candidate = actual[name]
        if candidate.shape != value.shape or candidate.dtype != value.dtype:
            raise AssertionError(f"{variant} canonical state shape/dtype differs at {name}")
        if not torch.equal(candidate, value):
            raise AssertionError(f"{variant} canonical initial state differs at {name}")
    return {"canonical_tensor_count": len(reference), "router_tensor_count": len(extras)}


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
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if not gradients or any(
        gradient is not None and not bool(torch.isfinite(gradient).all()) for gradient in gradients
    ):
        raise AssertionError("synthetic WMSE produced missing/nonfinite model gradients")
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


def check_streaming_cache(
    model: nn.Module, sample: Tensor, chunk_rows: int, context_rows: int, mode: str
) -> dict[str, float | int]:
    """Compare explicit streaming to the canonical callback and validate rolling caches."""
    from wnn.training.sequence_attention import TransformerState
    from wnn.training.sequence_precision import forward_context
    from wnn.training.sequence_predict import predict_sequence

    model.eval()
    state: Any = None
    outputs: list[Tensor] = []
    observed_cache_rows = 0
    with torch.no_grad():
        for start in range(0, sample.shape[0], chunk_rows):
            stop = min(start + chunk_rows, sample.shape[0])
            with forward_context(mode, sample.device.type):
                prediction, hints, state = model(sample[None, start:stop], state)
            outputs.append(torch.cat((prediction.float(), hints.float()), dim=-1).squeeze(0))
            if not isinstance(state, TransformerState) or state.next_position != stop:
                raise AssertionError("streaming model returned an invalid causal cache state")
            expected_cache = min(stop, context_rows - 1)
            if len(state.keys) != 2 or len(state.values) != 2:
                raise AssertionError("probe transformer cache layer count differs from two")
            for key, value in zip(state.keys, state.values, strict=True):
                if key.shape != value.shape or key.shape[-2] != expected_cache:
                    raise AssertionError("streaming KV cache length differs from rolling context")
                if key.dtype != torch.float32 or value.dtype != torch.float32:
                    raise AssertionError("streaming KV cache must retain canonical FP32 state")
            observed_cache_rows = max(observed_cache_rows, expected_cache)
    streamed = torch.cat(outputs, dim=0).cpu()
    callback = torch.from_numpy(predict_sequence(model, sample, chunk_rows, sample.device))
    gap = assert_close(streamed, callback, "canonical streaming callback", atol=ATOL)
    return {
        "callback_max_abs": gap,
        "max_cache_rows": observed_cache_rows,
        "next_position": sample.shape[0],
    }
