"""Receiver-only causal and serialization probes for the PR81 residual modules."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

CHUNK_ATOL = 0.01
CHUNK_RTOL = 0.01


def _predict(model: nn.Module, values: Tensor, state: object = None) -> tuple[Tensor, object]:
    from wnn.training.sequence_precision import forward_context

    with forward_context("bf16_bf16x9", values.device.type):
        prediction, _, next_state = model(values, state)
    return prediction, next_state


def _max_abs(left: Tensor, right: Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _branch_parameters(model: nn.Module, variant: str) -> list[tuple[str, nn.Parameter]]:
    module = model.output_head if variant == "target_adapters" else model.input_projection
    return [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if not name.startswith("base.")
    ]


def _check_gradients(model: nn.Module, variant: str, values: Tensor) -> dict[str, Any]:
    named_parameters = _branch_parameters(model, variant)
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters:
        raise RuntimeError(f"{variant} has no separately trainable branch parameters")
    if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
        raise RuntimeError(f"{variant} has non-finite initialized branch parameters")
    if not any(torch.count_nonzero(parameter).item() > 0 for parameter in parameters):
        raise RuntimeError(f"{variant} has no nonzero initialized branch parameters")

    model.zero_grad(set_to_none=True)
    prediction, _ = _predict(model, values[:1, :8])
    prediction.float().sum().backward()
    missing_gradients = sum(parameter.grad is None for parameter in parameters)
    if missing_gradients:
        raise RuntimeError(f"{variant} has {missing_gradients} unused branch parameters")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError(f"{variant} has non-finite branch gradients")
    maximum_gradient = max(float(gradient.abs().max().item()) for gradient in gradients)
    if maximum_gradient == 0:
        raise RuntimeError(f"{variant} produced no nonzero branch gradient")
    model.zero_grad(set_to_none=True)
    return {
        "added_parameter_names": [name for name, _ in named_parameters],
        "parameter_values": sum(parameter.numel() for parameter in parameters),
        "max_abs_gradient": maximum_gradient,
    }


def _check_sequence_behavior(model: nn.Module, values: Tensor) -> dict[str, float | bool]:
    cut, split = 24, 13
    with torch.no_grad():
        full, _ = _predict(model, values)
        repeated, _ = _predict(model, values)
        if not torch.equal(full, repeated):
            raise RuntimeError("fresh-state prediction replay was not bitwise identical")

        changed = values.clone()
        changed[:, cut:] = values[:, cut:].flip(1) + 0.25
        altered, _ = _predict(model, changed)
        prefix_equal = torch.equal(full[:, :cut], altered[:, :cut])
        prefix_max = _max_abs(full[:, :cut], altered[:, :cut])
        if not prefix_equal:
            raise RuntimeError("a changed suffix altered an earlier prediction")

        first, state = _predict(model, values[:, :split])
        second, _ = _predict(model, values[:, split:], state)
        chunked = torch.cat((first, second), dim=1)
        chunk_max = _max_abs(full, chunked)
        if not torch.allclose(full.float(), chunked.float(), atol=CHUNK_ATOL, rtol=CHUNK_RTOL):
            raise RuntimeError(
                f"chunk-carry predictions exceeded BF16 tolerance: max_abs={chunk_max}"
            )
        if not bool(torch.isfinite(full).all() and torch.isfinite(chunked).all()):
            raise RuntimeError("sequence behavior check produced non-finite predictions")
    return {
        "suffix_prefix_max_abs": prefix_max,
        "chunk_carry_max_abs": chunk_max,
    }


def _check_state_roundtrip(
    model: nn.Module,
    config: dict[str, Any],
    builder: Callable[[dict[str, Any], torch.device], nn.Module],
    values: Tensor,
    device: torch.device,
) -> dict[str, str | int | float | bool]:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    payload = buffer.getvalue()
    buffer.seek(0)
    state = torch.load(buffer, map_location=device, weights_only=True)
    torch.manual_seed(204101)
    restored = builder(config, device).eval()
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        expected, _ = _predict(model, values)
        actual, _ = _predict(restored, values)
    max_abs = _max_abs(expected, actual)
    if not torch.equal(expected, actual):
        raise RuntimeError(f"state roundtrip changed predictions: max_abs={max_abs}")
    return {
        "serialized_bytes": len(payload),
        "state_sha256": hashlib.sha256(payload).hexdigest(),
        "prediction_max_abs": max_abs,
    }


def activate_branches(model: nn.Module, variant: str) -> dict[str, float]:
    """Set a fixed nonzero residual so causal checks cannot pass through a zero bypass."""
    if variant == "plain":
        return {"probe_output_weight": 0.0}
    layers = (
        list(model.output_head.up)
        if variant == "target_adapters"
        else [model.input_projection.output_projection]
    )
    with torch.no_grad():
        for layer in layers:
            layer.weight.fill_(0.01)
            layer.bias.zero_()
    return {"probe_output_weight": 0.01}
