"""Focused receiver probes for impact81 depth model gradients, caches, and reloads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

# Gradient and receipt summaries have heterogeneous JSON-compatible values.
GRAD_ATOL = 1e-5
GRAD_RTOL = 0.02


def _forward(
    model: nn.Module, values: Tensor, state: object = None
) -> tuple[Tensor, Tensor, object]:
    """Run one BF16/BF16x9 model call under the canonical precision context."""
    from wnn.training.sequence_precision import forward_context

    with forward_context("bf16_bf16x9", values.device.type):
        prediction, auxiliary, next_state = model(values, state)
    return prediction, auxiliary, next_state


def _check_auxiliary_gradients(model: nn.Module, probe: Tensor) -> dict[str, Any]:
    """Require each card04 head to receive finite, nonzero gradient from one loss."""
    model.train()
    model.zero_grad(set_to_none=True)
    _, heads, _ = _forward(model, probe)
    expected_shape = (probe.shape[0], probe.shape[1], 3, 2)
    if tuple(heads.shape) != expected_shape:
        raise RuntimeError(
            f"deep-supervision output must be {expected_shape}, got {tuple(heads.shape)}"
        )
    loss = heads.float().square().mean()
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("combined intermediate-head loss is non-finite")
    loss.backward()
    head_gradients: list[float] = []
    for index, head in enumerate(model.intermediate_heads):
        parameters = list(head.parameters())
        if not parameters or any(parameter.grad is None for parameter in parameters):
            raise RuntimeError(f"intermediate head {index} has a missing parameter gradient")
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError(f"intermediate head {index} has non-finite gradients")
        norm = torch.sqrt(sum(gradient.float().square().sum() for gradient in gradients))
        value = float(norm.item())
        if not value > 0.0:
            raise RuntimeError(f"intermediate head {index} has no nonzero gradient")
        head_gradients.append(value)
    model.zero_grad(set_to_none=True)
    model.eval()
    return {"loss": float(loss.detach().item()), "head_gradient_norms": head_gradients}


def _check_export_pruning(
    treatment: nn.Module,
    config: dict[str, Any],
    builder: Callable[[dict[str, Any], torch.device], nn.Module],
    probe: Tensor,
    device: torch.device,
) -> dict[str, bool | list[int]]:
    """Strict-load a training checkpoint, remove auxiliary heads, and compare serving output."""
    from impact81.depth_models import drop_auxiliary_heads

    checkpoint_state = {
        name: value.detach().clone() for name, value in treatment.state_dict().items()
    }
    exported = builder(config, device).eval()
    exported.load_state_dict(checkpoint_state, strict=True)
    before, _, _ = _forward(exported, probe)
    drop_auxiliary_heads(exported)
    after, hints, _ = _forward(exported, probe)
    if len(exported.intermediate_heads) != 0:
        raise RuntimeError("export pruning left intermediate heads registered")
    if tuple(hints.shape) != (probe.shape[0], probe.shape[1], 0):
        raise RuntimeError("pruned evaluation model must return the canonical empty hint tensor")
    if not torch.equal(before, after):
        raise RuntimeError("dropping auxiliary heads changed final predictions")
    return {"strict_load_passed": True, "heads_removed": True, "prediction_equal": True}


def _check_block_copies(control: nn.Module, treatment: nn.Module) -> dict[str, int | bool]:
    """Verify card05 copied values are equal and control parameter storage is independent."""
    if len(control.blocks) != 24 or len(treatment.blocks) != 12:
        raise RuntimeError("copied/tied depth must register 24 versus 12 unique blocks")
    checked = 0
    for index in range(12):
        first = dict(control.blocks[index].named_parameters())
        second = dict(control.blocks[index + 12].named_parameters())
        shared = dict(treatment.blocks[index].named_parameters())
        if first.keys() != second.keys() or first.keys() != shared.keys():
            raise RuntimeError(f"block parameter layouts differ for pair {index}")
        for name, source in first.items():
            copied = second[name]
            tied = shared[name]
            if source is copied or source.data_ptr() == copied.data_ptr():
                raise RuntimeError(f"copied control pair {index} unexpectedly shares {name}")
            if not torch.equal(source, copied) or not torch.equal(source, tied):
                raise RuntimeError(f"card05 initial block pair {index} differs at {name}")
            checked += source.numel()
    if tuple(treatment.logical_block_map) != tuple(range(12)) * 2:
        raise RuntimeError("tied model does not route 24 logical layers through twelve blocks")
    return {
        "copied_values_equal": True,
        "independent_control_parameters": True,
        "unique_block_parameter_values": checked,
    }


def _check_logical_caches(model: nn.Module, probe: Tensor) -> dict[str, int | bool]:
    """Require separate nonempty key/value storage for every logical layer."""
    _, _, state = _forward(model, probe[:1, :4])
    keys = getattr(state, "keys", None)
    values = getattr(state, "values", None)
    if not isinstance(keys, tuple) or not isinstance(values, tuple):
        raise RuntimeError("Transformer forward did not return a structured cache state")
    if len(keys) != 24 or len(values) != 24:
        raise RuntimeError("each 24-layer model must return 24 logical cache pairs")
    caches = (*keys, *values)
    if any(cache.shape[-2] <= 0 for cache in caches):
        raise RuntimeError("logical cache probe returned an empty cache")
    pointers = [cache.data_ptr() for cache in caches]
    if len(set(pointers)) != len(pointers):
        raise RuntimeError("logical cache pairs alias storage")
    return {
        "logical_cache_pairs": 24,
        "nonempty_cache_tensors": len(caches),
        "distinct_storage": True,
    }


def _check_optimizer_ownership(
    model: nn.Module, config: dict[str, Any], model_config: dict[str, object]
) -> dict[str, int | bool]:
    """Use canonical Muon and verify each unique tied parameter is owned once."""
    from wnn.training.sequence_fit_utils import make_optimizer

    config["model"] = model_config
    optimizer, scheduler = make_optimizer(model, config, 1)
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    expected_ids = {id(parameter) for parameter in expected}
    actual_ids = [id(parameter) for parameter in actual]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise RuntimeError("canonical optimizer does not own each tied parameter exactly once")
    if len(model.blocks) != 12:
        raise RuntimeError("canonical optimizer probe requires twelve registered tied blocks")
    names = [
        name
        for group in optimizer.param_groups
        for name in group.get("param_names", [])
        if name.endswith("qkv.weight")
    ]
    if len(names) != 12 or len(set(names)) != 12:
        raise RuntimeError("canonical Muon grouping did not recognize twelve unique qkv matrices")
    del optimizer, scheduler
    return {
        "trainable_parameters": len(expected_ids),
        "optimizer_owned_once": True,
        "unique_qkv_matrices": len(names),
    }


def _build_depth_optimizer_config() -> dict[str, Any]:
    """Load the pinned full optimizer recipe for the small canonical optimizer probe."""
    from impact81.bindings import PACKET
    from wnn.training.sequence_config import load_config

    config = load_config(PACKET / "sources/transformer_plain_control.json")
    if not isinstance(config, dict):
        raise TypeError("canonical optimizer fixture must load as a config mapping")
    return config


def _check_tied_gradient_sum(
    control: nn.Module,
    treatment: nn.Module,
    probe: Tensor,
    atol: float = GRAD_ATOL,
    rtol: float = GRAD_RTOL,
) -> dict[str, float | int]:
    """Compare each tied block gradient with the sum from its copied control pair."""
    control.train()
    treatment.train()
    control.zero_grad(set_to_none=True)
    treatment.zero_grad(set_to_none=True)
    control_prediction, _, _ = _forward(control, probe)
    tied_prediction, _, _ = _forward(treatment, probe)
    control_prediction.float().square().mean().backward()
    tied_prediction.float().square().mean().backward()
    maximum_error = 0.0
    compared = 0
    control_gradient_square = torch.zeros((), dtype=torch.float64, device=probe.device)
    tied_gradient_square = torch.zeros((), dtype=torch.float64, device=probe.device)
    for index in range(12):
        first = dict(control.blocks[index].named_parameters())
        second = dict(control.blocks[index + 12].named_parameters())
        shared = dict(treatment.blocks[index].named_parameters())
        for name, parameter in shared.items():
            left, right = first[name].grad, second[name].grad
            actual = parameter.grad
            if left is None or right is None or actual is None:
                raise RuntimeError(f"gradient comparison is missing block {index} parameter {name}")
            expected = left + right
            if not bool(torch.isfinite(expected).all() and torch.isfinite(actual).all()):
                raise RuntimeError(f"gradient comparison is non-finite at block {index} {name}")
            control_gradient_square += expected.double().square().sum()
            tied_gradient_square += actual.double().square().sum()
            error = float((actual.float() - expected.float()).abs().max().item())
            maximum_error = max(maximum_error, error)
            if not torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol):
                raise RuntimeError(
                    f"tied gradient differs from summed copied gradients at {index}.{name}: {error}"
                )
            compared += actual.numel()
    control_gradient_norm = float(torch.sqrt(control_gradient_square).item())
    tied_gradient_norm = float(torch.sqrt(tied_gradient_square).item())
    if not control_gradient_norm > 0.0 or not tied_gradient_norm > 0.0:
        raise RuntimeError("summed control and tied block gradients must both have positive norm")
    control.zero_grad(set_to_none=True)
    treatment.zero_grad(set_to_none=True)
    control.eval()
    treatment.eval()
    return {
        "compared_gradient_values": compared,
        "max_abs_error": maximum_error,
        "summed_control_gradient_norm": control_gradient_norm,
        "tied_gradient_norm": tied_gradient_norm,
    }
