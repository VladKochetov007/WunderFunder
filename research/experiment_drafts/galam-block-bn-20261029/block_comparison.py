from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import torch
from torch import nn

from wnn.galam.config import GALAMConfig
from wnn.galam.variants import GALAMAdaptation, build_galam_variant

ATOL = 2e-5
RTOL = 2e-4
GRAD_ATOL = 5e-5
GRAD_RTOL = 5e-4


def pair(config: GALAMConfig, device: torch.device) -> tuple[GALAMAdaptation, GALAMAdaptation]:
    """Build equal-weight step and block graphs with strict state-dict compatibility."""
    torch.manual_seed(1001)
    step = build_galam_variant(config, "schematic_gru_full", execution_backend="step")
    block = build_galam_variant(config, "schematic_gru_full", execution_backend="block")
    if not isinstance(step, GALAMAdaptation) or not isinstance(block, GALAMAdaptation):
        raise TypeError("Unexpected GALAM implementation")
    block.load_state_dict(step.state_dict(), strict=True)
    return step.to(device), block.to(device)


def compare(
    left: object, right: object, name: str, *, atol: float = ATOL, rtol: float = RTOL
) -> float:
    """Recursively compare all state fields, retaining the worst absolute error."""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            raise AssertionError(f"{name}: tensor type differs")
        if not left.is_floating_point():
            atol, rtol = 0.0, 0.0
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol, msg=name)
        if left.numel() == 0:
            return 0.0
        return float((left.detach().double() - right.detach().double()).abs().max().item())
    if is_dataclass(left) and not isinstance(left, type):
        if type(right) is not type(left):
            raise AssertionError(f"{name}: state type differs")
        return max(
            compare(getattr(left, field.name), getattr(right, field.name), f"{name}.{field.name}")
            for field in fields(left)
        )
    if isinstance(left, (tuple, list)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f"{name}: sequence structure differs")
        return max(
            (
                compare(a, b, f"{name}.{index}")
                for index, (a, b) in enumerate(zip(left, right, strict=True))
            ),
            default=0.0,
        )
    if isinstance(left, Mapping):
        if not isinstance(right, Mapping) or left.keys() != right.keys():
            raise AssertionError(f"{name}: mapping keys differ")
        return max(
            (compare(value, right[key], f"{name}.{key}") for key, value in left.items()),
            default=0.0,
        )
    if left != right:
        raise AssertionError(f"{name}: scalar fields differ")
    return 0.0


def compare_modules(step: nn.Module, block: nn.Module, *, gradients: bool) -> float:
    """Compare every named buffer or gradient, rejecting missing parameters."""
    left = dict(step.named_parameters()) if gradients else dict(step.named_buffers())
    right = dict(block.named_parameters()) if gradients else dict(block.named_buffers())
    if left.keys() != right.keys():
        raise AssertionError("Model key sets differ")
    worst = 0.0
    for name, value in left.items():
        other = right[name]
        if gradients:
            if value.grad is None or other.grad is None:
                raise AssertionError(f"Missing gradient for {name}")
            worst = max(
                worst,
                compare(value.grad, other.grad, name, atol=GRAD_ATOL, rtol=GRAD_RTOL),
            )
        else:
            worst = max(worst, compare(value, other, name))
    return worst
