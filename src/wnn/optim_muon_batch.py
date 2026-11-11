from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Protocol, cast

import torch
from torch import Tensor


class Orthogonalizer(Protocol):
    """Canonical per-trailing-matrix Newton–Schulz implementation."""

    def __call__(
        self, values: Tensor, steps: int = 5, eps: float = 1e-7, *, strict_fp32: bool = False
    ) -> Tensor:
        """Orthogonalize each trailing matrix independently."""
        ...


@torch.no_grad()
def batched_muon_group_step(
    group: dict[str, object],
    state: MutableMapping[Tensor, dict[str, object]],
    orthogonalize: Orthogonalizer,
) -> None:
    """Batch canonical independent Muon matrices by shape without changing their state schema.

    Momentum, Nesterov interpolation, per-block normalization, five-step polynomial,
    weight decay and RMS scaling retain the original definitions. GEMM packing
    changes numerical execution and needs actual-device parity and timing evidence.
    """
    parameters = cast(list[Tensor], group["params"])
    beta = float(group["betas"][0])
    blocks = int(group["blocks"])
    strict = bool(group.get("strict_fp32", False))
    buckets: dict[
        tuple[tuple[int, ...], torch.dtype, torch.device], list[tuple[Tensor, Tensor]]
    ] = defaultdict(list)
    for parameter in parameters:
        gradient = parameter.grad
        entry = state.get(parameter)
        if strict:
            if parameter.dtype != torch.float32 or (
                gradient is not None and gradient.dtype != torch.float32
            ):
                raise TypeError("strict-FP32 batched Muon requires FP32 parameters and gradients")
            if any(
                isinstance(value, Tensor)
                and value.is_floating_point()
                and value.dtype != torch.float32
                for value in (() if entry is None else entry.values())
            ):
                raise TypeError("strict-FP32 batched Muon requires FP32 optimizer state")
        if gradient is None:
            continue
        if entry is None:
            entry = state.setdefault(parameter, {})
        if not entry:
            entry["momentum_buffer"] = torch.zeros_like(parameter)
        buffer = entry["momentum_buffer"]
        if not isinstance(buffer, Tensor):
            raise TypeError("Muon momentum buffer must be a tensor")
        buffer.lerp_(gradient, 1 - beta)
        rows, columns = parameter.shape[0] // blocks, parameter.shape[1]
        momentum = gradient.lerp(buffer, beta).reshape(blocks, rows, columns)
        key = (tuple(parameter.shape), parameter.dtype, parameter.device)
        buckets[key].append((parameter, momentum))
    for entries in buckets.values():
        # Leading dimensions are only independent batch axes. The canonical
        # orthogonalizer reduces norms over (-2,-1), never over this new axis.
        momentum = torch.stack([value for _, value in entries])
        update = orthogonalize(
            momentum,
            int(group["ns_steps"]),
            float(group.get("ns_eps", 1e-7)) if strict else 1e-7,
            strict_fp32=strict,
        )
        rows, columns = momentum.shape[-2:]
        rate = float(group["lr"])
        scale = -rate * float(group["rms"]) * math.sqrt(max(rows, columns))
        for (parameter, _), delta in zip(entries, update.unbind(0), strict=True):
            parameter.mul_(1 - rate * float(group["weight_decay"]))
            parameter.add_(delta.reshape(parameter.shape), alpha=scale)
