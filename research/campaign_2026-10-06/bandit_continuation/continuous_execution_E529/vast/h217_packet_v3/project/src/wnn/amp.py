"""Explicit GradScaler construction for reproducible FP16 recipe variants."""

from __future__ import annotations

import math
from typing import Protocol

DEFAULT_AMP_INITIAL_SCALE = 65536.0


class _GradScalerFactory(Protocol):
    """Torch AMP factory interface used without importing Torch in tests."""

    def __call__(self, device: str, *, init_scale: float, enabled: bool) -> object: ...


class _AmpNamespace(Protocol):
    """Minimal Torch AMP namespace needed by the constructor."""

    GradScaler: _GradScalerFactory


class _TorchNamespace(Protocol):
    """Minimal Torch namespace needed by the constructor."""

    amp: _AmpNamespace


def create_grad_scaler(
    torch_module: _TorchNamespace, device_type: str, precision: str, initial_scale: float
) -> object:
    """Create a scaler with a validated and provenance-visible initial scale."""
    if precision not in {"fp16", "bf16", "fp32"}:
        raise ValueError(f"unsupported precision: {precision}")
    if not math.isfinite(initial_scale) or initial_scale <= 0:
        raise ValueError("AMP initial scale must be finite and positive")
    return torch_module.amp.GradScaler(
        device_type,
        init_scale=initial_scale,
        enabled=precision == "fp16",
    )
