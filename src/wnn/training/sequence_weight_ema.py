from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch


class SequenceWeightEMA:
    """Maintain per-update FP32 exponential averages of trainable parameters."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if isinstance(decay, bool) or not isinstance(decay, (int, float)):
            raise TypeError("EMA decay must be numeric")
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = self._parameters(self._unwrap(model))

    @staticmethod
    def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
        """Use stable parameter names across ordinary and DDP-wrapped models."""
        module = getattr(model, "module", None)
        return module if isinstance(module, torch.nn.Module) else model

    @staticmethod
    def _parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        parameters = {
            name: parameter.detach().clone()
            for name, parameter in SequenceWeightEMA._unwrap(model).named_parameters()
            if parameter.requires_grad
        }
        if not parameters:
            raise ValueError("EMA requires at least one trainable model parameter")
        for name, value in parameters.items():
            if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise TypeError(f"EMA requires finite FP32 trainable parameters: {name}")
        return parameters

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Apply one EMA update after a successful optimizer step."""
        current = {
            name: parameter
            for name, parameter in self._unwrap(model).named_parameters()
            if parameter.requires_grad
        }
        if set(current) != set(self.shadow):
            raise ValueError("model trainable parameter names changed after EMA initialization")
        for name, parameter in current.items():
            shadow = self.shadow[name]
            if parameter.dtype != torch.float32 or parameter.shape != shadow.shape:
                raise TypeError(f"EMA parameter shape or dtype changed: {name}")
            # finish_update has already synchronized finite model parameters across ranks.
            # With a convex coefficient, averaging two finite FP32 tensors stays finite.
            shadow.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
        self.updates += 1

    def state_dict(self) -> dict[str, Any]:
        """Return a resumable EMA state with independent tensor storage."""
        return {
            "schema": "wnn.sequence-weight-ema.v1",
            "decay": self.decay,
            "updates": self.updates,
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
        }

    def states_are_finite_fp32(self) -> bool:
        """Check the shadow once at a safe fit boundary, not on every GPU update."""
        return all(
            value.dtype == torch.float32 and bool(torch.isfinite(value).all())
            for value in self.shadow.values()
        )

    @torch.no_grad()
    def load_state_dict(self, state: object, model: torch.nn.Module, expected_updates: int) -> None:
        """Restore a safe-boundary EMA state and bind it to the loaded model shape."""
        if not isinstance(state, dict) or set(state) != {"schema", "decay", "updates", "shadow"}:
            raise ValueError("EMA resume state has an invalid shape")
        if state["schema"] != "wnn.sequence-weight-ema.v1" or state["decay"] != self.decay:
            raise ValueError("EMA resume state does not match the configured decay")
        if type(state["updates"]) is not int or state["updates"] != expected_updates:
            raise ValueError("EMA update count does not match the safe fit boundary")
        current = self._parameters(self._unwrap(model))
        shadow = state["shadow"]
        if not isinstance(shadow, dict) or set(shadow) != set(current):
            raise ValueError("EMA resume parameter names differ from the model")
        restored: dict[str, torch.Tensor] = {}
        for name, parameter in current.items():
            value = shadow[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.shape != parameter.shape
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"EMA resume tensor is invalid: {name}")
            restored[name] = value.to(device=parameter.device).clone()
        self.shadow = restored
        self.updates = expected_updates

    @contextmanager
    def apply(self, model: torch.nn.Module) -> Iterator[None]:
        """Temporarily swap EMA parameters into a model for read-only evaluation/export."""
        parameters = {
            name: parameter
            for name, parameter in self._unwrap(model).named_parameters()
            if parameter.requires_grad
        }
        if set(parameters) != set(self.shadow):
            raise ValueError("EMA evaluation parameter names differ from the model")
        backup = {name: value.detach().clone() for name, value in parameters.items()}
        try:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(self.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, parameter in parameters.items():
                    parameter.copy_(backup[name])

