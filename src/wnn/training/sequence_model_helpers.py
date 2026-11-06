"""Small shared hint and streaming-state helpers for sequence adapters."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wnn.training.sequence_attention import TransformerState

SequenceState = Tensor | TransformerState | None
_MODEL_CONFIG_FIELDS = {
    "architecture",
    "hidden",
    "layers",
    "heads",
    "ff_multiplier",
    "ff_activation",
    "ff_hidden",
    "context_rows",
    "dropout",
    "checkpoint_blocks",
    "hint_dim",
    "hint_projection",
    "precision_mode",
    "attention_backend",
    "compile_blocks",
}


def validate_model_config(config: Mapping[str, object]) -> None:
    """Reject unknown direct model-builder options before constructing modules."""
    unknown = set(config) - _MODEL_CONFIG_FIELDS
    if unknown:
        raise ValueError(f"unknown sequence model config fields: {sorted(unknown)}")


def transformer_feedforward(
    config: Mapping[str, object], hidden: int, multiplier: float
) -> tuple[str, int]:
    """Resolve GELU/SwiGLU width without changing the legacy GELU expansion."""
    activation = config.get("ff_activation", "gelu")
    if not isinstance(activation, str):
        raise TypeError("model config 'ff_activation' must be a string")
    if activation not in ("gelu", "swiglu"):
        raise ValueError("model config 'ff_activation' must be 'gelu' or 'swiglu'")
    if "ff_hidden" in config:
        width = config["ff_hidden"]
        if type(width) is not int:
            raise TypeError("model config 'ff_hidden' must be an integer")
    elif activation == "swiglu":
        width = int(2 * hidden * multiplier / 3)
    else:
        width = int(hidden * multiplier)
    if not isinstance(width, int) or width <= 0:
        raise ValueError("Transformer feed-forward width must be a positive integer")
    return activation, width


def integer_option(config: Mapping[str, object], name: str, default: int | None = None) -> int:
    """Read one required integer model option, rejecting booleans and missing values."""
    if name not in config:
        if default is None:
            raise ValueError(f"model config requires {name!r}")
        return default
    value = config[name]
    if type(value) is not int:
        raise TypeError(f"model config {name!r} must be an integer")
    return value


def float_option(config: Mapping[str, object], name: str, default: float) -> float:
    """Read one finite numeric model option while rejecting booleans."""
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"model config {name!r} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"model config {name!r} must be finite")
    return parsed


def has_transformer_feedforward_options(config: Mapping[str, object]) -> bool:
    """Report whether a caller supplied Transformer-only feed-forward settings."""
    return "ff_activation" in config or "ff_hidden" in config


class HintProjector(nn.Module):
    """Produce learned hints or a fixed, normalized random representation."""

    def __init__(self, hidden: int, hint_dim: int, mode: str) -> None:
        super().__init__()
        self.hint_dim = hint_dim
        self.linear: nn.Linear | None = None
        self.normalizer: nn.LayerNorm | None = None
        self.register_buffer("fixed_projection", None)
        if hint_dim == 0:
            return
        if mode == "learned":
            self.linear = nn.Linear(hidden, hint_dim)
            return
        self.normalizer = nn.LayerNorm(hidden, elementwise_affine=False)
        generator = torch.Generator(device="cpu").manual_seed(1729)
        projection = torch.randn(hint_dim, hidden, generator=generator, dtype=torch.float32)
        self.fixed_projection = projection / math.sqrt(hidden)

    def forward(self, hidden: Tensor) -> Tensor:
        """Project hidden row vectors when hints are enabled."""
        if self.hint_dim == 0:
            return hidden.new_empty((*hidden.shape[:-1], 0))
        if self.linear is not None:
            return self.linear(hidden)
        if self.normalizer is None or self.fixed_projection is None:
            raise RuntimeError("fixed hint projection was not initialized")
        return F.linear(self.normalizer(hidden), self.fixed_projection)


def detach_state(state: SequenceState) -> SequenceState:
    """Detach recurrent or Transformer state at an explicit TBPTT boundary."""
    if state is None:
        return None
    if isinstance(state, Tensor):
        return state.detach()
    if isinstance(state, TransformerState):
        return TransformerState(
            tuple(key.detach() for key in state.keys),
            tuple(value.detach() for value in state.values),
            state.next_position,
        )
    raise TypeError(f"unsupported sequence state: {type(state).__name__}")
