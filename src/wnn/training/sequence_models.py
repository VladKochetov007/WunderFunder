"""Causal Transformer numerical-input and canonical raw-112 GRU adapters."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from wnn.nn import RNNModel
from wnn.training.sequence_attention import CausalTransformerBlock, TransformerState
from wnn.training.sequence_model_support import (
    _check_fp32_environment,
    _check_input_shape,
    _HintProjector,
)
from wnn.training.sequence_numeric import NumericBasis

RAW_INPUT_WIDTH = 112
SequenceState = Tensor | TransformerState | None


class _GRUSequenceAdapter(RNNModel):
    """Expose canonical GRU hidden rows for hints without changing its prediction path."""

    def __init__(
        self, hidden: int, layers: int, dropout: float, hint_dim: int, hint_mode: str
    ) -> None:
        super().__init__(
            RAW_INPUT_WIDTH, hidden, layers, dropout, n_out=2, cell="gru", enc="linear"
        )
        self.hint_projector = _HintProjector(hidden, hint_dim, hint_mode)

    def forward(self, values: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Return canonical GRU predictions, hidden-state hints, and recurrent state."""
        _check_input_shape(values)
        _check_fp32_environment(self, values)
        if state is not None and (
            state.shape != (self.gru.num_layers, values.shape[0], self.hidden)
            or state.device != values.device
            or state.dtype != torch.float32
        ):
            raise ValueError(
                "GRU state must match the model layers, batch, device, and float32 dtype"
            )
        encoded = self.encode((values - self.mean) / self.std)
        recurrent, next_state = self.gru(encoded, state)
        prediction = self.head(torch.cat((recurrent, encoded), dim=-1))
        return prediction, self.hint_projector(recurrent), next_state


class _TransformerSequenceAdapter(nn.Module):
    """Causal RoPE Transformer; receptive field spans up to 1 + layers * (context_rows - 1)."""

    def __init__(
        self,
        hidden: int,
        layers: int,
        heads: int,
        ff_width: int,
        context_rows: int,
        dropout: float,
        checkpoint_blocks: bool,
        hint_dim: int,
        hint_mode: str,
        input_encoding: str = "raw112",
    ) -> None:
        super().__init__()
        self.input_encoding = input_encoding
        self.input_width = 112 if input_encoding == "raw112" else 323
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.context_rows = context_rows
        self.dropout = dropout
        self.checkpoint_blocks = checkpoint_blocks
        self.register_buffer("mean", torch.zeros(self.input_width, dtype=torch.float32))
        self.register_buffer("std", torch.ones(self.input_width, dtype=torch.float32))
        self.input_projection = nn.Linear(self.input_width, hidden)
        self.blocks = nn.ModuleList(
            CausalTransformerBlock(hidden, heads, ff_width, dropout) for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.output_head = nn.Linear(hidden, 2)
        self.hint_projector = _HintProjector(hidden, hint_dim, hint_mode)
        self.numeric_basis = (
            NumericBasis(hidden, input_encoding)
            if input_encoding in ("uniform16", "quantile16")
            else None
        )

    def _state_for_batch(self, values: Tensor, state: TransformerState | None) -> TransformerState:
        if state is None:
            empty = values.new_empty(values.shape[0], self.heads, 0, self.head_dim)
            return TransformerState(
                tuple(empty for _ in self.blocks), tuple(empty for _ in self.blocks), 0
            )
        if len(state.keys) != len(self.blocks) or len(state.values) != len(self.blocks):
            raise ValueError("Transformer state layer count differs from this model")
        if type(state.next_position) is not int or state.next_position < 0:
            raise ValueError("Transformer state has an invalid absolute position")
        cache_length: int | None = None
        for key, value in zip(state.keys, state.values, strict=True):
            if key.shape != value.shape or key.ndim != 4:
                raise ValueError("Transformer state key/value shapes do not match")
            if key.shape[:2] != (values.shape[0], self.heads) or key.shape[-1] != self.head_dim:
                raise ValueError("Transformer state batch or head shape differs from this model")
            if (
                key.device != values.device
                or value.device != values.device
                or key.dtype != torch.float32
                or value.dtype != torch.float32
            ):
                raise ValueError("Transformer state must use the input device and float32")
            if cache_length is None:
                cache_length = key.shape[-2]
            elif cache_length != key.shape[-2]:
                raise ValueError("Transformer layer caches must have the same row count")
        if (cache_length or 0) > min(self.context_rows - 1, state.next_position):
            raise ValueError("Transformer state exceeds the configured causal context")
        return state

    def forward(
        self,
        values: Tensor,
        state: TransformerState | None = None,
    ) -> tuple[Tensor, Tensor, TransformerState]:
        """Predict each raw row using only current and bounded past sequence rows."""
        _check_input_shape(values, self.input_width)
        _check_fp32_environment(self, values)
        previous = self._state_for_batch(values, state)
        position = previous.next_position
        normalized = (values - self.mean) / self.std
        hidden = self.input_projection(normalized)
        if self.numeric_basis is not None:
            hidden = hidden + self.numeric_basis(normalized[..., :112])
        dropout_p = self.dropout if self.training else 0.0
        next_keys: list[Tensor] = []
        next_values: list[Tensor] = []
        for block, past_key, past_value in zip(
            self.blocks, previous.keys, previous.values, strict=True
        ):
            if self.checkpoint_blocks and self.training:

                def _run_block(
                    x: Tensor,
                    key: Tensor,
                    value: Tensor,
                    layer: CausalTransformerBlock = block,
                ) -> tuple[Tensor, Tensor, Tensor]:
                    return layer(x, key, value, position, self.context_rows, dropout_p)

                hidden, key, value = checkpoint(
                    _run_block, hidden, past_key, past_value, use_reentrant=False
                )
            else:
                hidden, key, value = block(
                    hidden, past_key, past_value, position, self.context_rows, dropout_p
                )
            next_keys.append(key)
            next_values.append(value)
        latent = self.output_norm(hidden)
        prediction = self.output_head(latent)
        next_state = TransformerState(
            tuple(next_keys), tuple(next_values), position + values.shape[1]
        )
        return prediction, self.hint_projector(latent), next_state


def _integer_option(config: dict[str, object], name: str, default: int | None = None) -> int:
    if name not in config:
        if default is None:
            raise ValueError(f"model config requires {name!r}")
        return default
    value = config[name]
    if type(value) is not int:
        raise TypeError(f"model config {name!r} must be an integer")
    return value


def _float_option(config: dict[str, object], name: str, default: float) -> float:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"model config {name!r} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"model config {name!r} must be finite")
    return parsed


def build_sequence_model(config: dict[str, object]) -> nn.Module:
    """Build a strict-FP32 causal Transformer or canonical GRU adapter.

    Args:
        config: Model settings with architecture, hidden, layers, heads,
            ff_multiplier, and context_rows. Dropout, activation checkpointing,
            hint_dim, and hint_projection are optional.

    Returns:
        A module accepting raw ``[batch, rows, 112]`` values and returning
        predictions, hints, and resumable sequence state.

    Raises:
        TypeError: If a setting has the wrong type or the default dtype is reduced.
        ValueError: If the architecture or dimensions are invalid.
    """
    if torch.get_default_dtype() != torch.float32:
        raise TypeError("sequence model construction requires the default dtype to be float32")
    input_encoding = config.get("input_encoding", "raw112")
    if input_encoding not in ("raw112", "raw_events", "uniform16", "quantile16"):
        raise ValueError("unsupported input encoding")
    architecture = config.get("architecture")
    if architecture not in ("transformer", "gru"):
        raise ValueError("model config architecture must be 'transformer' or 'gru'")
    hidden = _integer_option(config, "hidden")
    layers = _integer_option(config, "layers")
    heads = _integer_option(config, "heads")
    context_rows = _integer_option(config, "context_rows")
    hint_dim = _integer_option(config, "hint_dim", 0)
    dropout = _float_option(config, "dropout", 0.0)
    ff_multiplier = _float_option(config, "ff_multiplier", 4.0)
    checkpoint_blocks = config.get("checkpoint_blocks", False)
    hint_mode = config.get("hint_projection", "learned")
    if type(checkpoint_blocks) is not bool:
        raise TypeError("model config 'checkpoint_blocks' must be boolean")
    if hint_mode not in ("learned", "fixed"):
        raise ValueError("model config hint_projection must be 'learned' or 'fixed'")
    if min(hidden, layers, heads, context_rows) <= 0 or hint_dim < 0:
        raise ValueError(
            "hidden, layers, heads, and context_rows must be positive; hint_dim nonnegative"
        )
    if not 0.0 <= dropout < 1.0 or ff_multiplier <= 0.0:
        raise ValueError("dropout must be in [0, 1) and ff_multiplier must be positive")
    if architecture == "gru":
        if input_encoding != "raw112":
            raise ValueError("numerical encoders require a Transformer")
        return _GRUSequenceAdapter(hidden, layers, dropout, hint_dim, hint_mode)
    if hidden % heads != 0 or (hidden // heads) % 2 != 0:
        raise ValueError("Transformer hidden width must divide evenly into even rotary head widths")
    ff_width = int(hidden * ff_multiplier)
    if ff_width <= 0:
        raise ValueError("Transformer feed-forward width must be positive")
    return _TransformerSequenceAdapter(
        hidden,
        layers,
        heads,
        ff_width,
        context_rows,
        dropout,
        checkpoint_blocks,
        hint_dim,
        hint_mode,
        str(input_encoding),
    )


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
