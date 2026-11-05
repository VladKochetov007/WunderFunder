"""Causal Transformer and canonical GRU adapters for raw-112 sequence training."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from wnn.nn import RNNModel
from wnn.training.sequence_attention import CausalTransformerBlock, TransformerState
from wnn.training.sequence_compile import compile_transformer_blocks
from wnn.training.sequence_model_helpers import HintProjector as _HintProjector
from wnn.training.sequence_model_helpers import detach_state as detach_state
from wnn.training.sequence_precision import (
    PRECISION_MODES,
    checkpoint_context_fn,
    configure_precision_mode,
    validate_forward_state,
)

RAW_INPUT_WIDTH = 112


def _check_fp32_environment(model: nn.Module, values: Tensor, precision_mode: str) -> None:
    if values.dtype != torch.float32:
        raise TypeError("sequence models require float32 inputs")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError("sequence models require float32 parameters")
    if any(buffer is not None and buffer.dtype != torch.float32 for buffer in model.buffers()):
        raise TypeError("sequence models require float32 buffers")
    validate_forward_state(precision_mode, values.device.type)


def _check_input_shape(values: Tensor) -> None:
    if values.ndim != 3 or values.shape[-1] != RAW_INPUT_WIDTH:
        raise ValueError("sequence models accept only raw input shaped [batch, rows, 112]")
    if values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("sequence batches and row chunks must be non-empty")


class _GRUSequenceAdapter(RNNModel):
    """Expose canonical GRU hidden rows for hints without changing its prediction path."""

    def __init__(
        self,
        hidden: int,
        layers: int,
        dropout: float,
        hint_dim: int,
        hint_mode: str,
        precision_mode: str,
    ) -> None:
        super().__init__(
            RAW_INPUT_WIDTH, hidden, layers, dropout, n_out=2, cell="gru", enc="linear"
        )
        self.precision_mode = precision_mode
        self.hint_projector = _HintProjector(hidden, hint_dim, hint_mode)

    def forward(self, values: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Return canonical GRU predictions, hidden-state hints, and recurrent state."""
        _check_input_shape(values)
        _check_fp32_environment(self, values, self.precision_mode)
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
        # Autocast can return BF16 recurrent state; the next chunk's public
        # streaming interface and saved activation state remain FP32.
        return prediction, self.hint_projector(recurrent), next_state.float()


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
        precision_mode: str,
        attention_backend: str,
        recency_bias: float,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.context_rows = context_rows
        self.dropout = dropout
        self.checkpoint_blocks = checkpoint_blocks
        self.precision_mode = precision_mode
        self.attention_backend = attention_backend
        self.register_buffer("mean", torch.zeros(RAW_INPUT_WIDTH, dtype=torch.float32))
        self.register_buffer("std", torch.ones(RAW_INPUT_WIDTH, dtype=torch.float32))
        self.input_projection = nn.Linear(RAW_INPUT_WIDTH, hidden)
        self.blocks = nn.ModuleList(
            CausalTransformerBlock(
                hidden, heads, ff_width, dropout, attention_backend, recency_bias
            )
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.output_head = nn.Linear(hidden, 2)
        self.hint_projector = _HintProjector(hidden, hint_dim, hint_mode)

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
        _check_input_shape(values)
        _check_fp32_environment(self, values, self.precision_mode)
        previous = self._state_for_batch(values, state)
        position = previous.next_position
        hidden = self.input_projection((values - self.mean) / self.std)
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
                    _run_block,
                    hidden,
                    past_key,
                    past_value,
                    use_reentrant=False,
                    context_fn=lambda: checkpoint_context_fn(
                        self.precision_mode, values.device.type
                    ),
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
    """Build a causal model with FP32 master parameters and resumable state.

    The config specifies architecture, dimensions, precision and optional hints.
    """
    if torch.get_default_dtype() != torch.float32:
        raise TypeError("sequence model construction requires the default dtype to be float32")
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
    precision_mode = config.get("precision_mode", "fp32")
    attention_backend = config.get("attention_backend", "math")
    compile_blocks = config.get("compile_blocks", False)
    recency_bias = _float_option(config, "recency_bias", 0.0)
    if type(checkpoint_blocks) is not bool:
        raise TypeError("model config 'checkpoint_blocks' must be boolean")
    if hint_mode not in ("learned", "fixed"):
        raise ValueError("model config hint_projection must be 'learned' or 'fixed'")
    if precision_mode not in PRECISION_MODES:
        raise ValueError("unsupported sequence precision mode")
    if type(compile_blocks) is not bool:
        raise TypeError("model config 'compile_blocks' must be boolean")
    configure_precision_mode(str(precision_mode))
    if attention_backend not in ("math", "efficient", "flex"):
        raise ValueError("unsupported sequence attention backend")
    if min(hidden, layers, heads, context_rows) <= 0 or hint_dim < 0:
        raise ValueError(
            "hidden, layers, heads, and context_rows must be positive; hint_dim nonnegative"
        )
    if not 0.0 <= dropout < 1.0 or ff_multiplier <= 0.0:
        raise ValueError("dropout must be in [0, 1) and ff_multiplier must be positive")
    if recency_bias and (architecture != "transformer" or attention_backend != "math"):
        raise ValueError("recency bias requires Transformer and math attention")
    if not 0 <= recency_bias <= 4:
        raise ValueError("recency bias must be in [0,4]")
    if architecture == "gru":
        if compile_blocks:
            raise ValueError("compile_blocks requires a Transformer model")
        if attention_backend != "math":
            raise ValueError("attention backend selection requires a Transformer model")
        return _GRUSequenceAdapter(hidden, layers, dropout, hint_dim, hint_mode, precision_mode)
    if hidden % heads != 0 or (hidden // heads) % 2 != 0:
        raise ValueError("Transformer hidden width must divide evenly into even rotary head widths")
    ff_width = int(hidden * ff_multiplier)
    if ff_width <= 0:
        raise ValueError("Transformer feed-forward width must be positive")
    model = _TransformerSequenceAdapter(
        hidden,
        layers,
        heads,
        ff_width,
        context_rows,
        dropout,
        checkpoint_blocks,
        hint_dim,
        hint_mode,
        precision_mode,
        attention_backend,
        recency_bias,
    )
    compile_transformer_blocks(model.blocks, enabled=compile_blocks)
    return model
