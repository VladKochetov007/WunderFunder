"""Causal Transformer and canonical GRU adapters for sequence training."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from wnn.training.sequence_attention import CausalTransformerBlock, TransformerState
from wnn.training.sequence_compact_config import COMPACT_ARCHITECTURES, compact_model_settings
from wnn.training.sequence_compile import compile_transformer_blocks
from wnn.training.sequence_features import RAW_INPUT_MODE, sequence_input_width
from wnn.training.sequence_gru_adapter import GRUSequenceAdapter
from wnn.training.sequence_lstm_adapter import LSTMSequenceAdapter
from wnn.training.sequence_model_helpers import HintProjector as _HintProjector
from wnn.training.sequence_model_helpers import detach_state as detach_state
from wnn.training.sequence_model_helpers import (
    validate_sequence_fp32_environment,
    validate_sequence_input_shape,
)
from wnn.training.sequence_precision import (
    PRECISION_MODES,
    checkpoint_context_fn,
    configure_precision_mode,
)


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
        input_feature_mode: str,
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
        input_width = sequence_input_width(input_feature_mode)
        self.input_feature_mode = input_feature_mode
        self.register_buffer("mean", torch.zeros(input_width, dtype=torch.float32))
        self.register_buffer("std", torch.ones(input_width, dtype=torch.float32))
        self.input_projection = nn.Linear(input_width, hidden)
        self.blocks = nn.ModuleList(
            CausalTransformerBlock(hidden, heads, ff_width, dropout, attention_backend)
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
        validate_sequence_input_shape(values, self.mean.numel())
        validate_sequence_fp32_environment(self, values, self.precision_mode)
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
    """Build a causal Transformer or canonical GRU with FP32 master parameters.

    Args:
        config: Model settings with architecture, hidden, layers, heads,
            ff_multiplier, and context_rows. Dropout, activation checkpointing,
            hint_dim, and hint_projection are optional.

    Returns:
        A module accepting its configured causal input width and returning predictions,
        hints, and resumable sequence state.

    Raises:
        TypeError: If a setting has the wrong type or the default dtype is reduced.
        ValueError: If the architecture or dimensions are invalid.
    """
    if torch.get_default_dtype() != torch.float32:
        raise TypeError("sequence model construction requires the default dtype to be float32")
    architecture = config.get("architecture")
    input_feature_mode = config.get("input_feature_mode", RAW_INPUT_MODE)
    if not isinstance(input_feature_mode, str):
        raise TypeError("model input_feature_mode must be a string")
    sequence_input_width(input_feature_mode)
    if architecture in COMPACT_ARCHITECTURES:
        settings = compact_model_settings(config)
        configure_precision_mode("fp32")
        if architecture == "gru_local":
            from wnn.training.sequence_gru_local import GRULocalSequenceAdapter

            return GRULocalSequenceAdapter(settings)
        if architecture == "mamba2":
            from wnn.training.sequence_mamba2 import Mamba2SequenceAdapter

            return Mamba2SequenceAdapter(settings)
        from wnn.training.sequence_ema_tabm import EMATabMStyleSequenceAdapter

        return EMATabMStyleSequenceAdapter(settings)
    if architecture == "galam":
        if input_feature_mode != RAW_INPUT_MODE:
            raise ValueError("GALAM sequence adapters accept only raw112 inputs")
        from wnn.training.galam_adapter import GALAMSequenceAdapter

        return GALAMSequenceAdapter(
            {name: value for name, value in config.items() if name != "input_feature_mode"}
        )
    if architecture not in ("transformer", "gru", "lstm"):
        raise ValueError("model config architecture must be transformer, gru, or lstm")
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
    if architecture == "gru":
        if compile_blocks:
            raise ValueError("compile_blocks requires a Transformer model")
        if attention_backend != "math":
            raise ValueError("attention backend selection requires a Transformer model")
        return GRUSequenceAdapter(
            hidden,
            layers,
            dropout,
            hint_dim,
            hint_mode,
            precision_mode,
            input_feature_mode,
        )
    if architecture == "lstm":
        if compile_blocks or attention_backend != "math" or input_feature_mode != RAW_INPUT_MODE:
            raise ValueError("LSTM supports raw112 inputs without Transformer-only options")
        return LSTMSequenceAdapter(hidden, layers, dropout, hint_dim, hint_mode, precision_mode)
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
        input_feature_mode,
    )
    compile_transformer_blocks(model.blocks, enabled=compile_blocks)
    return model
