"""Causal Transformer adapter with training-only future-label heads for card07."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from wnn.training.sequence_attention import TransformerState
from wnn.training.sequence_model_helpers import check_model_precision
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_precision import checkpoint_context_fn

# Model configuration values are heterogeneous at this validated JSON boundary.
RAW_WIDTH = 112
TARGETS = 2
FUTURE_OFFSETS = (1, 16, 64)
EXPECTED_LAYERS = 24


class FutureHeadTransformer(nn.Module):
    """Reuse a canonical Transformer and expose three future-label heads in training."""

    def __init__(self, canonical_model: nn.Module, seed: int) -> None:
        """Attach zero-impact-on-prediction future heads without reinitializing the backbone."""
        super().__init__()
        blocks = getattr(canonical_model, "blocks", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) != EXPECTED_LAYERS:
            raise ValueError("card07 requires the canonical 24-layer Transformer")
        if type(seed) is not int:
            raise TypeError("future-head seed must be an integer")
        self.hidden = _positive_integer(canonical_model, "hidden")
        self.heads = _positive_integer(canonical_model, "heads")
        self.head_dim = _positive_integer(canonical_model, "head_dim")
        self.context_rows = _positive_integer(canonical_model, "context_rows")
        self.dropout = _nonnegative_float(canonical_model, "dropout")
        self.checkpoint_blocks = _boolean(canonical_model, "checkpoint_blocks")
        self.precision_mode = _string(canonical_model, "precision_mode")
        if self.hidden % self.heads or self.hidden // self.heads != self.head_dim:
            raise ValueError("canonical Transformer hidden/head dimensions are inconsistent")

        mean = getattr(canonical_model, "mean", None)
        std = getattr(canonical_model, "std", None)
        if not isinstance(mean, Tensor) or mean.shape != (RAW_WIDTH,):
            raise TypeError("canonical Transformer must expose its raw112 mean buffer")
        if not isinstance(std, Tensor) or std.shape != (RAW_WIDTH,):
            raise TypeError("canonical Transformer must expose its raw112 std buffer")
        if mean.dtype != torch.float32 or std.dtype != torch.float32:
            raise TypeError("canonical Transformer normalization buffers must be float32")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())
        self.input_projection = _module(canonical_model, "input_projection")
        self.output_norm = _module(canonical_model, "output_norm")
        self.output_head = _module(canonical_model, "output_head")
        self.hint_projector = _module(canonical_model, "hint_projector")
        if getattr(self.hint_projector, "hint_dim", None) != 0:
            raise ValueError("card07 requires the canonical empty-hint configuration")
        self.blocks = nn.ModuleList(list(blocks))
        self.future_heads = nn.ModuleList()
        self._future_heads_dropped = False
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 8107)
            self.future_heads = nn.ModuleList(
                nn.Linear(self.hidden, TARGETS) for _ in FUTURE_OFFSETS
            )

    def forward(
        self, values: Tensor, state: TransformerState | None = None
    ) -> tuple[Tensor, Tensor, TransformerState]:
        """Return primary predictions, training heads or empty hints, and stream state."""
        if self.training and self._future_heads_dropped:
            raise RuntimeError("future-label heads were removed for inference")
        return self._forward_impl(values, state, capture_future=self.training)

    def _forward_impl(
        self,
        values: Tensor,
        state: TransformerState | None,
        *,
        capture_future: bool,
    ) -> tuple[Tensor, Tensor, TransformerState]:
        """Apply the pinned causal blocks and optionally project final hidden rows."""
        if values.ndim != 3 or values.shape[-1] != RAW_WIDTH or min(values.shape[:2]) <= 0:
            raise ValueError(
                "sequence models require nonempty raw inputs shaped [batch, rows, 112]"
            )
        check_model_precision(self, values, self.precision_mode)
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
                    rows: Tensor,
                    key: Tensor,
                    value: Tensor,
                    layer: nn.Module = block,
                ) -> tuple[Tensor, Tensor, Tensor]:
                    """Checkpoint one canonical Transformer block and its cache."""
                    return layer(rows, key, value, position, self.context_rows, dropout_p)

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
        if capture_future:
            second_output = torch.stack([head(latent) for head in self.future_heads], dim=-2)
        else:
            second_output = self.hint_projector(latent)
        next_state = TransformerState(
            tuple(next_keys), tuple(next_values), position + values.shape[1]
        )
        return prediction, second_output, next_state

    def _state_for_batch(self, values: Tensor, state: TransformerState | None) -> TransformerState:
        """Validate or initialize the canonical 24-layer key/value cache."""
        if state is None:
            empty = values.new_empty(values.shape[0], self.heads, 0, self.head_dim)
            return TransformerState(
                tuple(empty for _ in self.blocks), tuple(empty for _ in self.blocks), 0
            )
        if len(state.keys) != EXPECTED_LAYERS or len(state.values) != EXPECTED_LAYERS:
            raise ValueError("future-head state must contain one cache pair per Transformer layer")
        if type(state.next_position) is not int or state.next_position < 0:
            raise ValueError("Transformer state has an invalid absolute position")
        cache_length: int | None = None
        for key, value in zip(state.keys, state.values, strict=True):
            if key.shape != value.shape or key.ndim != 4:
                raise ValueError("Transformer state key/value shapes do not match")
            if key.shape[:2] != (values.shape[0], self.heads) or key.shape[-1] != self.head_dim:
                raise ValueError("Transformer cache batch or head shape differs from this model")
            if any(
                tensor.device != values.device or tensor.dtype != torch.float32
                for tensor in (key, value)
            ):
                raise ValueError("Transformer caches must use the input device and float32")
            if cache_length is None:
                cache_length = key.shape[-2]
            elif cache_length != key.shape[-2]:
                raise ValueError("Transformer layer caches must have equal row counts")
        if (cache_length or 0) > min(self.context_rows - 1, state.next_position):
            raise ValueError("Transformer state exceeds the configured causal context")
        return state


def build_future_model(model_config: dict[str, Any], seed: int) -> FutureHeadTransformer:
    """Build the pinned Transformer backbone and attach card07's training heads."""
    if model_config.get("architecture") != "transformer":
        raise ValueError("future-label heads require a Transformer model")
    if model_config.get("layers") != EXPECTED_LAYERS:
        raise ValueError("future-label heads require 24 Transformer layers")
    return FutureHeadTransformer(build_sequence_model(model_config), seed)


def drop_future_heads(model: FutureHeadTransformer) -> None:
    """Remove training-only heads after strict checkpoint loading for serving."""
    if model._future_heads_dropped or not len(model.future_heads):
        raise ValueError("only an unexported future-head model can drop its auxiliary heads")
    if model.training:
        raise RuntimeError("switch the loaded future-head model to eval before export")
    model.future_heads = nn.ModuleList()
    model._future_heads_dropped = True


def _module(model: nn.Module, name: str) -> nn.Module:
    """Read one canonical child module while keeping its state-dict path unchanged."""
    value = getattr(model, name, None)
    if not isinstance(value, nn.Module):
        raise TypeError(f"canonical Transformer must expose module {name!r}")
    return value


def _positive_integer(model: nn.Module, name: str) -> int:
    """Read one positive integer from the pinned model interface."""
    value = getattr(model, name, None)
    if type(value) is not int or value <= 0:
        raise TypeError(f"canonical Transformer attribute {name!r} must be a positive integer")
    return value


def _nonnegative_float(model: nn.Module, name: str) -> float:
    """Read one finite nonnegative numeric model setting."""
    value = getattr(model, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise TypeError(f"canonical Transformer attribute {name!r} must be nonnegative numeric")
    return float(value)


def _boolean(model: nn.Module, name: str) -> bool:
    """Read one boolean model setting without accepting integer lookalikes."""
    value = getattr(model, name, None)
    if type(value) is not bool:
        raise TypeError(f"canonical Transformer attribute {name!r} must be boolean")
    return value


def _string(model: nn.Module, name: str) -> str:
    """Read one nonempty string model setting."""
    value = getattr(model, name, None)
    if not isinstance(value, str) or not value:
        raise TypeError(f"canonical Transformer attribute {name!r} must be a nonempty string")
    return value
