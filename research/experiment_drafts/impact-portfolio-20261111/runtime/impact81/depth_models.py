"""Depth-supervision and tied-depth Transformer adapters for impact81 cards 04–05."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from wnn.training.sequence_attention import TransformerState
from wnn.training.sequence_model_helpers import check_model_precision
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_precision import checkpoint_context_fn

DepthVariant = Literal["plain", "deep_supervision", "copied_depth", "tied_depth"]
LOGICAL_DEPTH = 24
TIED_DEPTH = 12
AUXILIARY_BLOCKS = (5, 11, 17)


class DepthTransformerModel(nn.Module):
    """Keep canonical parameter paths while exposing experiment-specific depth behavior."""

    def __init__(self, canonical_model: nn.Module, variant: DepthVariant, seed: int) -> None:
        """Reuse the canonical factory's modules and add the declared depth treatment."""
        super().__init__()
        canonical_blocks = getattr(canonical_model, "blocks", None)
        if (
            not isinstance(canonical_blocks, nn.ModuleList)
            or len(canonical_blocks) != LOGICAL_DEPTH
        ):
            raise ValueError("impact81 depth models require the canonical 24-block Transformer")
        if variant not in ("plain", "deep_supervision", "copied_depth", "tied_depth"):
            raise ValueError(f"unsupported impact81 depth variant: {variant}")
        if type(seed) is not int:
            raise TypeError("depth model seed must be an integer")

        self.variant = variant
        self.logical_depth = LOGICAL_DEPTH
        self.logical_block_map = tuple(range(LOGICAL_DEPTH))
        self.hidden = _integer_attribute(canonical_model, "hidden")
        self.heads = _integer_attribute(canonical_model, "heads")
        self.head_dim = _integer_attribute(canonical_model, "head_dim")
        self.context_rows = _integer_attribute(canonical_model, "context_rows")
        self.dropout = _float_attribute(canonical_model, "dropout")
        self.checkpoint_blocks = _bool_attribute(canonical_model, "checkpoint_blocks")
        self.precision_mode = _string_attribute(canonical_model, "precision_mode")

        mean = getattr(canonical_model, "mean", None)
        std = getattr(canonical_model, "std", None)
        if not isinstance(mean, Tensor) or not isinstance(std, Tensor):
            raise TypeError("canonical Transformer must expose FP32 mean and std buffers")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())
        self.input_projection = _module_attribute(canonical_model, "input_projection")
        self.output_norm = _module_attribute(canonical_model, "output_norm")
        self.output_head = _module_attribute(canonical_model, "output_head")
        self.hint_projector = _module_attribute(canonical_model, "hint_projector")
        self.blocks = nn.ModuleList(list(canonical_blocks))
        self.intermediate_heads = nn.ModuleList()
        self._auxiliary_heads_dropped = False

        if variant == "copied_depth":
            self._copy_first_half_into_second_half()
        elif variant == "tied_depth":
            self.blocks = nn.ModuleList(list(canonical_blocks[:TIED_DEPTH]))
            self.logical_block_map = tuple(range(TIED_DEPTH)) * 2
        elif variant == "deep_supervision":
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 8104)
                self.intermediate_heads = nn.ModuleList(
                    nn.Linear(self.hidden, 2) for _ in AUXILIARY_BLOCKS
                )

    def _copy_first_half_into_second_half(self) -> None:
        """Value-copy the first twelve blocks without sharing parameter objects."""
        for index in range(TIED_DEPTH):
            self.blocks[index + TIED_DEPTH].load_state_dict(
                self.blocks[index].state_dict(), strict=True
            )

    def forward(
        self, values: Tensor, state: TransformerState | None = None
    ) -> tuple[Tensor, Tensor, TransformerState]:
        """Run the causal stream; training returns card04 heads in the second slot."""
        if self.training and self.variant == "deep_supervision":
            if self._auxiliary_heads_dropped:
                raise RuntimeError("deep-supervision heads were removed for inference")
            return self.forward_training(values, state)
        return self._forward_impl(values, state, capture_auxiliary=False)

    def forward_training(
        self, values: Tensor, state: TransformerState | None = None
    ) -> tuple[Tensor, Tensor, TransformerState]:
        """Return predictions, three auxiliary heads, and state in training mode.

        A DDP caller should invoke the wrapped module's normal ``forward`` so
        reducer preparation sees the returned auxiliary-head graph.
        """
        if not self.training or self.variant != "deep_supervision":
            raise RuntimeError("forward_training requires a training deep-supervision model")
        if self._auxiliary_heads_dropped:
            raise RuntimeError("deep-supervision heads were removed for inference")
        return self._forward_impl(values, state, capture_auxiliary=True)

    def _forward_impl(
        self,
        values: Tensor,
        state: TransformerState | None,
        *,
        capture_auxiliary: bool,
    ) -> tuple[Tensor, Tensor, TransformerState]:
        """Apply every logical block with an independent cache and local captures."""
        if values.ndim != 3 or values.shape[-1] != 112 or min(values.shape[:2]) <= 0:
            raise ValueError("sequence models accept nonempty raw inputs shaped [batch, rows, 112]")
        check_model_precision(self, values, self.precision_mode)
        previous = self._state_for_batch(values, state)
        position = previous.next_position
        hidden = self.input_projection((values - self.mean) / self.std)
        dropout_p = self.dropout if self.training else 0.0
        next_keys: list[Tensor] = []
        next_values: list[Tensor] = []
        auxiliary_predictions: list[Tensor] = []

        for logical_index, block_index in enumerate(self.logical_block_map):
            block = self.blocks[block_index]
            past_key = previous.keys[logical_index]
            past_value = previous.values[logical_index]
            if self.checkpoint_blocks and self.training:

                def _run_block(
                    rows: Tensor,
                    key: Tensor,
                    value: Tensor,
                    layer: nn.Module = block,
                ) -> tuple[Tensor, Tensor, Tensor]:
                    """Checkpoint one block call with its logical-layer cache."""
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
                    hidden,
                    past_key,
                    past_value,
                    position,
                    self.context_rows,
                    dropout_p,
                )
            next_keys.append(key)
            next_values.append(value)
            if capture_auxiliary and logical_index in AUXILIARY_BLOCKS:
                head_index = AUXILIARY_BLOCKS.index(logical_index)
                auxiliary_predictions.append(self.intermediate_heads[head_index](hidden))

        latent = self.output_norm(hidden)
        prediction = self.output_head(latent)
        next_state = TransformerState(
            tuple(next_keys), tuple(next_values), position + values.shape[1]
        )
        if capture_auxiliary:
            if len(auxiliary_predictions) != len(AUXILIARY_BLOCKS):
                raise RuntimeError("intermediate supervision did not capture all configured depths")
            second_output = torch.stack(auxiliary_predictions, dim=-2)
        else:
            second_output = self.hint_projector(latent)
        return prediction, second_output, next_state

    def _state_for_batch(self, values: Tensor, state: TransformerState | None) -> TransformerState:
        """Validate or initialize the 24 logical-layer cache, including tied models."""
        if state is None:
            empty = values.new_empty(values.shape[0], self.heads, 0, self.head_dim)
            return TransformerState(
                tuple(empty for _ in range(self.logical_depth)),
                tuple(empty for _ in range(self.logical_depth)),
                0,
            )
        if len(state.keys) != self.logical_depth or len(state.values) != self.logical_depth:
            raise ValueError("Transformer state must contain one key/value cache per logical layer")
        if type(state.next_position) is not int or state.next_position < 0:
            raise ValueError("Transformer state has an invalid absolute position")
        cache_length: int | None = None
        for key, value in zip(state.keys, state.values, strict=True):
            if key.shape != value.shape or key.ndim != 4:
                raise ValueError("Transformer state key/value shapes do not match")
            if key.shape[:2] != (values.shape[0], self.heads) or key.shape[-1] != self.head_dim:
                raise ValueError("Transformer state batch or head shape differs from this model")
            if any(
                tensor.device != values.device or tensor.dtype != torch.float32
                for tensor in (key, value)
            ):
                raise ValueError("Transformer state must use the input device and float32")
            if cache_length is None:
                cache_length = key.shape[-2]
            elif cache_length != key.shape[-2]:
                raise ValueError("Transformer layer caches must have equal row counts")
        if (cache_length or 0) > min(self.context_rows - 1, state.next_position):
            raise ValueError("Transformer state exceeds the configured causal context")
        return state


def build_depth_model(
    model_config: dict[str, object], variant: DepthVariant, seed: int
) -> DepthTransformerModel:
    """Build a card04/05 model from the canonical 24-layer factory."""
    if model_config.get("architecture") != "transformer":
        raise ValueError("impact81 depth experiments require a Transformer")
    if model_config.get("layers") != LOGICAL_DEPTH:
        raise ValueError("impact81 depth experiments require 24 logical layers")
    canonical_model = build_sequence_model(model_config)
    return DepthTransformerModel(canonical_model, variant, seed)


def drop_auxiliary_heads(model: DepthTransformerModel) -> None:
    """Remove card04 training heads after a strict checkpoint load for serving."""
    if model.variant != "deep_supervision" or model._auxiliary_heads_dropped:
        raise ValueError("only an unexported deep-supervision model can drop auxiliary heads")
    if model.training:
        raise RuntimeError("switch the loaded deep-supervision model to eval before export")
    model.intermediate_heads = nn.ModuleList()
    model._auxiliary_heads_dropped = True


def _module_attribute(model: nn.Module, name: str) -> nn.Module:
    """Read and validate one canonical child module without wrapping its parameter names."""
    value = getattr(model, name, None)
    if not isinstance(value, nn.Module):
        raise TypeError(f"canonical Transformer must expose module {name!r}")
    return value


def _integer_attribute(model: nn.Module, name: str) -> int:
    """Read one positive canonical integer dimension."""
    value = getattr(model, name, None)
    if type(value) is not int or value <= 0:
        raise TypeError(f"canonical Transformer attribute {name!r} must be a positive integer")
    return value


def _float_attribute(model: nn.Module, name: str) -> float:
    """Read one finite canonical floating-point setting."""
    value = getattr(model, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"canonical Transformer attribute {name!r} must be numeric")
    return float(value)


def _bool_attribute(model: nn.Module, name: str) -> bool:
    """Read one canonical boolean setting."""
    value = getattr(model, name, None)
    if type(value) is not bool:
        raise TypeError(f"canonical Transformer attribute {name!r} must be boolean")
    return value


def _string_attribute(model: nn.Module, name: str) -> str:
    """Read one canonical string setting."""
    value = getattr(model, name, None)
    if not isinstance(value, str):
        raise TypeError(f"canonical Transformer attribute {name!r} must be a string")
    return value
