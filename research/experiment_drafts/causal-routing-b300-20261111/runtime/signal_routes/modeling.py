"""Factory for the causal attention-map and dense FFN-routing variants."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from signal_routes.attention_block import AttentionVariant, SplitQKLocalAttention
from signal_routes.ffn_block import FFNVariant, SignalRouteBlock
from wnn.training.sequence_attention import CausalTransformerBlock
from wnn.training.sequence_attention_backends import LocalCausalAttention
from wnn.training.sequence_compile import compile_transformer_blocks
from wnn.training.sequence_models import build_sequence_model

_VARIANTS = frozenset(
    (
        "plain",
        "attention_additive",
        "attention_differential",
        "ffn_static",
        "ffn_conditional",
    )
)


def _mapping(value: object, name: str) -> dict[str, object]:
    """Validate a JSON object before reading its named fields."""
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _normalization(artifact: dict[str, object]) -> tuple[Tensor, Tensor]:
    """Read the fixed 112-feature train-only normalization vectors."""
    mean = torch.as_tensor(artifact.get("mean"), dtype=torch.float32)
    std = torch.as_tensor(artifact.get("std"), dtype=torch.float32)
    if mean.shape != (112,) or std.shape != (112,):
        raise ValueError("preprocessing mean and std must each have 112 entries")
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
        raise ValueError("preprocessing vectors must be finite with positive scales")
    return mean, std


def build_model(config: dict[str, object], device: torch.device) -> nn.Module:
    """Build a canonical PR75 Transformer with one declared signal-route adapter.

    ``config['model']`` remains the native Transformer recipe. The packet-owned
    ``config['signal_routes']`` object supplies ``cell.variant`` and the
    train-only ``preprocessing.mean/std`` vectors.
    """
    native_model = _mapping(config.get("model"), "model")
    metadata = _mapping(config.get("signal_routes"), "signal_routes")
    cell = _mapping(metadata.get("cell"), "signal_routes.cell")
    preprocessing = _mapping(metadata.get("preprocessing"), "signal_routes.preprocessing")
    variant = cell.get("variant")
    if not isinstance(variant, str) or variant not in _VARIANTS:
        raise ValueError("signal_routes.cell.variant is unsupported")
    if native_model.get("architecture") != "transformer":
        raise ValueError("signal routes require the canonical Transformer model family")
    compile_requested = native_model.get("compile_blocks", False)
    if type(compile_requested) is not bool:
        raise TypeError("model.compile_blocks must be boolean")

    mean, std = _normalization(preprocessing)
    # Build eager canonical blocks first, then replace the selected submodules,
    # and finally use the canonical compiler on the finalized blocks.
    build_config = dict(native_model)
    build_config["compile_blocks"] = False
    model = build_sequence_model(build_config)
    blocks = cast(nn.ModuleList, model.blocks)

    if variant in ("attention_additive", "attention_differential"):
        for block in blocks:
            canonical = cast(CausalTransformerBlock, block)
            attention = cast(LocalCausalAttention, canonical.local_attention)
            canonical.local_attention = SplitQKLocalAttention(
                attention, cast(AttentionVariant, variant)
            )
    elif variant in ("ffn_static", "ffn_conditional"):
        model.blocks = nn.ModuleList(
            [
                SignalRouteBlock(cast(CausalTransformerBlock, block), cast(FFNVariant, variant))
                for block in blocks
            ]
        )

    compile_transformer_blocks(cast(nn.ModuleList, model.blocks), enabled=compile_requested)
    mean_buffer = cast(Tensor, model.mean)
    std_buffer = cast(Tensor, model.std)
    with torch.no_grad():
        mean_buffer.copy_(mean)
        std_buffer.copy_(std)
    model.signal_route_variant = variant
    return model.to(device=device, dtype=torch.float32)
