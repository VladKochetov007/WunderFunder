"""Factory for matched raw-input and local-attention Transformer adapters."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from bilinear_entmax.attention_block import AttentionNormalizer, ExplicitLocalAttention
from bilinear_entmax.input_block import InputBlock, InputVariant
from wnn.training.sequence_attention import CausalTransformerBlock
from wnn.training.sequence_compile import compile_transformer_blocks
from wnn.training.sequence_models import build_sequence_model

_INPUT_VARIANTS = frozenset(("input_linear", "input_bilinear"))
_ATTENTION_VARIANTS: dict[str, AttentionNormalizer] = {
    "attention_softmax": "softmax",
    "attention_entmax": "entmax",
}
_VARIANTS = _INPUT_VARIANTS | frozenset(_ATTENTION_VARIANTS)


def _mapping(value: object, name: str) -> dict[str, object]:
    """Validate one JSON object before reading its named fields."""
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
    """Build the pinned native Transformer with one declared model adapter.

    ``config['model']`` remains the native model recipe. The packet-owned
    ``config['bilinear_entmax']`` object supplies ``cell.variant`` and the
    train-only ``preprocessing.mean/std`` vectors.
    """
    native_model = _mapping(config.get("model"), "model")
    metadata = _mapping(config.get("bilinear_entmax"), "bilinear_entmax")
    cell = _mapping(metadata.get("cell"), "bilinear_entmax.cell")
    preprocessing = _mapping(metadata.get("preprocessing"), "bilinear_entmax.preprocessing")
    variant = cell.get("variant")
    if not isinstance(variant, str) or variant not in _VARIANTS:
        raise ValueError("bilinear_entmax.cell.variant is unsupported")
    if native_model.get("architecture") != "transformer":
        raise ValueError("Bilinear Entmax adapters require the canonical Transformer model family")
    compile_requested = native_model.get("compile_blocks", False)
    if type(compile_requested) is not bool:
        raise TypeError("model.compile_blocks must be boolean")

    mean, std = _normalization(preprocessing)
    # Construct canonical parameters first, then add only the selected adapter.
    # Input-branch initialization is forked so model and subsequent RNG streams
    # remain identical to the native control at each declared seed.
    build_config = dict(native_model)
    build_config["compile_blocks"] = False
    model = build_sequence_model(build_config)
    blocks = cast(nn.ModuleList, model.blocks)

    if variant in _INPUT_VARIANTS:
        native_projection = cast(nn.Linear, model.input_projection)
        branch_device = native_projection.weight.device
        rng_devices = [branch_device] if branch_device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            model.input_projection = InputBlock(native_projection, cast(InputVariant, variant))
    else:
        normalizer = _ATTENTION_VARIANTS[variant]
        for block in blocks:
            canonical = cast(CausalTransformerBlock, block)
            canonical.local_attention = ExplicitLocalAttention(normalizer)

    compile_transformer_blocks(cast(nn.ModuleList, model.blocks), enabled=compile_requested)
    mean_buffer = cast(Tensor, model.mean)
    std_buffer = cast(Tensor, model.std)
    with torch.no_grad():
        mean_buffer.copy_(mean)
        std_buffer.copy_(std)
    model.bilinear_entmax_variant = variant
    return model.to(device=device, dtype=torch.float32)
