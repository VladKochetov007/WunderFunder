"""Measured structural counts for depth-model receipts; no throughput inference."""

from __future__ import annotations

from torch import nn


def parameter_layout(model: nn.Module) -> dict[str, int]:
    """Separate unique registered parameters from the repeated logical layer layout."""
    blocks = model.blocks
    logical_map = getattr(model, "logical_block_map", tuple(range(len(blocks))))
    unique_total = sum(parameter.numel() for parameter in model.parameters())
    unique_block = sum(parameter.numel() for block in blocks for parameter in block.parameters())
    logical_block = sum(
        sum(parameter.numel() for parameter in blocks[index].parameters()) for index in logical_map
    )
    return {
        "unique_total_parameters": unique_total,
        "unique_temporal_blocks": len(blocks),
        "logical_block_applications": len(logical_map),
        "logical_parameter_occurrences": unique_total - unique_block + logical_block,
        "logical_kv_cache_pairs": len(logical_map),
    }
