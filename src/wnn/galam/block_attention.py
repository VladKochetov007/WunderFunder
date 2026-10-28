from __future__ import annotations

import math

import torch
from torch import nn

from wnn.galam.attention import TemporalSelfAttention
from wnn.galam.state import AttentionCache


def attention_chunk(
    module: TemporalSelfAttention,
    x: torch.Tensor,
    timestamps: torch.Tensor,
    cache: AttentionCache | None,
    *,
    shared_ordinal_times: bool = False,
) -> tuple[torch.Tensor, AttentionCache]:
    """Apply the same causal history window to all chunk queries in one batch.

    Stored keys retain their original computation graph until the caller detaches
    GALAMState. Provided timestamps change the bias, not the causal position mask.
    The ordinal-only flag requires a model-generated shared time grid including
    its cache; it avoids evaluating the same bias MLP once per batch member.
    """
    if module._monitor_observer is not None:
        raise ValueError("sampled step diagnostics require the step forward path")
    batch, length, _ = x.shape
    query = _project(module, module.query(x), module.query_norm)
    keys = _project(module, module.key(x), module.key_norm)
    values = _project(module, module.value(x))
    prefix = 0 if cache is None else cache.keys.shape[2]
    key_times = timestamps
    if cache is not None:
        keys = torch.cat((cache.keys, keys), dim=2)
        values = torch.cat((cache.values, values), dim=2)
        key_times = torch.cat((cache.timestamps, timestamps), dim=1)
    query_positions = prefix + torch.arange(length, device=x.device)
    key_positions = torch.arange(keys.shape[2], device=x.device)
    distance = query_positions[:, None] - key_positions[None, :]
    allowed = (distance >= 0) & (distance < module.history)
    scores = torch.matmul(query, keys.transpose(-1, -2)) / math.sqrt(module.head_width)
    if shared_ordinal_times:
        # Broadcasting accumulates parameter gradients from all batch members;
        # this is reused only inside one forward, never across optimizer updates.
        bias = module.relative_bias(timestamps[:1], key_times[:1])
    else:
        bias = module.relative_bias(timestamps, key_times)
    scores = scores + bias
    weights = torch.softmax(scores.masked_fill(~allowed[None, None], -torch.inf), dim=-1)
    attended = torch.matmul(weights, values).transpose(1, 2).reshape(batch, length, module.width)
    keep = slice(-module.history, None)
    next_cache = AttentionCache(keys[:, :, keep], values[:, :, keep], key_times[:, keep])
    return module.output(attended), next_cache


def _project(
    module: TemporalSelfAttention, values: torch.Tensor, norm: nn.RMSNorm | None = None
) -> torch.Tensor:
    """Split all time positions into heads without merging the batch and time axes."""
    heads = values.reshape(*values.shape[:2], module.heads, module.head_width).transpose(1, 2)
    return norm(heads) if norm is not None else heads
