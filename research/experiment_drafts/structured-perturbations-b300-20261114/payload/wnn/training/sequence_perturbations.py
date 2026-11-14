"""Training-only, matched-count stream staleness and old-history dropout."""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from wnn.data import FEATURE_COLUMNS
from wnn.training.sequence_attention_masks import (
    _mask_storage_rows,
    _trim_window_keys,
)

INPUT_BLOCK_ROWS = 128
HISTORY_PROTECTED_ROWS = 32
I1_INDICES = tuple(i for i, name in enumerate(FEATURE_COLUMNS) if name.startswith("i1_"))
if tuple(range(52, 104)) != I1_INDICES:
    raise ValueError("the pinned raw112 i1 feature contract has changed")


def validate_perturbation_mode(value: object) -> str:
    """Reject unregistered corruption settings at the configuration boundary."""
    if not isinstance(value, str) or value not in ("none", "iid", "burst"):
        raise ValueError("perturbation mode must be 'none', 'iid', or 'burst'")
    return value


def _iid_mask(
    batch: int,
    eligible_rows: int,
    count: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    """Sample a fixed-size subset independently for each sequence."""
    result = torch.zeros((batch, eligible_rows), dtype=torch.bool, device=device)
    if count:
        ranks = torch.rand((batch, eligible_rows), device=device, generator=generator)
        selected = ranks.topk(count, dim=-1, largest=False).indices
        result.scatter_(1, selected, True)
    return result


def input_drop_mask(
    batch: int,
    rows: int,
    mode: str,
    device: torch.device,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Select the same count of i1 stale rows using individual rows or 128-row blocks.

    Row zero and the incomplete final block are protected. Exactly one quarter
    (rounded down in blocks) of complete blocks contributes to the drop count.
    For 20,000 rows both modes mark 4,992 of 19,968 eligible rows per sequence.
    Each eligible row has the same marginal probability in both treatments.
    """
    validate_perturbation_mode(mode)
    if batch < 1 or rows < 1:
        raise ValueError("input corruption requires a nonempty sequence batch")
    result = torch.zeros((batch, rows), dtype=torch.bool, device=device)
    blocks = (rows - 1) // INPUT_BLOCK_ROWS
    count_blocks = blocks // 4
    if mode == "none" or count_blocks == 0:
        return result
    eligible_rows = blocks * INPUT_BLOCK_ROWS
    if mode == "iid":
        selected = _iid_mask(
            batch, eligible_rows, count_blocks * INPUT_BLOCK_ROWS, device, generator
        )
    else:
        selected = _iid_mask(batch, blocks, count_blocks, device, generator)
        selected = selected.repeat_interleave(INPUT_BLOCK_ROWS, dim=1)
    result[:, 1 : 1 + eligible_rows] = selected
    return result


def perturb_input(
    values: Tensor,
    mode: str,
    *,
    seed: int,
    epoch: int,
    rank: int,
    batch_start: int,
) -> Tensor:
    """Replace selected i1 rows by their last observed same-sequence predecessor.

    The clean input supplies normalization and labels. This function runs only
    in train_epoch before chunking, so a held value survives chunk boundaries.
    Its private generator does not consume the model-initialization/global RNG.
    Metadata, not features or labels, determines the random corruption mask.
    """
    validate_perturbation_mode(mode)
    if mode == "none":
        return values
    if values.ndim != 3 or values.shape[-1] != 112:
        raise ValueError("input corruption requires [batch, rows, 112]")
    if min(seed, epoch, rank, batch_start) < 0:
        raise ValueError("input RNG identity fields must be nonnegative")
    identity = f"wnn-input-perturbation-v1:{seed}:{epoch}:{rank}:{batch_start}"
    random_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "little")
    generator = torch.Generator(device=values.device).manual_seed(random_seed)
    mask = input_drop_mask(
        values.shape[0], values.shape[1], mode, values.device, generator=generator
    )
    positions = torch.arange(values.shape[1], device=values.device).expand(values.shape[0], -1)
    latest_observed = positions.masked_fill(mask, 0).cummax(dim=1).values
    result = values.clone()
    indices = latest_observed.unsqueeze(-1).expand(-1, -1, len(I1_INDICES))
    result[..., 52:104] = values[..., 52:104].gather(1, indices)
    return result


def history_drop_mask(
    batch: int,
    past_rows: int,
    mode: str,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Drop a quarter of old cached keys while protecting the most recent 32.

    Both modes delete floor(eligible/4) keys. Burst uses one interval on a ring
    of eligible positions: it may wrap into two spans at the ring boundary.
    Uniform random start gives equal marginal dropout at every eligible age.
    The same per-sequence mask is shared across heads and current-chunk queries.
    """
    validate_perturbation_mode(mode)
    if batch < 1 or past_rows < 0:
        raise ValueError("history corruption requires valid batch and cache sizes")
    result = torch.zeros((batch, past_rows), dtype=torch.bool, device=device)
    eligible_rows = max(0, past_rows - HISTORY_PROTECTED_ROWS)
    count = eligible_rows // 4
    if mode == "none" or count == 0:
        return result
    if mode == "iid":
        selected = _iid_mask(batch, eligible_rows, count, device, generator)
    else:
        start = torch.randint(eligible_rows, (batch, 1), device=device, generator=generator)
        positions = torch.arange(eligible_rows, device=device)[None, :]
        selected = (positions - start).remainder(eligible_rows) < count
    result[:, :eligible_rows] = selected
    return result


def perturb_history_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_start: int,
    key_start: int,
    context_rows: int,
    dropout_p: float,
    mode: str,
    backend: str,
) -> Tensor:
    """Apply old-key dropout before softmax without editing the stored K/V cache.

    The patch calls this only for train mode and a nonempty eligible history.
    Exact native local-causal constraints remain in the additive attention mask.
    The current chunk and the nearest 32 cached rows never receive dropout.
    Additional keys naturally leaving the rolling context remain disallowed.
    No output rescaling is applied: softmax renormalizes over retained keys.
    """
    if backend not in ("math", "efficient"):
        raise ValueError("history dropout supports explicit math or efficient SDPA only")
    if backend == "efficient" and not query.is_cuda:
        raise RuntimeError("efficient history dropout requires CUDA")
    key, value, past_rows = _trim_window_keys(
        query, key, value, query_start, key_start, context_rows
    )
    batch, _, query_rows, _ = query.shape
    key_rows = key.shape[-2]
    drop = history_drop_mask(batch, past_rows, mode, query.device)
    storage_rows = _mask_storage_rows(key_rows)
    positions = torch.arange(storage_rows, device=query.device)
    queries = past_rows + torch.arange(query_rows, device=query.device)
    distance = queries[:, None] - positions[None, :]
    allowed = (distance >= 0) & (distance < context_rows) & (positions[None, :] < key_rows)
    # Aligned backing width is required by the native efficient SDPA backend.
    bias = torch.zeros((batch, 1, query_rows, storage_rows), device=query.device, dtype=query.dtype)
    bias.masked_fill_(~allowed[None, None, :, :], float("-inf"))
    bias[..., :past_rows].masked_fill_(drop[:, None, None, :], float("-inf"))
    selected_backend = SDPBackend.MATH if backend == "math" else SDPBackend.EFFICIENT_ATTENTION
    with sdpa_kernel(selected_backend):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias[..., :key_rows],
            dropout_p=dropout_p,
            is_causal=False,
        )
