"""Focused tests for bounded rolling attention backend semantics."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from wnn.training.sequence_attention import CausalTransformerBlock, _rotate
from wnn.training.sequence_attention_backends import LocalCausalAttention
from wnn.training.sequence_attention_capabilities import attention_backend_capabilities
from wnn.training.sequence_attention_masks import _relative_window_mask, _window_mask


def _legacy_bounded_math(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_start: int,
    key_start: int,
    context_rows: int,
) -> Tensor:
    """Independent reference for the pre-selector FP32 math implementation."""
    outputs = []
    for query_offset in range(0, query.shape[-2], 256):
        query_end = min(query_offset + 256, query.shape[-2])
        absolute_start = query_start + query_offset
        absolute_end = query_start + query_end
        first_key = max(0, absolute_start - (context_rows - 1) - key_start)
        end_key = min(key.shape[-2], absolute_end - key_start)
        key_offsets = torch.arange(first_key, end_key, device=query.device)
        query_positions = torch.arange(absolute_start, absolute_end, device=query.device)
        key_positions = key_offsets + key_start
        allowed = (query_positions[:, None] - key_positions[None, :] >= 0) & (
            query_positions[:, None] - key_positions[None, :] < context_rows
        )
        with sdpa_kernel(SDPBackend.MATH):
            outputs.append(
                F.scaled_dot_product_attention(
                    query[..., query_offset:query_end, :],
                    key[..., first_key:end_key, :],
                    value[..., first_key:end_key, :],
                    attn_mask=allowed[None, None, :, :],
                    dropout_p=0.0,
                    is_causal=False,
                )
            )
    return torch.cat(outputs, dim=-2)


def test_default_math_is_numerically_identical_to_previous_fp32_path() -> None:
    torch.manual_seed(17)
    query = torch.randn(2, 3, 513, 8, dtype=torch.float32)
    key = torch.randn(2, 3, 517, 8, dtype=torch.float32)
    value = torch.randn_like(key)
    expected = _legacy_bounded_math(query, key, value, 2048, 2044, 32)
    default = LocalCausalAttention()(query, key, value, 2048, 2044, 32)
    explicit = LocalCausalAttention("math")(query, key, value, 2048, 2044, 32)

    torch.testing.assert_close(default, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(explicit, expected, rtol=0.0, atol=0.0)


def test_relative_mask_is_exact_for_causal_rolling_window() -> None:
    expected = torch.tensor(
        [
            [True, True, True, False, False],
            [False, True, True, True, False],
            [False, False, True, True, True],
        ]
    )
    actual = _window_mask(3, 5, past_rows=2, context_rows=3, device=torch.device("cpu"))[0, 0]
    assert torch.equal(actual, expected)

    query_index = torch.arange(3)[:, None]
    key_index = torch.arange(5)[None, :]
    assert torch.equal(_relative_window_mask(query_index, key_index, 2, 3), expected)


def test_rotary_bf16_computes_in_fp32_and_casts_back() -> None:
    torch.manual_seed(31)
    values = torch.randn(2, 2, 5, 8, dtype=torch.float32).to(torch.bfloat16)
    positions = torch.arange(13, 18)
    inverse_frequency = torch.tensor([1.0, 0.1, 0.01, 0.001], dtype=torch.float32)
    angles = positions[:, None] * inverse_frequency[None, :]
    cosine = angles.cos()[None, None, :, :]
    sine = angles.sin()[None, None, :, :]
    paired = values.float().reshape(*values.shape[:-1], -1, 2)
    expected = (
        torch.stack(
            (
                paired[..., 0] * cosine - paired[..., 1] * sine,
                paired[..., 0] * sine + paired[..., 1] * cosine,
            ),
            dim=-1,
        )
        .flatten(-2)
        .to(torch.bfloat16)
    )

    actual = _rotate(values, positions, inverse_frequency)

    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_block_keeps_public_kv_cache_fp32_under_bf16_autocast() -> None:
    torch.manual_seed(37)
    block = CausalTransformerBlock(8, 2, 16, 0.0).eval()
    values = torch.randn(1, 5, 8, dtype=torch.float32)
    past_key = torch.randn(1, 2, 3, 4, dtype=torch.float32)
    past_value = torch.randn_like(past_key)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, next_key, next_value = block(values, past_key, past_value, 3, 4, 0.0)

    assert output.shape == values.shape
    assert next_key.dtype is torch.float32
    assert next_value.dtype is torch.float32
    assert next_key.shape[-2] == 3


def test_capability_report_names_flash_and_cudnn_without_selecting_them() -> None:
    query = torch.zeros(1, 2, 3, 4)
    key = torch.zeros(1, 2, 5, 4)
    value = torch.zeros_like(key)

    report = attention_backend_capabilities(query, key, value, 2, 0, 3)

    assert report["math"]["eligible"] is True
    assert report["efficient"]["selectable"] is True
    for backend in ("flash", "cudnn"):
        assert report[backend]["eligible"] is False
        assert report[backend]["selectable"] is False
        assert report[backend]["reason"] == "requires CUDA tensors"


def test_explicit_efficient_backend_does_not_fall_back_on_cpu() -> None:
    query = torch.zeros(1, 2, 3, 4)
    key = torch.zeros(1, 2, 5, 4)
    value = torch.zeros_like(key)

    with pytest.raises(RuntimeError, match="requires CUDA; no fallback"):
        LocalCausalAttention("efficient")(query, key, value, 2, 0, 3)


def test_attention_backend_name_is_validated() -> None:
    with pytest.raises(ValueError, match="attention_backend"):
        LocalCausalAttention("auto")
