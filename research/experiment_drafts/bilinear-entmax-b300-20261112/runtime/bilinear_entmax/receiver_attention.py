"""Server-only analytical and causal-window checks for exact 1.5-entmax attention."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from bilinear_entmax.entmax import entmax15
from bilinear_entmax.receiver_support import (
    assert_close,
    assert_dtype_aware_close,
    build_probe_model,
    synthetic_tensor,
)
from wnn.training.sequence_attention_backends import LocalCausalAttention


def _offset_mask(
    query_start: int,
    key_start: int,
    query_rows: int,
    key_rows: int,
    context_rows: int,
    device: torch.device,
) -> Tensor:
    """Build the canonical local-causal mask from absolute sequence offsets."""
    query_positions = torch.arange(query_start, query_start + query_rows, device=device)
    key_positions = torch.arange(key_start, key_start + key_rows, device=device)
    distance = query_positions[:, None] - key_positions[None, :]
    return (distance >= 0) & (distance < context_rows)


def _check_analytic_cases(device: torch.device, seed: int) -> dict[str, float | bool | int]:
    """Check ties, masks, exact zeros, shift invariance, mass, and derivatives."""
    tied = torch.tensor([[[[3.0, 3.0, 0.0, -4.0]]]], device=device)
    all_allowed = torch.ones_like(tied, dtype=torch.bool)
    tied_probabilities = entmax15(tied, all_allowed, dim=-1)
    expected_ties = torch.tensor([[[[0.5, 0.5, 0.0, 0.0]]]], device=device)
    if not torch.allclose(tied_probabilities, expected_ties, atol=1e-6, rtol=0.0):
        raise AssertionError("1.5-entmax ties or exact sparse support differ from the oracle")

    single_mask = torch.tensor([[[[False, True, False, False]]]], device=device)
    single = entmax15(tied, single_mask, dim=-1)
    if not torch.equal(single, torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]], device=device)):
        raise AssertionError("1.5-entmax must return unit mass for a single allowed key")

    logits = synthetic_tensor((1, 2, 3, 7), device, seed)
    allowed = _offset_mask(8, 2, 3, 7, 4, device)[None, None]
    probabilities = entmax15(logits, allowed, dim=-1)
    row_mass = probabilities.sum(dim=-1)
    if not bool(torch.allclose(row_mass, torch.ones_like(row_mass), atol=1e-6, rtol=0.0)):
        raise AssertionError("nonempty causal rows must have unit entmax mass")
    if bool(probabilities.masked_select(~allowed.expand_as(probabilities)).count_nonzero()):
        raise AssertionError("masked causal positions must have exact zero entmax mass")
    shifted = entmax15(logits + 17.0, allowed, dim=-1)
    shift_gap = float((shifted - probabilities).abs().max().item())
    if shift_gap > 2e-6:
        raise AssertionError(f"entmax shift invariance gap {shift_gap:.8g} exceeds tolerance")

    empty = entmax15(logits[:, :, :1], torch.zeros_like(allowed[:, :, :1]), dim=-1)
    if bool(empty.count_nonzero()):
        raise AssertionError("all-masked entmax rows must return exact zeros")

    finite_logits = torch.tensor(
        [[[[1.0, 0.0, -3.0, 9.0]]]], device=device, dtype=torch.float32, requires_grad=True
    )
    finite_mask = torch.tensor([[[[True, True, True, False]]]], device=device)
    weights = torch.tensor([[[[0.4, -0.2, 0.9, 1.5]]]], device=device)
    finite_probabilities = entmax15(finite_logits, finite_mask, dim=-1)
    active = finite_mask & (finite_probabilities > 0)
    inactive = finite_mask & ~active
    if int(active.sum().item()) != 2:
        raise AssertionError("finite-difference fixture must have two active keys")
    transformed = 0.5 * finite_logits.detach()
    threshold = (transformed[active] - finite_probabilities[active].sqrt()).mean()
    support_margin = float(
        torch.minimum(
            (transformed[active] - threshold).min(),
            (threshold - transformed[inactive]).min(),
        ).item()
    )
    if support_margin <= 0.05:
        raise AssertionError("entmax derivative fixture is too close to a support boundary")

    objective = (finite_probabilities * weights).sum()
    gradient = torch.autograd.grad(objective, finite_logits)[0]
    root = finite_probabilities.detach().sqrt()
    weighted_mean = (root * weights).sum(dim=-1, keepdim=True) / root.sum(dim=-1, keepdim=True)
    analytic = root * (weights - weighted_mean)
    analytic_gap = float((gradient - analytic).abs().max().item())
    if analytic_gap > 2e-6:
        raise AssertionError(f"entmax backward differs from its analytic Jacobian: {analytic_gap}")

    epsilon = 1e-3
    finite_difference = torch.zeros_like(finite_logits)
    base = finite_logits.detach()
    for index in range(base.shape[-1]):
        plus, minus = base.clone(), base.clone()
        plus[..., index] += epsilon
        minus[..., index] -= epsilon
        plus_value = (entmax15(plus, finite_mask, dim=-1) * weights).sum()
        minus_value = (entmax15(minus, finite_mask, dim=-1) * weights).sum()
        finite_difference[..., index] = (plus_value - minus_value) / (2 * epsilon)
    finite_difference_gap = float((gradient - finite_difference).abs().max().item())
    if finite_difference_gap > 3e-3:
        raise AssertionError(
            f"entmax autograd/finite-difference gap {finite_difference_gap:.8g} exceeds tolerance"
        )
    return {
        "tie_and_exact_zero_max_abs": float((tied_probabilities - expected_ties).abs().max()),
        "single_allowed_unit_mass": True,
        "masked_exact_zeros": True,
        "shift_invariance_max_abs": shift_gap,
        "causal_row_mass_max_abs": float((row_mass - 1).abs().max().item()),
        "all_masked_exact_zeros": True,
        "support_margin": support_margin,
        "analytic_backward_max_abs": analytic_gap,
        "finite_difference_max_abs": finite_difference_gap,
    }


def _check_attention_offsets(
    config: dict[str, Any], artifact: dict[str, Any], device: torch.device, seed: int
) -> dict[str, Any]:
    """Compare both wrappers with direct FP32 local attention at nonzero offsets."""
    query_start, key_start, context_rows = 8, 2, 4
    query = synthetic_tensor((1, 2, 3, 8), device, seed + 1)
    key = synthetic_tensor((1, 2, 9, 8), device, seed + 2)
    value = synthetic_tensor((1, 2, 9, 8), device, seed + 3)
    allowed = _offset_mask(
        query_start, key_start, query.shape[-2], key.shape[-2], context_rows, device
    )[None, None]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    entmax_expected = torch.matmul(entmax15(scores, allowed, dim=-1), value)
    masked_scores = scores.masked_fill(~allowed, float("-inf"))
    softmax_expected = torch.matmul(torch.softmax(masked_scores, dim=-1), value)
    canonical = LocalCausalAttention("math")
    softmax_canonical = canonical(query, key, value, query_start, key_start, context_rows, 0.0)
    canonical_gap = float((softmax_canonical.float() - softmax_expected.float()).abs().max().item())
    if canonical_gap > 3e-4:
        raise AssertionError(
            f"canonical local softmax differs from the direct oracle: {canonical_gap}"
        )

    gaps: dict[str, float] = {}
    for variant in ("attention_softmax", "attention_entmax"):
        model = build_probe_model(config, artifact, variant, seed + 11, device)
        attention = model.blocks[0].local_attention
        actual = attention(query, key, value, query_start, key_start, context_rows, 0.0)
        expected = softmax_expected if variant == "attention_softmax" else entmax_expected
        gaps[variant] = assert_close(
            actual.float(), expected.float(), f"{variant} offset causal attention", atol=3e-4
        )
        del model
    gaps["attention_softmax_vs_canonical"] = assert_close(
        softmax_canonical.float(), softmax_expected.float(), "canonical softmax parity", atol=3e-4
    )
    return {
        "offset_mask_verified": True,
        "query_start": query_start,
        "key_start": key_start,
        "context_rows": context_rows,
        "max_abs_by_variant": gaps,
    }


def _check_fp32_normalizers_under_autocast(
    config: dict[str, Any], artifact: dict[str, Any], device: torch.device, seed: int
) -> dict[str, Any]:
    """Require FP32 score normalization while the surrounding fit uses BF16 autocast."""
    query_start, key_start, context_rows = 8, 2, 4
    allowed = _offset_mask(query_start, key_start, 3, 9, context_rows, device)[None, None]
    gaps: dict[str, float] = {}
    canonical_gaps: dict[str, dict[str, float]] = {}
    gradient_l1: dict[str, float] = {}
    for variant in ("attention_softmax", "attention_entmax"):
        model = build_probe_model(config, artifact, variant, seed + 29, device)
        attention = model.blocks[0].local_attention
        query = synthetic_tensor((1, 2, 3, 8), device, seed + 21).to(torch.bfloat16)
        key = synthetic_tensor((1, 2, 9, 8), device, seed + 22).to(torch.bfloat16)
        value = synthetic_tensor((1, 2, 9, 8), device, seed + 23, scale=128.0).to(torch.bfloat16)
        query.requires_grad_()
        key.requires_grad_()
        value.requires_grad_()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            with torch.autocast(device_type=device.type, enabled=False):
                scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
                scores = scores / math.sqrt(query.shape[-1])
                if variant == "attention_entmax":
                    probabilities = entmax15(scores, allowed, dim=-1)
                else:
                    probabilities = torch.softmax(
                        scores.masked_fill(~allowed, float("-inf")), dim=-1
                    )
            if probabilities.dtype != torch.float32:
                raise AssertionError(f"{variant} probability normalizer left FP32 under autocast")
            expected = torch.matmul(probabilities, value)
            actual = attention(query, key, value, query_start, key_start, context_rows, 0.0)
            if variant == "attention_softmax":
                parity_value = synthetic_tensor((1, 2, 9, 8), device, seed + 24).to(torch.bfloat16)
                parity_actual = attention(
                    query, key, parity_value, query_start, key_start, context_rows, 0.0
                )
                native = LocalCausalAttention("efficient")(
                    query, key, parity_value, query_start, key_start, context_rows, 0.0
                )
        gaps[variant] = assert_close(
            actual.float(), expected.float(), f"{variant} FP32-normalizer autocast path", atol=0.03
        )
        if variant == "attention_softmax":
            canonical_gaps[variant] = assert_dtype_aware_close(
                parity_actual.float(),
                native.float(),
                "BF16 canonical efficient softmax parity",
                atol=2e-2,
                rtol=2e-2,
            )
        probe = synthetic_tensor(tuple(actual.shape), device, seed + 37).to(actual.dtype)
        (actual.float() * probe.float()).sum().backward()
        required_gradients = (query.grad, key.grad, value.grad)
        if any(
            gradient is None or not bool(torch.isfinite(gradient).all())
            for gradient in required_gradients
        ):
            raise AssertionError(
                f"{variant} BF16 attention path has missing or nonfinite gradients"
            )
        gradient_l1[variant] = sum(
            float(gradient.float().abs().sum().item())
            for gradient in required_gradients
            if gradient is not None
        )
        if gradient_l1[variant] <= 1e-8:
            raise AssertionError(f"{variant} BF16 attention path has no gradient")
        del model
    return {
        "normalizer_output_fp32": True,
        "max_abs_by_variant": gaps,
        "bf16_native_softmax_parity": canonical_gaps,
        "qkv_gradient_l1": gradient_l1,
    }


def check_attention_contract(
    config: dict[str, Any], artifact: dict[str, Any], device: torch.device, seed: int
) -> dict[str, Any]:
    """Run the small-vector entmax oracles and paired causal attention probes."""
    return {
        "entmax_analytic": _check_analytic_cases(device, seed),
        "causal_attention_offsets": _check_attention_offsets(config, artifact, device, seed + 17),
        "bf16_autocast_normalizer": _check_fp32_normalizers_under_autocast(
            config, artifact, device, seed + 31
        ),
        "alpha": 1.5,
    }
