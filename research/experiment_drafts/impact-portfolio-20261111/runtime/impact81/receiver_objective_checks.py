"""Server-only deterministic checks for the PR81 auxiliary objective contract."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .auxiliary_objectives import (
    DEEP_SUPERVISION_COEFFICIENT,
    PAIRWISE_COEFFICIENT,
    deep_supervision_scaled_loss,
    deep_supervision_wmse_numerators,
    pairwise_objective_sums,
    pairwise_scaled_loss,
    pairwise_window_denominators,
)
from .receiver_objective_ddp_checks import check_distributed_equivalence

TOLERANCE = 1e-5


def _close(actual: Tensor, expected: Tensor, label: str) -> float:
    """Require finite allclose tensors and return their maximum absolute gap."""
    if not bool(torch.isfinite(actual).all()):
        raise AssertionError(f"{label} produced nonfinite values")
    if actual.shape != expected.shape or not torch.allclose(
        actual, expected, atol=TOLERANCE, rtol=TOLERANCE
    ):
        if actual.shape != expected.shape:
            raise AssertionError(f"{label} shape differs: {actual.shape} != {expected.shape}")
        difference = float((actual - expected).abs().max().detach().item())
        raise AssertionError(f"{label} differs by {difference:.8g}")
    return float((actual - expected).abs().max().detach().item()) if actual.numel() else 0.0


def _manual_deep_supervision(device: torch.device) -> dict[str, float]:
    """Check clipped WMSE sums, mean coefficient, per-head gradients and masked NaNs."""
    target = torch.tensor([[[1.0, -3.0], [0.5, 0.0]]], device=device)
    clipped = torch.tensor([[[1.0, -2.0], [0.5, 0.0]]], device=device)
    mask = torch.ones((1, 2), dtype=torch.bool, device=device)
    heads = tuple((clipped + offset).detach().requires_grad_(True) for offset in (1.0, 2.0, 3.0))
    numerators = deep_supervision_wmse_numerators(heads, target, mask)
    expected_numerators = torch.tensor([3.5, 14.0, 31.5], device=device)
    _close(numerators, expected_numerators, "card04 manual per-head WMSE numerators")
    denominator = torch.tensor(3.5, device=device)
    loss = deep_supervision_scaled_loss(numerators, denominator, world_size=1)
    expected_loss = torch.tensor(
        DEEP_SUPERVISION_COEFFICIENT * (1.0 + 4.0 + 9.0) / 3.0, device=device
    )
    _close(loss.reshape(1), expected_loss.reshape(1), "card04 manual auxiliary loss")
    gradients = torch.autograd.grad(loss, heads)
    weights = clipped.abs()
    gradient_gaps = []
    for offset, gradient in zip((1.0, 2.0, 3.0), gradients, strict=True):
        expected_gradient = (
            DEEP_SUPERVISION_COEFFICIENT / 3.0 * 2.0 * offset * weights / denominator
        )
        gradient_gaps.append(_close(gradient, expected_gradient, "card04 per-head gradient"))

    masked_target = torch.tensor([[[1.0, -3.0], [float("nan"), float("nan")]]], device=device)
    masked_rows = torch.tensor([[True, False]], dtype=torch.bool, device=device)
    masked_clipped = torch.tensor([[[1.0, -2.0], [0.0, 0.0]]], device=device)
    masked_heads = []
    for offset in (1.0, 2.0, 3.0):
        prediction = masked_clipped + offset
        prediction[0, 1, :] = float("nan")
        masked_heads.append(prediction.detach().requires_grad_(True))
    masked_numerators = deep_supervision_wmse_numerators(
        tuple(masked_heads), masked_target, masked_rows
    )
    _close(
        masked_numerators,
        torch.tensor([3.0, 12.0, 27.0], device=device),
        "card04 masked nonfinite rows",
    )
    masked_loss = deep_supervision_scaled_loss(
        masked_numerators, torch.tensor(3.0, device=device), world_size=1
    )
    masked_gradients = torch.autograd.grad(masked_loss, masked_heads)
    for gradient in masked_gradients:
        _close(
            gradient[:, 1, :],
            torch.zeros_like(gradient[:, 1, :]),
            "card04 masked-row gradient",
        )

    empty_target = torch.full((1, 2, 2), float("nan"), device=device)
    empty_mask = torch.zeros((1, 2), dtype=torch.bool, device=device)
    empty_heads = tuple(
        torch.full_like(empty_target, float("nan")).requires_grad_(True) for _ in range(3)
    )
    empty_numerators = deep_supervision_wmse_numerators(empty_heads, empty_target, empty_mask)
    empty_loss = deep_supervision_scaled_loss(
        empty_numerators, torch.tensor(0.0, device=device), world_size=1
    )
    _close(empty_numerators, torch.zeros(3, device=device), "card04 empty numerators")
    _close(empty_loss.reshape(1), torch.zeros(1, device=device), "card04 empty loss")
    empty_gradients = torch.autograd.grad(empty_loss, empty_heads)
    for gradient in empty_gradients:
        _close(gradient, torch.zeros_like(gradient), "card04 empty gradient")
    return {
        "card04_manual_loss": float(loss.detach().item()),
        "card04_max_gradient_gap": max(gradient_gaps),
        "card04_masked_loss": float(masked_loss.detach().item()),
        "card04_empty_loss": float(empty_loss.detach().item()),
    }


def _manual_pairwise_edges(device: torch.device) -> dict[str, float | int]:
    """Check lag, threshold, clipping, warmup mask, padding, ties and chunk edges."""
    target = torch.zeros((2, 140, 2), dtype=torch.float32, device=device)
    prediction = torch.zeros_like(target)
    mask = torch.ones((2, 140), dtype=torch.bool, device=device)
    mask[0, :99] = False
    target[0, 99, 0] = 1.5
    target[0, 131, 0] = 3.0  # Clips to2.0, leaving the exact eligible delta0.5.
    prediction[0, 99, 0] = 1.0
    prediction[0, 131, 0] = 0.0  # Wrong order gives softplus(+1).
    target[0, 100, 0] = 1.0
    target[0, 132, 0] = 1.49  # Clipped delta0.49 is below threshold.
    target[0, 98, 0] = float("nan")
    target[0, 130, 0] = 2.0  # Large delta is excluded by the row98 warmup mask.
    prediction[0, 98, 0] = float("nan")
    target[1, :, :] = float("nan")  # Padded sequence has no selected rows.
    prediction[1, :, :] = float("nan")
    mask[1, :] = False
    prediction.requires_grad_(True)

    denominator = pairwise_window_denominators(target, mask)
    sums = pairwise_objective_sums(prediction, target, mask)
    expected_weight = math.sqrt(3.0)
    expected_denominator = torch.tensor([expected_weight, 0.0], device=device)
    _close(denominator, expected_denominator, "card06 clipped pair denominators")
    _close(sums.weight_denominator, expected_denominator, "card06 numerator denominator parity")
    _close(
        sums.loss_numerator,
        torch.tensor([expected_weight * math.log1p(math.exp(1.0)), 0.0], device=device),
        "card06 manual loss numerator",
    )
    _close(
        sums.eligible_count.to(torch.float32),
        torch.tensor([1.0, 0.0], device=device),
        "card06 eligible pair counts",
    )
    _close(
        sums.concordant_count.to(torch.float32),
        torch.zeros(2, device=device),
        "card06 concordant pair counts",
    )
    scaled_loss = pairwise_scaled_loss(sums, denominator, world_size=1)
    expected_loss = torch.tensor(
        PAIRWISE_COEFFICIENT * 0.5 * math.log1p(math.exp(1.0)), device=device
    )
    _close(scaled_loss.reshape(1), expected_loss.reshape(1), "card06 empty-target mean loss")
    gradient = torch.autograd.grad(scaled_loss, prediction)[0]
    expected_gradient = torch.zeros_like(prediction)
    signed_gradient = 0.5 * PAIRWISE_COEFFICIENT / (1.0 + math.exp(-1.0))
    expected_gradient[0, 131, 0] = -signed_gradient
    expected_gradient[0, 99, 0] = signed_gradient
    _close(gradient, expected_gradient, "card06 pair gradient and masked padding")

    cross_target = torch.zeros((1, 64, 2), dtype=torch.float32, device=device)
    cross_mask = torch.ones((1, 64), dtype=torch.bool, device=device)
    cross_target[0, 0, 0] = 0.5
    cross_target[0, 32, 0] = 1.0
    cross_prediction = torch.zeros_like(cross_target, requires_grad=True)
    chunk_losses = []
    chunk_weights = []
    chunk_counts = []
    for start, stop in ((0, 32), (32, 64)):
        chunk_target = cross_target[:, start:stop]
        chunk_mask = cross_mask[:, start:stop]
        chunk_prediction = cross_prediction[:, start:stop]
        local_weight = pairwise_window_denominators(chunk_target, chunk_mask)
        chunk_sums = pairwise_objective_sums(chunk_prediction, chunk_target, chunk_mask)
        chunk_weights.append(local_weight)
        chunk_counts.append(chunk_sums.eligible_count)
        chunk_losses.append(pairwise_scaled_loss(chunk_sums, local_weight, world_size=1))
    _close(
        torch.stack(chunk_weights).sum(dim=0),
        torch.zeros(2, device=device),
        "card06 no cross-chunk denominator",
    )
    _close(
        torch.stack(chunk_counts).sum(dim=0).to(torch.float32),
        torch.zeros(2, device=device),
        "card06 no cross-chunk pairs",
    )
    edge_loss = torch.stack(chunk_losses).sum()
    edge_gradient = torch.autograd.grad(edge_loss, cross_prediction)[0]
    _close(edge_loss.reshape(1), torch.zeros(1, device=device), "card06 cross-chunk loss")
    _close(edge_gradient, torch.zeros_like(edge_gradient), "card06 cross-chunk gradient")
    return {
        "card06_manual_loss": float(scaled_loss.detach().item()),
        "card06_eligible_t0": int(sums.eligible_count[0].item()),
        "card06_eligible_t1": int(sums.eligible_count[1].item()),
        "card06_cross_chunk_pairs": int(torch.stack(chunk_counts).sum().item()),
    }


def run_objective_checks(device: torch.device) -> dict[str, float | int | str]:
    """Run deterministic receiver checks for manual objectives and DDP normalization.

    This function creates only small synthetic tensors on `device`. It does not
    construct a model or touch training data. Results are returned only if every
    scalar, count, and gradient assertion passes at 1e-5 tolerance.
    """
    report: dict[str, float | int | str] = {}
    report.update(_manual_deep_supervision(device))
    report.update(_manual_pairwise_edges(device))
    report.update(check_distributed_equivalence(device))
    report["objective_checks"] = "passed"
    return report
