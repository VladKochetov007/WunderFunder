"""Server-side DDP normalization probes for PR81 auxiliary objectives."""

from __future__ import annotations

import torch
from torch import Tensor

from .auxiliary_objectives import (
    deep_supervision_scaled_loss,
    deep_supervision_wmse_numerators,
    pairwise_objective_sums,
    pairwise_scaled_loss,
    pairwise_window_denominators,
)

TOLERANCE = 1e-5


def _close(actual: Tensor, expected: Tensor, label: str) -> float:
    """Require finite allclose tensors and return their maximum absolute gap."""
    if not bool(torch.isfinite(actual).all()):
        raise AssertionError(f"{label} produced nonfinite values")
    if actual.shape != expected.shape:
        raise AssertionError(f"{label} shape differs: {actual.shape} != {expected.shape}")
    if not torch.allclose(actual, expected, atol=TOLERANCE, rtol=TOLERANCE):
        difference = float((actual - expected).abs().max().detach().item())
        raise AssertionError(f"{label} differs by {difference:.8g}")
    return float((actual - expected).abs().max().detach().item()) if actual.numel() else 0.0


def _primary_weight(target: Tensor, mask: Tensor) -> Tensor:
    """Compute clipped absolute-label mass for a synthetic primary-mask fixture."""
    clipped = torch.where(mask.unsqueeze(-1), target.float(), torch.zeros_like(target))
    return (clipped.clamp(-2.0, 2.0).abs() * mask.unsqueeze(-1)).sum()


def _deep_rank_chunks(
    device: torch.device,
) -> list[list[tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]]]:
    """Create four unequal-primary-mass microchunks for each of two ranks."""
    ranks = []
    for rank in range(2):
        chunks = []
        for microchunk in range(4):
            base = 0.2 + rank * 0.5 + microchunk * 0.15
            target = torch.tensor(
                [[[base, -2.5 * base], [base + 0.2, -base]]],
                dtype=torch.float32,
                device=device,
            )
            mask = torch.tensor(
                [[True, (rank + microchunk) % 3 != 0]],
                dtype=torch.bool,
                device=device,
            )
            clipped = target.clamp(-2.0, 2.0)
            heads = tuple(
                (clipped + 0.1 * (1 + rank + microchunk + head)).detach().requires_grad_(True)
                for head in range(3)
            )
            chunks.append((target, mask, heads))
        ranks.append(chunks)
    return ranks


def _pair_rank_chunks(device: torch.device) -> list[list[tuple[Tensor, Tensor, Tensor]]]:
    """Create four unequal-per-target-pair-mass chunks for each rank."""
    ranks = []
    for rank in range(2):
        chunks = []
        for microchunk in range(4):
            target = torch.zeros((1, 64, 2), dtype=torch.float32, device=device)
            prediction = torch.zeros_like(target)
            mask = torch.ones((1, 64), dtype=torch.bool, device=device)
            base0 = 0.25 + 0.1 * microchunk + 0.3 * rank
            delta0 = 0.5 + 0.1 * rank
            base1 = 0.15 + 0.08 * microchunk + 0.2 * rank
            target[0, 0, 0] = base0
            target[0, 32, 0] = base0 + delta0
            target[0, 1, 1] = -base1
            target[0, 33, 1] = -(base1 + 0.5)
            margin0 = -0.2 if (rank + microchunk) % 2 == 0 else 0.3
            margin1 = -0.25 if (rank + microchunk) % 2 == 0 else 0.2
            prediction[0, 0, 0] = 0.05 * microchunk
            prediction[0, 32, 0] = prediction[0, 0, 0] + margin0
            prediction[0, 1, 1] = -0.03 * microchunk
            prediction[0, 33, 1] = prediction[0, 1, 1] + margin1
            chunks.append((target, mask, prediction.detach().requires_grad_(True)))
        ranks.append(chunks)
    return ranks


def _check_deep_supervision_ddp(device: torch.device) -> tuple[float, float]:
    """Compare card04 full-window loss/gradients to two-rank accumulation."""
    ranks = _deep_rank_chunks(device)
    chunks = [chunk for rank_chunks in ranks for chunk in rank_chunks]
    global_primary = torch.stack(
        [_primary_weight(target, mask) for target, mask, _ in chunks]
    ).sum()
    rank_masses = [
        torch.stack([_primary_weight(target, mask) for target, mask, _ in local]).sum()
        for local in ranks
    ]
    if torch.allclose(rank_masses[0], rank_masses[1], atol=TOLERANCE, rtol=TOLERANCE):
        raise AssertionError("card04 fixture must have unequal rank masses")

    rank_losses = []
    for local in ranks:
        local_losses = [
            deep_supervision_scaled_loss(
                deep_supervision_wmse_numerators(heads, target, mask),
                global_primary,
                world_size=2,
            )
            for target, mask, heads in local
        ]
        rank_losses.append(torch.stack(local_losses).sum())
    distributed_loss = torch.stack(rank_losses).sum() / 2.0

    all_heads = tuple(heads[i] for _, _, heads in chunks for i in range(3))
    full_target = torch.cat([target for target, _, _ in chunks], dim=0)
    full_mask = torch.cat([mask for _, mask, _ in chunks], dim=0)
    full_heads = tuple(torch.cat([heads[i] for _, _, heads in chunks], dim=0) for i in range(3))
    reference_loss = deep_supervision_scaled_loss(
        deep_supervision_wmse_numerators(full_heads, full_target, full_mask),
        global_primary,
        world_size=1,
    )
    _close(distributed_loss.reshape(1), reference_loss.reshape(1), "card04 DDP loss")
    distributed_grads = torch.autograd.grad(distributed_loss, all_heads)
    reference_grads = torch.autograd.grad(reference_loss, all_heads)
    gap = max(
        _close(actual, expected, "card04 DDP/full-window gradient")
        for actual, expected in zip(distributed_grads, reference_grads, strict=True)
    )
    return float(distributed_loss.detach().item()), gap


def _check_pairwise_ddp(device: torch.device) -> tuple[float, float]:
    """Compare card06 global pair-mass loss and gradients across ranks/chunks."""
    ranks = _pair_rank_chunks(device)
    chunks = [chunk for rank_chunks in ranks for chunk in rank_chunks]
    rank_masses = [
        torch.stack([pairwise_window_denominators(target, mask) for target, mask, _ in local]).sum(
            dim=0
        )
        for local in ranks
    ]
    if torch.allclose(rank_masses[0], rank_masses[1], atol=TOLERANCE, rtol=TOLERANCE):
        raise AssertionError("card06 fixture must have unequal rank masses")
    global_denominator = torch.stack(rank_masses).sum(dim=0)

    rank_losses = []
    for local in ranks:
        local_losses = [
            pairwise_scaled_loss(
                pairwise_objective_sums(prediction, target, mask),
                global_denominator,
                world_size=2,
            )
            for target, mask, prediction in local
        ]
        rank_losses.append(torch.stack(local_losses).sum())
    distributed_loss = torch.stack(rank_losses).sum() / 2.0

    all_predictions = tuple(prediction for _, _, prediction in chunks)
    full_target = torch.cat([target for target, _, _ in chunks], dim=0)
    full_mask = torch.cat([mask for _, mask, _ in chunks], dim=0)
    full_prediction = torch.cat([prediction for _, _, prediction in chunks], dim=0)
    full_denominator = pairwise_window_denominators(full_target, full_mask)
    _close(global_denominator, full_denominator, "card06 summed pair denominator")
    reference_loss = pairwise_scaled_loss(
        pairwise_objective_sums(full_prediction, full_target, full_mask),
        full_denominator,
        world_size=1,
    )
    _close(distributed_loss.reshape(1), reference_loss.reshape(1), "card06 DDP loss")
    distributed_grads = torch.autograd.grad(distributed_loss, all_predictions)
    reference_grads = torch.autograd.grad(reference_loss, all_predictions)
    gap = max(
        _close(actual, expected, "card06 DDP/full-window gradient")
        for actual, expected in zip(distributed_grads, reference_grads, strict=True)
    )
    return float(distributed_loss.detach().item()), gap


def check_distributed_equivalence(device: torch.device) -> dict[str, float | int]:
    """Assert loss and gradient parity for two ranks and four microchunks each."""
    deep_loss, deep_gap = _check_deep_supervision_ddp(device)
    pair_loss, pair_gap = _check_pairwise_ddp(device)
    return {
        "ddp_deep_supervision_loss": deep_loss,
        "ddp_deep_supervision_gradient_gap": deep_gap,
        "ddp_pairwise_loss": pair_loss,
        "ddp_pairwise_gradient_gap": pair_gap,
        "ddp_microchunks_per_rank": 4,
        "ddp_ranks": 2,
    }
