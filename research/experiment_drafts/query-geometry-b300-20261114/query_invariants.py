"""Independent algebraic invariants for the server-only query geometry probe."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from wnn.training.sequence_query_perturbation import perturb_query


def require(condition: bool, message: str) -> None:
    """Reject receiver admission when an observed invariant fails."""
    if not condition:
        raise ValueError(message)


def relative_error(actual: Tensor, expected: Tensor) -> float:
    """Measure a bounded error without amplifying values close to zero."""
    denominator = expected.abs().clamp_min(1.0)
    return float(((actual - expected).abs() / denominator).max())


def run_invariants(device: torch.device) -> dict[str, object]:
    """Check geometry using explicit two-sign averaging and independent identities."""
    torch.manual_seed(209401)
    query = torch.randn(2, 12, 64, 64, device=device, dtype=torch.float32)
    original = query.clone()
    pairs = query.reshape(2, 12, 64, 32, 2)
    plus = torch.ones(2, 12, 1, 32, device=device)
    cosine, sine = math.cos(0.35), math.sin(0.35)
    expected_energy = pairs.square().sum(-1)
    expected_displacement = 2 * (1 - cosine) * expected_energy
    outcomes = {}
    before = torch.cuda.get_rng_state(device)
    identity = perturb_query(query, "none")
    require(identity is query, "neutral path must return the original query")
    require(torch.equal(before, torch.cuda.get_rng_state(device)), "neutral path consumed RNG")
    for mode in ("angular", "radial"):
        positive = perturb_query(query, mode, signs=plus).reshape_as(pairs)
        negative = perturb_query(query, mode, signs=-plus).reshape_as(pairs)
        mean_error = relative_error((positive + negative) / 2, cosine * pairs)
        energy_error = relative_error(
            (positive.square().sum(-1) + negative.square().sum(-1)) / 2, expected_energy
        )
        displacement_error = relative_error(
            ((positive - pairs).square().sum(-1) + (negative - pairs).square().sum(-1)) / 2,
            expected_displacement,
        )
        require(max(mean_error, energy_error, displacement_error) <= 2e-6, "moment mismatch")
        if mode == "angular":
            norm_error = relative_error(positive.square().sum(-1), expected_energy)
            dot_error = relative_error((positive * pairs).sum(-1), cosine * expected_energy)
            # The sign matters: a reversed rotation could satisfy the moment identities.
            determinant = pairs[..., 0] * positive[..., 1] - pairs[..., 1] * positive[..., 0]
            direction_error = relative_error(determinant, sine * expected_energy)
        else:
            norm_error = relative_error(positive, (cosine + sine) * pairs)
            dot_error = relative_error(negative, (cosine - sine) * pairs)
            determinant = pairs[..., 0] * positive[..., 1] - pairs[..., 1] * positive[..., 0]
            direction_error = relative_error(determinant, torch.zeros_like(determinant))
        require(max(norm_error, dot_error, direction_error) <= 2e-6, "geometry mismatch")
        bf16 = query.to(torch.bfloat16)
        from_fp32 = perturb_query(bf16.float(), mode, signs=plus).to(torch.bfloat16)
        actual = perturb_query(bf16, mode, signs=plus)
        require(actual.dtype == torch.bfloat16 and torch.equal(actual, from_fp32), "dtype drift")
        torch.manual_seed(209401)
        rng = torch.cuda.get_rng_state(device)
        stochastic = perturb_query(query, mode)
        torch.cuda.set_rng_state(rng, device)
        replay = perturb_query(query, mode)
        require(torch.equal(stochastic, replay), "query RNG replay failed")
        changed = query.clone()
        changed[:, :, 32:] += 10
        torch.cuda.set_rng_state(rng, device)
        future = perturb_query(changed, mode)
        require(torch.equal(stochastic[:, :, :32], future[:, :, :32]), "future query leaked")
        # Repeated identical rows must see one shared pair sign within this call.
        repeated = query[:, :, :1].expand(-1, -1, 64, -1).clone()
        sampled = perturb_query(repeated, mode)
        require(torch.equal(sampled, sampled[:, :, :1].expand_as(sampled)), "signs vary by row")
        differentiable = query.detach().clone().requires_grad_()
        result = perturb_query(differentiable, mode, signs=plus)
        result.square().mean().backward()
        gradient = differentiable.grad
        require(
            gradient is not None
            and bool(torch.isfinite(gradient).all())
            and bool(gradient.abs().sum() > 0),
            "missing or nonfinite query gradient",
        )
        require(torch.equal(query, original), "helper mutated its query input")
        outcomes[mode] = {
            "mean_error": mean_error,
            "energy_error": energy_error,
            "displacement_error": displacement_error,
            "norm_error": norm_error,
            "dot_error": dot_error,
            "direction_error": direction_error,
            "bf16_cast_exact": True,
            "rng_replay_exact": True,
            "future_prefix_exact": True,
            "row_shared_signs": True,
            "input_unchanged": True,
            "gradient_finite_nonzero": True,
        }
    for bad in (torch.zeros_like(plus), plus[..., :1], plus * float("nan")):
        try:
            perturb_query(query, "angular", signs=bad)
        except (ValueError, TypeError):
            continue
        raise ValueError("malformed explicit signs were accepted")
    return outcomes
