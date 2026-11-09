from __future__ import annotations

import pytest
import torch

from wnn.gmetric import GlobalAccumulator
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_ensemble_config import MIXTURE_MODE
from wnn.training.sequence_ensemble_correlation import weighted_microbatch_correlation
from wnn.training.sequence_objectives import (
    objective_numerators,
    scaled_objective,
    window_denominators,
)


def context() -> DistributedContext:
    """Run the exact FP32 loss on CPU without constructing or fitting a model."""
    return DistributedContext(0, 0, 1, torch.device("cpu"), False)


def official(y: torch.Tensor, p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Use the unchanged float64 official centered scorer as an independent known answer."""
    accumulator = GlobalAccumulator()
    accumulator.add(y.detach().numpy(), p.detach().numpy(), mask.numpy())
    score = accumulator.result()
    return torch.tensor([score["t0"], score["t1"]], dtype=torch.float32)


@pytest.mark.parametrize("near_constant", [False, True])
def test_centered_fp32_correlations_match_official_clipping_and_target_mean(
    near_constant: bool,
) -> None:
    y = torch.tensor(
        [[[-3.0, 0.2], [-0.7, 1.3], [0.4, -0.5], [1.8, -3.0], [0.9, 0.7], [-0.3, 0.8]]]
    )
    p = torch.tensor([[[-2.4, 0.4], [0.1, 3.0], [0.2, -0.7], [1.1, -1.0], [0.7, 0.8], [0.8, 0.2]]])
    if near_constant:
        p = 0.5 + 2**-20 * torch.arange(p.numel()).reshape_as(p)
    p.requires_grad_()
    mask = torch.tensor([[True, True, True, True, False, True]])
    rho, loss = weighted_microbatch_correlation(p, y, mask, context())
    torch.testing.assert_close(rho, official(y, p, mask), atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(loss, 1 - rho.mean())
    loss.backward()
    assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("case", ["empty", "zero_labels", "constant_labels", "constant_prediction"])
def test_empty_and_constant_semantics_have_finite_zero_correlation_gradients(case: str) -> None:
    y = torch.tensor([[[-1.0, 0.4], [0.5, -0.3], [1.2, 0.8]]])
    p = torch.tensor([[[0.3, 0.8], [-0.2, 0.6], [0.7, -0.1]]], requires_grad=True)
    mask = torch.ones((1, 3), dtype=torch.bool)
    if case == "empty":
        mask.zero_()
    elif case == "zero_labels":
        y.zero_()
    elif case == "constant_labels":
        y.fill_(0.4)
    else:
        p = torch.full_like(p, 0.5, requires_grad=True)
    rho, loss = weighted_microbatch_correlation(p, y, mask, context())
    assert torch.equal(rho, torch.zeros(2))
    assert loss.item() == (0 if case == "empty" else 1)
    loss.backward()
    assert torch.equal(p.grad, torch.zeros_like(p))


def test_selected_nonfinite_rejected_before_clipping_even_at_zero_weight() -> None:
    p, y = torch.zeros((1, 2, 2)), torch.zeros((1, 2, 2))
    p[0, 0, 0] = float("inf")
    mask = torch.ones((1, 2), dtype=torch.bool)
    with pytest.raises(FloatingPointError):
        weighted_microbatch_correlation(p, y, mask, context())
    mask[0, 0] = False
    rho, loss = weighted_microbatch_correlation(p.requires_grad_(), y, mask, context())
    assert torch.isfinite(rho).all() and loss.item() == 1


def test_package_uses_unscaled_mixture_wmse_and_ignores_masked_nans() -> None:
    p = torch.tensor([[[3.0, -0.4], [0.8, -0.7], [float("nan"), float("nan")]]], requires_grad=True)
    y = torch.tensor([[[0.6, -0.8], [1.1, 0.5], [float("nan"), float("nan")]]])
    reference = torch.tensor([[[-0.5, 0.2], [0.3, -0.6], [float("nan"), float("nan")]]])
    mask = torch.tensor([[True, True, False]])
    hints = torch.empty((1, 3, 0))
    nums = objective_numerators(
        p,
        hints,
        y,
        reference,
        hints,
        mask,
        blend_objective_alpha=0.1,
        objective_mode=MIXTURE_MODE,
        context=context(),
    )
    safe_y = torch.where(mask[..., None], y, 0).clamp(-2, 2)
    mixture = 0.9 * reference[:, :2] + 0.1 * p[:, :2]
    expected = ((mixture - safe_y[:, :2]).square() * safe_y[:, :2].abs()).sum()
    torch.testing.assert_close(nums.label, expected)
    denom = torch.tensor([safe_y.abs().sum(), 0.0])
    loss = scaled_objective(nums, denom, 1, 0, 0, microchunks=4)
    torch.testing.assert_close(loss, 0.5 * expected / denom[0] + 0.5 * nums.correlation_loss / 4)
    loss.backward()
    assert torch.isfinite(p.grad).all() and torch.equal(
        p.grad[:, 2], torch.zeros_like(p.grad[:, 2])
    )


def test_mixture_is_formed_before_prediction_clipping() -> None:
    p = torch.tensor([[[5.0, -5.0], [-5.0, 5.0], [0.0, 0.2]]])
    reference = torch.tensor([[[-0.2, 0.1], [0.4, -0.3], [0.8, -0.5]]])
    y = torch.tensor([[[1.0, -0.8], [-0.4, 0.7], [0.8, 0.3]]])
    mask = torch.ones((1, 3), dtype=torch.bool)
    rho, _ = weighted_microbatch_correlation(0.9 * reference + 0.1 * p, y, mask, context())
    wrong, _ = weighted_microbatch_correlation(
        0.9 * reference.clamp(-2, 2) + 0.1 * p.clamp(-2, 2), y, mask, context()
    )
    assert not torch.allclose(rho, wrong, atol=0.001)


def test_window_denominator_uses_both_prescribed_target_weights() -> None:
    y = torch.tensor([[[3.0, -0.5], [0.2, -0.8]]])
    hints = torch.empty((1, 2, 0))
    assert window_denominators(y, torch.tensor([[True, False]]), hints).tolist() == [2.5, 0]
