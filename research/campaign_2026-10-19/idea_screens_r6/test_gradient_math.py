from __future__ import annotations

import torch
from gradient_math import add_weighted_pair, pair_summary, pooled_summary, target_loss_parts

from wnn.nn import weighted_mse_loss


def test_target_gradients_sum_to_canonical_wmse_gradient() -> None:
    """The shared joint denominator makes the two true target gradients additive."""
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    features = torch.randn(2, 11, 3)
    target = torch.randn(2, 11, 2) * 1.5
    scored = torch.rand(2, 11) > 0.25
    prediction = model(features)
    loss0, loss1, weight, _ = target_loss_parts(prediction, target, scored)
    joint = weighted_mse_loss(prediction, target, scored)
    assert float(weight) > 0.0
    assert torch.allclose(loss0 + loss1, joint, rtol=2e-6, atol=1e-7)
    parameters = tuple(model.parameters())
    gradient0 = torch.autograd.grad(loss0, parameters, retain_graph=True)
    gradient1 = torch.autograd.grad(loss1, parameters, retain_graph=True)
    joint_gradient = torch.autograd.grad(joint, parameters)
    for left, right, combined in zip(gradient0, gradient1, joint_gradient, strict=True):
        assert torch.allclose(left + right, combined, rtol=2e-5, atol=2e-6)


def test_weighted_crossproducts_reconstruct_final_pooled_pair() -> None:
    """Per-window norms and four running cross-products reconstruct pooled cosine."""
    windows = [
        (
            [torch.tensor([1.0, -2.0]), torch.tensor([0.5])],
            [torch.tensor([2.0, 1.0]), torch.tensor([-1.0])],
            3.0,
        ),
        (
            [torch.tensor([-0.5, 3.0]), torch.tensor([2.0])],
            [torch.tensor([1.5, -2.0]), torch.tensor([0.25])],
            2.0,
        ),
        (
            [torch.tensor([0.25, 1.0]), torch.tensor([-2.0])],
            [torch.tensor([-1.0, -0.5]), torch.tensor([3.0])],
            4.0,
        ),
    ]
    zero = (torch.zeros(2, dtype=torch.float64), torch.zeros(1, dtype=torch.float64))
    pooled0, pooled1 = zero, zero
    total_weight = 0.0
    audits = []
    for gradient0, gradient1, weight in windows:
        pooled0, pooled1, audit = add_weighted_pair(pooled0, pooled1, gradient0, gradient1, weight)
        audits.append(audit)
        total_weight += weight
    direct0 = tuple(sum(window[2] * window[0][i].double() for window in windows) for i in range(2))
    direct1 = tuple(sum(window[2] * window[1][i].double() for window in windows) for i in range(2))
    for actual, expected in zip(pooled0, direct0, strict=True):
        assert torch.equal(actual, expected)
    for actual, expected in zip(pooled1, direct1, strict=True):
        assert torch.equal(actual, expected)
    summary = pooled_summary(pooled0, pooled1, total_weight)
    direct_summary = pair_summary(direct0, direct1)
    assert abs(summary["pooled_gradient_cosine"] - direct_summary["gradient_cosine"]) < 1e-12
    assert len(audits) == 3
    assert set(audits[0]) == {
        "weighted_target0_norm_sq",
        "weighted_target1_norm_sq",
        "weighted_target_inner_product",
        "cross_target0_with_previous0",
        "cross_target0_with_previous1",
        "cross_target1_with_previous0",
        "cross_target1_with_previous1",
    }


def test_zero_target_gradient_is_retained_with_undefined_cosine() -> None:
    """A zero gradient remains in the statistics instead of being filtered out."""
    summary = pair_summary((torch.zeros(2),), (torch.tensor([1.0, -2.0]),))
    assert summary["target0_gradient_norm"] == 0.0
    assert summary["target1_gradient_norm"] > 0.0
    assert summary["gradient_cosine"] is None
    assert summary["target0_over_target1_norm"] == 0.0
