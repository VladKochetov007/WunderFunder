from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wnn.splits import CampaignSplit
from wnn.training.sequence_blend import blend_loss_metrics, compose_prediction, fit_reference
from wnn.training.sequence_blend_prediction import prediction_reference
from wnn.training.sequence_config import load_config
from wnn.training.sequence_objectives import (
    objective_numerators,
    scaled_objective,
    window_denominators,
)
from wnn.training.sequence_predict import load_endpoint


def _numerators(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    empty = prediction.new_empty((*prediction.shape[:-1], 0))
    return objective_numerators(
        prediction,
        empty,
        target,
        reference,
        empty,
        mask,
        blend_objective_alpha=alpha,
    ).label


def test_blend_uses_original_label_weights_and_does_not_clip_parts() -> None:
    """Large base/head values may cancel; only the original target is loss-clipped."""
    prediction = torch.tensor([[[30.0, -30.0], [9.0, 8.0]]], requires_grad=True)
    reference = torch.tensor([[[-3.0, 3.0], [-6.0, 5.0]]], requires_grad=True)
    target = torch.tensor([[[4.0, -0.5], [2.0, -2.0]]])
    mask = torch.tensor([[True, False]])
    alpha = 0.1
    loss = _numerators(prediction, reference, target, mask, alpha)
    clipped = target.clamp(-2, 2)
    expected = (
        ((0.9 * reference.detach() + 0.1 * prediction - clipped) / alpha).square()
        * clipped.abs()
        * mask[..., None]
    ).sum()
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    gradient, reference_gradient = torch.autograd.grad(
        loss, (prediction, reference), allow_unused=True
    )
    assert reference_gradient is None
    assert torch.count_nonzero(gradient[:, 1:]) == 0
    assert not torch.equal(
        compose_prediction(prediction, reference, alpha), prediction.clamp(-2, 2)
    )


def test_alpha_squared_compensation_matches_label_objective_curvature() -> None:
    """The declared scaling keeps the Hessian wrt the head equal to standalone WMSE."""
    target = torch.tensor([[[1.3, -0.7]]])
    reference = torch.tensor([[[1.0, -1.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    curvatures = []
    for alpha in (0.0, 0.1, 1.0):
        head = torch.tensor([[[0.2, -0.3]]], requires_grad=True)
        first = torch.autograd.grad(
            _numerators(head, reference, target, mask, alpha), head, create_graph=True
        )[0]
        curvatures.append(torch.autograd.grad(first.sum(), head)[0])
    torch.testing.assert_close(curvatures[1], curvatures[0])
    torch.testing.assert_close(curvatures[2], curvatures[0])


def test_default_loss_preserves_exact_historical_numerators() -> None:
    """Omitting the opt-in must retain the existing loss arithmetic and KD outputs."""
    prediction = torch.tensor([[[0.2, 0.3], [-0.7, 0.9]]])
    teacher = prediction + 0.1
    target = torch.tensor([[[2.4, -3.0], [0.0, 0.5]]])
    hints = torch.ones(1, 2, 3)
    hint_target = hints * 0.4
    mask = torch.tensor([[True, False]])
    nums = objective_numerators(prediction, hints, target, teacher, hint_target, mask)
    weight = target.clamp(-2, 2).abs() * mask[..., None]
    assert torch.equal(nums.label, ((prediction - target.clamp(-2, 2)).square() * weight).sum())
    assert torch.equal(
        nums.distillation, ((prediction - teacher.clamp(-2, 2)).square() * weight).sum()
    )
    assert torch.equal(nums.hints, ((hints - hint_target).square() * mask[..., None]).sum())


def test_distributed_unequal_mass_and_chunk_accumulation_match_global_gradient() -> None:
    """World-size scaling covers uneven labels, masked rows and a zero-mass rank."""
    target = torch.tensor([[[0.1, 0.2], [-2.5, 1.9]], [[0.7, -1.0], [0.2, 0.4]]])
    reference = target * 0.7
    mask = torch.tensor([[True, True], [False, False]])
    empty = torch.empty(2, 2, 0)
    denominator = window_denominators(target, mask, empty)
    head = torch.zeros_like(target, requires_grad=True)
    full = _numerators(head, reference, target, mask, 0.1) / denominator[0]
    expected = torch.autograd.grad(full, head)[0]
    head = torch.zeros_like(target, requires_grad=True)
    loss = torch.zeros(())
    for rank in range(2):
        for row in range(2):
            sl = (slice(rank, rank + 1), slice(row, row + 1))
            nums = objective_numerators(
                head[sl],
                empty[sl],
                target[sl],
                reference[sl],
                empty[sl],
                mask[sl],
                blend_objective_alpha=0.1,
            )
            loss = loss + scaled_objective(nums, denominator, 2, 0.0, 0.0) / 2
    observed = torch.autograd.grad(loss, head)[0]
    torch.testing.assert_close(observed, expected)
    assert torch.count_nonzero(observed[1]) == 0


def test_config_omission_and_explicit_zero_retain_legacy_identity(tmp_path: Path) -> None:
    """No extra default key appears in the normalized historical recipe."""
    path = tmp_path / "config.json"
    path.write_text("{}")
    historical = load_config(path)
    path.write_text(json.dumps({"blend_objective_alpha": 0.0}))
    assert load_config(path) == historical
    assert "blend_objective_alpha" not in historical
    assert blend_loss_metrics(historical, 12.0) == {}
    assert blend_loss_metrics({"blend_objective_alpha": 0.1}, 12.0)[
        "composite_wmse"
    ] == pytest.approx(0.12)


@pytest.mark.parametrize("value", [True, -0.1, 1.1, "0.1", float("nan"), float("inf")])
def test_invalid_blend_configuration_rejected(tmp_path: Path, value: object) -> None:
    """Malformed coefficients must fail before a model or dataset is loaded."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"blend_objective_alpha": value}))
    with pytest.raises(ValueError, match="blend_objective_alpha"):
        load_config(path)


@pytest.mark.parametrize(
    "extra",
    [
        {"kd_weight": 1},
        {"hint_weight": 1, "model": {"hint_dim": 2}},
        {"model": {"architecture": "gru"}},
    ],
)
def test_blend_cannot_mix_with_distillation_or_recurrent_architecture(
    tmp_path: Path, extra: dict[str, object]
) -> None:
    """The initial mechanism excludes unrelated KD, hidden hints and GRU treatments."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"blend_objective_alpha": 0.1} | extra))
    with pytest.raises(ValueError, match="blend objective"):
        load_config(path)


def test_composite_checkpoint_cannot_be_exported_as_standalone(tmp_path: Path) -> None:
    """A treatment head alone must not silently become a scored endpoint or KD cache."""
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"checkpoint_schema": "wnn.sequence-fit.v1", "config": {"blend_objective_alpha": 0.1}},
        checkpoint,
    )
    with pytest.raises(ValueError, match="reference-bound export"):
        load_endpoint(checkpoint, torch.device("cpu"))


def test_fit_cache_arguments_are_unambiguous(tmp_path: Path) -> None:
    """The fit cache is mandatory for treatment, prohibited for controls and separate from KD."""
    split = CampaignSplit(tmp_path / "split.json", "a" * 64, 0, 1, (0,), (), (0,), ())
    fitting = [("train", 0)]
    with pytest.raises(ValueError, match="requires --reference-cache"):
        fit_reference({"blend_objective_alpha": 0.1}, None, None, split, fitting)
    with pytest.raises(ValueError, match="controls must omit"):
        fit_reference({}, tmp_path, None, split, fitting)
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit_reference({"blend_objective_alpha": 0.1}, tmp_path, tmp_path, split, fitting)
    assert fit_reference({}, None, None, split, fitting) is None


def test_composite_fit_export_and_missing_reference_rejected(tmp_path: Path) -> None:
    """Unimplemented KD export cannot silently produce a cache of head-only predictions."""
    split = CampaignSplit(tmp_path / "split.json", "a" * 64, 0, 1, (0,), (), (0,), ())
    payload = {"config": {"blend_objective_alpha": 0.1, "model": {"hint_dim": 0}}}
    with pytest.raises(ValueError, match="require --reference-cache"):
        prediction_reference(None, None, payload, split, [("valid", 0)], "evaluation")
    with pytest.raises(ValueError, match="not implemented"):
        prediction_reference(tmp_path, 0.1, payload, split, [("train", 0)], "fit")
    with pytest.raises(ValueError, match="differs from the frozen"):
        prediction_reference(tmp_path, 0.2, payload, split, [("valid", 0)], "evaluation")
