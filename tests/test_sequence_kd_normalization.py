from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wnn.training.sequence_config import DEFAULTS, MODEL_DEFAULTS, load_config
from wnn.training.sequence_fit_artifacts import epoch_summary
from wnn.training.sequence_loop import EpochResult
from wnn.training.sequence_objectives import (
    ObjectiveNumerators,
    objective_numerators,
    objective_total,
    scaled_objective,
    window_denominators,
)
from wnn.training.sequence_provenance import canonical_sha256


def test_omission_preserves_legacy_config_identity(tmp_path: Path) -> None:
    """Old recipes must retain their exact normalized config and config hash."""
    recipe = tmp_path / "recipe.json"
    recipe.write_text("{}")
    config = load_config(recipe)
    legacy = DEFAULTS | {"model": MODEL_DEFAULTS | {"precision_mode": "fp32"}}
    assert "kd_loss_normalization" not in config
    assert config == legacy
    assert canonical_sha256(config) == canonical_sha256(legacy)
    recipe.write_text(json.dumps({"kd_loss_normalization": "weighted_mean"}))
    normalized = load_config(recipe)
    assert normalized["kd_loss_normalization"] == "weighted_mean"
    assert canonical_sha256(normalized) != canonical_sha256(config)


@pytest.mark.parametrize("value", [None, True, 1, "mean", [], {}])
def test_invalid_normalization_is_rejected(tmp_path: Path, value: object) -> None:
    """Invalid modes must fail at the config boundary instead of falling back."""
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"kd_loss_normalization": value}))
    with pytest.raises(ValueError, match="kd_loss_normalization"):
        load_config(recipe)


@pytest.mark.parametrize("kd_weight", [0.0, 0.25, 1.0, 3.0])
def test_normalized_prediction_gradient_matches_mixed_target(kd_weight: float) -> None:
    """KD changes the regression target without increasing its total loss mass."""
    prediction = torch.tensor([[[0.3, -0.7], [0.1, 0.9]]], requires_grad=True)
    hints = torch.tensor([[[0.2], [0.6]]], requires_grad=True)
    target = torch.tensor([[[0.5, -3.0], [0.8, 1.0]]])
    teacher = torch.tensor([[[0.9, 0.5], [3.0, -0.6]]])
    hint_target = torch.zeros_like(hints)
    mask = torch.tensor([[True, False]])
    denominators = window_denominators(target, mask, hint_target)
    numerators = objective_numerators(prediction, hints, target, teacher, hint_target, mask)
    fitted = scaled_objective(numerators, denominators, 2, kd_weight, 0.3, "weighted_mean")
    clipped = target.clamp(-2, 2)
    mixed_target = (clipped + kd_weight * teacher.clamp(-2, 2)) / (1 + kd_weight)
    weight = clipped.abs() * mask[..., None]
    reference = (
        2 * ((prediction - mixed_target).square() * weight).sum() / weight.sum()
        + 2 * 0.3 * (hints.square() * mask[..., None]).sum() / mask.sum()
    )
    actual_grad = torch.autograd.grad(fitted, (prediction, hints), retain_graph=True)
    expected_grad = torch.autograd.grad(reference, (prediction, hints))
    for actual, expected in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, expected)


def test_legacy_additive_values_and_hint_weight_are_preserved() -> None:
    """Normalization affects the label/KD block only; legacy math stays additive."""
    sums = ObjectiveNumerators(torch.tensor(8.0), torch.tensor(20.0), torch.tensor(12.0))
    denominators = torch.tensor([4.0, 3.0])
    assert scaled_objective(sums, denominators, 1, 1, 0.5).item() == 9
    assert scaled_objective(sums, denominators, 1, 1, 0.5, "weighted_mean").item() == 5.5
    assert objective_total(2, 5, 4, 1, 0.5) == 9
    assert objective_total(2, 5, 4, 1, 0.5, "weighted_mean") == 5.5
    assert scaled_objective(sums, denominators, 1, 0, 0.5).item() == 4
    assert scaled_objective(sums, denominators, 1, 0, 0.5, "weighted_mean").item() == 4


def test_epoch_telemetry_reports_normalized_training_objective() -> None:
    """Run metrics must describe the actual fitted loss, not the additive legacy loss."""
    result = EpochResult(1, True, torch.tensor([8, 20, 12, 1, 2, 2, 4, 3]), 1.0)
    config = {"kd_weight": 1.0, "hint_weight": 0.5, "kd_loss_normalization": "weighted_mean"}
    summary = epoch_summary(result, config)
    assert summary["label_loss"] == 2
    assert summary["kd_loss"] == 5
    assert summary["hint_loss"] == 4
    assert summary["total_loss"] == 5.5
