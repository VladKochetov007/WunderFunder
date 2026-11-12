from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wnn.data import I0
from wnn.features import FEATURE_NAMES, causal_since_i0px_torch, event_features_torch
from wnn.training.sequence_config import load_config
from wnn.training.sequence_objectives import (
    objective_numerators,
    since_event_row_weights,
    window_denominators,
)


def _inputs(steps: int = 40) -> torch.Tensor:
    values = torch.zeros(1, steps, 112, dtype=torch.float32)
    values[:, 2:, I0.bid_px.start] = 1.0
    values[:, 2:, I0.ask_px.start] = 1.0
    return values


def test_causal_since_age_matches_event_feature_and_resets_at_price_event() -> None:
    values = _inputs()
    age = causal_since_i0px_torch(values)
    event_age = torch.expm1(event_features_torch(values)[..., FEATURE_NAMES.index("since_i0px")])

    torch.testing.assert_close(age, event_age)
    assert age[0, :4].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert age[0, 39].item() == 37.0


def test_since_weights_change_only_label_numerator_and_keep_metric_mass() -> None:
    values = _inputs(steps=4)
    weights = since_event_row_weights(values, tau=1.0, floor=0.25)
    assert weights[0].tolist() == [1.0, 0.5, 1.0, 0.5]

    prediction = torch.zeros(1, 4, 2)
    hint_prediction = torch.zeros(1, 4, 0)
    target = torch.tensor([[[1.0, -1.0], [2.0, -2.0], [1.0, 1.0], [-2.0, 2.0]]])
    teacher = torch.ones_like(target)
    hint_target = torch.zeros_like(hint_prediction)
    mask = torch.ones(1, 4, dtype=torch.bool)
    base = objective_numerators(
        prediction, hint_prediction, target, teacher, hint_target, mask
    )
    weighted = objective_numerators(
        prediction,
        hint_prediction,
        target,
        teacher,
        hint_target,
        mask,
        label_row_weights=weights,
    )

    expected_label = (
        target.abs() * weights[..., None] * prediction.sub(target).square()
    ).sum()
    assert float(weighted.label) == pytest.approx(float(expected_label))
    assert weighted.distillation == base.distillation
    assert float(window_denominators(target, mask, hint_target)[0]) == pytest.approx(12.0)

    with pytest.raises(ValueError, match="row weights"):
        objective_numerators(
            prediction,
            hint_prediction,
            target,
            teacher,
            hint_target,
            mask,
            label_row_weights=torch.full((1, 4), 2.0),
        )


def test_since_config_is_cold_labels_only_gru_and_validates_floor(tmp_path: Path) -> None:
    config_path = tmp_path / "since.json"
    config_path.write_text(
        json.dumps(
            {
                "since_weight": 32.0,
                "since_floor": 0.25,
                "model": {"architecture": "gru", "hidden": 8, "layers": 1},
            }
        )
    )
    config = load_config(config_path)
    assert config["since_weight"] == pytest.approx(32.0)
    assert config["since_floor"] == pytest.approx(0.25)

    config_path.write_text(
        json.dumps(
            {
                "since_weight": 32.0,
                "kd_weight": 0.1,
                "model": {"architecture": "gru", "hidden": 8, "layers": 1},
            }
        )
    )
    with pytest.raises(ValueError, match="labels-only GRU"):
        load_config(config_path)

    config_path.write_text(json.dumps({"since_floor": 0.0}))
    with pytest.raises(ValueError, match="since_floor"):
        load_config(config_path)
