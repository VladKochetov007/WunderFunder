from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from wnn.training.sequence_auxiliary import (
    auxiliary_denominators,
    auxiliary_loss_sums,
    build_auxiliary_targets,
)
from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_artifacts import epoch_summary
from wnn.training.sequence_loop import _backward_window
from wnn.training.sequence_loop_backprop import EpochResult
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_objectives import window_denominators
from wnn.training.sequence_predict import predict_sequence


def _tick_targets() -> torch.Tensor:
    ticks = torch.tensor(
        [[[-3, -1], [-2, -1], [-1, 0], [0, -1], [1, -1], [1, 2], [3, 1], [0, 0]]],
        dtype=torch.float32,
    )
    return ticks * 0.12543


def _manual_next_change(
    target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    difference = target[..., 0] - target[..., 1]
    rows = torch.zeros_like(difference)
    direction = torch.zeros_like(difference)
    available = torch.zeros_like(mask)
    for batch_index in range(target.shape[0]):
        changes = [
            index
            for index in range(1, target.shape[1])
            if difference[batch_index, index] != difference[batch_index, index - 1]
        ]
        for row_index in range(target.shape[1]):
            future = next((index for index in changes if index > row_index), target.shape[1])
            next_index = min(future, target.shape[1] - 1)
            rows[batch_index, row_index] = math.log1p(future - row_index)
            direction[batch_index, row_index] = float(
                difference[batch_index, next_index] > difference[batch_index, next_index - 1]
            )
            available[batch_index, row_index] = mask[batch_index, row_index] & (
                future < target.shape[1]
            )
    return rows, direction, available


def test_labels_and_losses_match_legacy_structure_and_next_change_equations() -> None:
    target = _tick_targets()
    mask = torch.tensor([[True, True, False, True, True, True, True, True]])
    labels = build_auxiliary_targets(target, mask, ["structure", "next_change"])
    assert labels is not None
    assert labels.structure is not None and labels.next_change is not None

    ticks = torch.round(target / 0.12543).long()
    t0, t1 = ticks[..., 0], ticks[..., 1]
    upward, downward = t0 >= 1, t1 >= 1
    spread = torch.where(upward, -t1 - 2, torch.where(downward, -t0 - 2, -(t0 + t1))).clamp_min(1)
    unit = torch.where(upward, t0 + spread, -t1)
    branch = torch.where(unit < 0, 0, torch.where(unit > spread, 2, 1))
    expected_spread = spread.clamp(max=12) - 1
    expected_classes = torch.zeros_like(spread)
    torch.testing.assert_close(labels.structure[0], expected_spread)
    torch.testing.assert_close(labels.structure[1], branch)
    torch.testing.assert_close(labels.structure[2], expected_classes)

    next_rows, next_direction, next_mask = _manual_next_change(target, mask)
    torch.testing.assert_close(labels.next_change[0], next_rows)
    torch.testing.assert_close(labels.next_change[1], next_direction)
    torch.testing.assert_close(labels.next_mask, next_mask)

    prediction = torch.linspace(-1.2, 1.1, 8 * 23).reshape(1, 8, 23).requires_grad_()
    sums = auxiliary_loss_sums(prediction, labels, 0, 8)
    spread_ce = F.cross_entropy(
        prediction[..., 2:14].reshape(-1, 12), expected_spread.reshape(-1), reduction="none"
    ).reshape(1, 8)
    branch_ce = F.cross_entropy(
        prediction[..., 14:17].reshape(-1, 3), branch.reshape(-1), reduction="none"
    ).reshape(1, 8)
    class_ce = F.cross_entropy(
        prediction[..., 17:21].reshape(-1, 4), expected_classes.reshape(-1), reduction="none"
    ).reshape(1, 8)
    expected_structure = ((spread_ce + branch_ce + class_ce) * mask).sum()
    regression = (prediction[..., 21] - (next_rows - 3.0) / 2.0).square()
    classification = F.binary_cross_entropy_with_logits(
        prediction[..., 22], next_direction, reduction="none"
    )
    expected_next = ((regression + classification) * next_mask).sum()
    torch.testing.assert_close(sums, torch.stack((expected_structure, expected_next)))

    denominators = auxiliary_denominators(labels, [(0, 3), (3, 8)], torch.device("cpu"))
    torch.testing.assert_close(
        denominators, torch.tensor([mask.sum(), next_mask.sum()], dtype=torch.float32)
    )
    sums.sum().backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad[..., 2:]).item() > 0


def test_auxiliary_gru_outputs_preserve_the_prediction_abi() -> None:
    model = build_sequence_model(
        {
            "architecture": "gru",
            "hidden": 8,
            "layers": 1,
            "heads": 1,
            "context_rows": 4,
            "hint_dim": 0,
            "auxiliary_heads": ["structure", "next_change"],
        }
    )
    model.eval()
    values = torch.randn(7, 112)
    prediction, _, _ = model(values.unsqueeze(0))
    assert prediction.shape == (1, 7, 23)
    exported = predict_sequence(model, values, chunk_rows=3, device=torch.device("cpu"))
    assert exported.shape == (7, 2)


def test_training_window_backpropagates_main_and_both_auxiliary_objectives() -> None:
    heads = ["structure", "next_change"]
    model = build_sequence_model(
        {
            "architecture": "gru",
            "hidden": 8,
            "layers": 1,
            "heads": 1,
            "context_rows": 4,
            "hint_dim": 0,
            "auxiliary_heads": heads,
        }
    )
    model.train()
    target = _tick_targets()
    mask = torch.ones((1, 8), dtype=torch.bool)
    teacher = torch.zeros_like(target)
    hints = torch.empty((1, 8, 0))
    values = torch.randn((1, 8, 112))
    auxiliary = build_auxiliary_targets(target, mask, heads)
    assert auxiliary is not None
    denominators = torch.cat(
        (
            window_denominators(target, mask, hints),
            auxiliary_denominators(auxiliary, [(0, 8)], torch.device("cpu")),
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    _, stats = _backward_window(
        model,
        optimizer,
        target,
        teacher,
        hints,
        mask,
        values,
        None,
        [(0, 8)],
        1,
        denominators,
        auxiliary,
        None,
        {
            "model": {"hint_dim": 0, "attention_backend": "math"},
            "precision_mode": "fp32",
            "kd_weight": 0.0,
            "hint_weight": 0.0,
            "aux_structure_weight": 0.3,
            "aux_next_change_weight": 0.3,
        },
        context,
    )
    assert stats.shape == (8,)
    assert model.head[2].weight.grad is not None
    assert model.head[2].weight.grad[:2].abs().sum() > 0
    assert model.head[2].weight.grad[2:21].abs().sum() > 0
    assert model.head[2].weight.grad[21:].abs().sum() > 0


def test_auxiliary_recipe_requires_a_matching_fresh_gru_contract(tmp_path: Path) -> None:
    path = tmp_path / "auxiliary.json"
    raw = {
        "aux_structure_weight": 0.3,
        "aux_next_change_weight": 0.3,
        "model": {
            "architecture": "gru",
            "hidden": 8,
            "layers": 1,
            "auxiliary_heads": ["structure", "next_change"],
        },
    }
    path.write_text(json.dumps(raw))
    config = load_config(path)
    assert config["model"]["auxiliary_heads"] == ["structure", "next_change"]

    raw["model"]["architecture"] = "transformer"
    raw["model"]["hidden"] = 32
    raw["model"]["heads"] = 4
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="fresh GRU"):
        load_config(path)

    raw["model"]["architecture"] = "gru"
    raw["model"]["hidden"] = 8
    raw["aux_next_change_weight"] = 0.0
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="coefficient"):
        load_config(path)


def test_epoch_summary_records_auxiliary_terms_and_weighted_objective() -> None:
    result = EpochResult(
        updates=1,
        complete=True,
        totals=torch.tensor([2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 4.0, 0.0], dtype=torch.float64),
        elapsed_seconds=1.0,
        auxiliary_totals=torch.tensor([6.0, 8.0, 2.0, 4.0], dtype=torch.float64),
    )
    summary = epoch_summary(
        result,
        {
            "kd_weight": 0.0,
            "hint_weight": 0.0,
            "aux_structure_weight": 0.3,
            "aux_next_change_weight": 0.3,
        },
    )
    assert summary["aux_structure_loss"] == pytest.approx(3.0)
    assert summary["aux_next_change_loss"] == pytest.approx(2.0)
    assert summary["total_loss"] == pytest.approx(0.5 + 0.3 * 3.0 + 0.3 * 2.0)
