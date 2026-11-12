from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

from wnn.training.sequence_complement_monitor import ComplementMonitor
from wnn.training.sequence_config import load_config
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_ensemble_config import (
    STANDALONE_LABELS_KD_MODE,
    STANDALONE_LABELS_MODE,
    objective_contract,
    requires_teacher_cache,
)
from wnn.training.sequence_ensemble_reporting import objective_metrics
from wnn.training.sequence_fit_admission import validate_fit_reference_inputs
from wnn.training.sequence_monitor_events import MONITOR_EVENT_FILENAME
from wnn.training.sequence_objectives import (
    global_window_denominators,
    objective_numerators,
    scaled_objective,
    window_denominators,
)
from wnn.training.sequence_reference_pins import PRIMARY_SPLIT_SHA256


def _config(
    *, seed: int = 203501, kd_weight: float = 0.0, since_weight: float = 0.0
) -> dict[str, Any]:
    """Build one cold, two-target GRU recipe for loader-level validation."""
    return {
        "epochs": 6,
        "batch_size": 32,
        "accumulate_chunks": 1,
        "chunk_rows": 1000,
        "seed": seed,
        "lr": 0.0003,
        "weight_decay": 0.01,
        "optimizer": "adamw",
        "since_weight": since_weight,
        "since_floor": 0.25,
        "grad_clip": 1.0,
        "warmup_fraction": 0.05,
        "schedule": "warmup_cosine_v2",
        "min_lr_ratio": 0.1,
        "kd_weight": kd_weight,
        "hint_weight": 0.0,
        "normalization_sequences": 10325,
        "training_mask": "all_after_warmup",
        "normalization_scope": "all_fit",
        "required_world_size": 1,
        "required_device": "L4",
        "precision_mode": "fp32",
        "objective_mode": STANDALONE_LABELS_KD_MODE,
        "monitoring": {
            "train_eval_sequences": 2,
            "eval_batch_size": 1,
            "train_eval_selection": "hash_rank_v1",
        },
        "model": {
            "architecture": "gru",
            "hidden": 16,
            "layers": 2,
            "dropout": 0.0,
            "hint_dim": 0,
            "hint_projection": "learned",
        },
    }


def _load(tmp_path: Path, raw: dict[str, Any], name: str = "recipe.json") -> dict[str, Any]:
    """Serialize and load a real runtime recipe through the production boundary."""
    path = tmp_path / name
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_config(path)


@pytest.mark.parametrize("seed", [203501, 203502])
@pytest.mark.parametrize("kd_weight", [0.0, 0.2])
def test_all_source_matched_panel_configs_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int, kd_weight: float
) -> None:
    """Load each lambda/seed variant through the actual runtime config validator."""
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "0")
    config = _load(tmp_path, _config(seed=seed, kd_weight=kd_weight), f"{seed}-{kd_weight}.json")
    assert config["objective_mode"] == STANDALONE_LABELS_KD_MODE
    assert config["seed"] == seed and config["kd_weight"] == kd_weight
    assert config["model"]["hint_dim"] == 0
    assert objective_contract(config)["shared_denominator"] == "sum_abs_clipped_label_targets"


@pytest.mark.parametrize(
    "changes",
    [
        {"hint_weight": 0.1},
        {"blend_objective_alpha": 0.1},
        {"supervised_target": 0},
        {"parent_initializer": "future_input_innovations_v1"},
        {"normalization_scope": "prefix"},
        {"training_mask": "is_scored"},
        {"precision_mode": "tf32"},
        {"since_weight": 32.0},
    ],
)
def test_new_mode_rejects_unrelated_training_treatments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]
) -> None:
    """Keep the KD screen cold, raw-student, FP32, and isolated to one mechanism."""
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "0")
    config = _config()
    config.update(changes)
    with pytest.raises(ValueError):
        _load(tmp_path, config, f"invalid-{len(changes)}-{next(iter(changes))}.json")


def test_kd_controls_are_cache_matched_and_existing_label_since_mode_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep the cache in lambda-zero controls and preserve the old since-label recipe."""
    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "0")
    control = _load(tmp_path, _config())
    teacher_cache = tmp_path / "fit-cache"
    validate_fit_reference_inputs(control, teacher_cache, None, None)
    assert requires_teacher_cache(control, False)
    with pytest.raises(ValueError, match="requires a validated fit-only"):
        validate_fit_reference_inputs(control, None, None, None)
    with pytest.raises(ValueError, match="reference-cache"):
        validate_fit_reference_inputs(control, teacher_cache, tmp_path / "reference", None)
    with pytest.raises(ValueError, match="raw student z"):
        validate_fit_reference_inputs(control, teacher_cache, None, tmp_path / "monitor-ref")

    old_label = _config(kd_weight=0.0, since_weight=32.0)
    old_label["objective_mode"] = STANDALONE_LABELS_MODE
    old_label_config = _load(tmp_path, old_label, "historical-label-since.json")
    assert old_label_config["since_weight"] == 32.0
    assert not requires_teacher_cache(old_label_config, False)
    with pytest.raises(ValueError, match="labels-only GRU"):
        _load(tmp_path, _config(since_weight=32.0), "kd-since-rejected.json")


def test_joint_kd_loss_and_gradient_match_independent_formula() -> None:
    """Check clipped target-magnitude weighting and the shared window denominator."""
    target = torch.tensor([[[3.0, -0.5], [0.2, -0.8], [-1.5, 2.5]]])
    prediction = torch.tensor([[[0.2, -0.3], [0.7, 0.1], [-0.6, 0.5]]], requires_grad=True)
    teacher = torch.tensor([[[-3.0, 0.6], [0.4, -0.1], [1.0, -2.5]]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    hints = torch.empty(1, 3, 0)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    kd_weight = 0.2
    intervals = [(0, 1), (1, 3)]
    denominators = global_window_denominators(target, mask, hints, intervals, context)
    denominator = denominators[0]
    numerators = objective_numerators(
        prediction,
        hints,
        target,
        teacher,
        hints,
        mask,
        objective_mode=STANDALONE_LABELS_KD_MODE,
        context=context,
    )
    actual = scaled_objective(numerators, denominators, 1, kd_weight, 0)
    clipped_target = target.clamp(-2, 2)
    clipped_teacher = teacher.detach().clamp(-2, 2)
    weights = clipped_target.abs() * mask[..., None]
    label_term = (weights * (prediction - clipped_target).square()).sum() / denominator
    kd_term = (weights * (prediction - clipped_teacher).square()).sum() / denominator
    oracle = label_term + kd_weight * kd_term
    actual_gradient = torch.autograd.grad(actual, prediction, retain_graph=True)[0]
    oracle_gradient = torch.autograd.grad(oracle, prediction)[0]
    torch.testing.assert_close(actual, oracle, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, oracle_gradient, rtol=0, atol=0)
    actual.backward()
    assert teacher.grad is None

    invalid_teacher = teacher.detach().clone()
    invalid_teacher[0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="finite selected raw values"):
        objective_numerators(
            prediction.detach(),
            hints,
            target,
            invalid_teacher,
            hints,
            mask,
            objective_mode=STANDALONE_LABELS_KD_MODE,
            context=context,
        )


def test_lambda_zero_matches_old_label_update_exactly() -> None:
    """A zero-KD step preserves the existing standalone-label gradient and update."""
    target = torch.tensor([[[0.4, -0.7], [1.1, 0.2]]])
    x = torch.tensor([[[0.2, -0.3], [0.9, 0.1]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    hints = torch.empty(1, 2, 0)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    old = nn.Linear(2, 2)
    new = copy.deepcopy(old)
    optimizer_old = torch.optim.AdamW(old.parameters(), lr=0.001)
    optimizer_new = torch.optim.AdamW(new.parameters(), lr=0.001)
    denominator = window_denominators(target, mask, hints)
    for model, optimizer, mode in (
        (old, optimizer_old, STANDALONE_LABELS_MODE),
        (new, optimizer_new, STANDALONE_LABELS_KD_MODE),
    ):
        prediction = model(x)
        parts = objective_numerators(
            prediction,
            hints,
            target,
            torch.zeros_like(target),
            hints,
            mask,
            objective_mode=mode,
            context=context,
        )
        loss = scaled_objective(parts, denominator, 1, 0.0, 0.0)
        loss.backward()
        optimizer.step()
    for old_parameter, new_parameter in zip(old.parameters(), new.parameters(), strict=True):
        torch.testing.assert_close(old_parameter.grad, new_parameter.grad, rtol=0, atol=0)
        torch.testing.assert_close(old_parameter, new_parameter, rtol=0, atol=0)


def test_raw_student_monitor_ignores_cache_and_scores_full_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Monitor the raw student on train-eval and full valid without loading references."""
    import wnn.training.sequence_complement_monitor as monitor_module

    data = SimpleNamespace(
        fit=SimpleNamespace(
            members=[("train", index) for index in range(10325)],
            sources={"train": SimpleNamespace(environment=SimpleNamespace(root=tmp_path))},
        ),
        evaluation=SimpleNamespace(members=[("valid", index) for index in range(1873)]),
        split=SimpleNamespace(sha256=PRIMARY_SPLIT_SHA256, valid_groups=()),
        provenance={"teacher": {"manifest_sha256": "a" * 64}},
    )
    record_inputs: list[tuple[int, object, str]] = []
    observed: dict[str, object] = {}

    class FakeRecords:
        def __init__(
            self,
            root: Path,
            members: list[tuple[str, int]],
            hint_dim: int,
            cache: object = None,
            hint_members: object = None,
            training_mask: str = "is_scored",
        ) -> None:
            record_inputs.append((len(members), cache, training_mask))

    def fake_evaluate(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        observed.update(kwargs)
        return (
            {"train_eval/wp": 0.4, "train_eval/sequence_count": 2},
            {
                "valid/wp": 0.6,
                "valid/z/wp": 0.6,
                "valid/sequence_count": 1873,
                "valid/selected_rows": 3_528_273,
            },
        )

    monkeypatch.setenv("CUBLAS_EMULATE_SINGLE_PRECISION", "0")
    monkeypatch.setattr(monitor_module, "SequenceRecords", FakeRecords)
    monkeypatch.setattr(monitor_module, "evaluate_monitoring_sets", fake_evaluate)
    config = _load(tmp_path, _config(kd_weight=0.2), "monitor.json")
    monitor = ComplementMonitor(data, None, config, tmp_path, resume=False, primary=True)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    monitor.evaluate(nn.Linear(1, 1), config, context, 0, 1)
    assert record_inputs == [(2, None, "all_after_warmup"), (1873, None, "is_scored")]
    assert observed["blend_alpha"] is None and observed["report_standalone"] is True
    assert (
        monitor.context["endpoint"]
        == "raw student z; label plus fit-cache KD; no reference or blend"
    )
    assert "reference_sha256" not in monitor.context
    with (tmp_path / MONITOR_EVENT_FILENAME).open(encoding="utf-8") as stream:
        event = json.loads(stream.readline())
    assert event["metrics"]["valid/wp"] == event["metrics"]["valid/z/wp"] == 0.6


def test_epoch_objective_reporting_includes_kd_term() -> None:
    """Report label WMSE, KD WMSE, lambda, and their actual combined objective."""
    config = _config(kd_weight=0.2)
    totals = torch.tensor([0.0, 0.0, 0.0, 0.3, 0.2, 2.0], dtype=torch.float64)
    result = objective_metrics(config, totals)
    assert result["objective_label_wmse"] == pytest.approx(0.15)
    assert result["objective_kd_wmse"] == pytest.approx(0.1)
    assert result["total_loss"] == pytest.approx(0.17)
    assert result["objective_kd_weight"] == 0.2
