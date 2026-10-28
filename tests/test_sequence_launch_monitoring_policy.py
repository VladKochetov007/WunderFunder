from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_loop import _training_telemetry_due
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_monitor import MonitorEventWriter, SequenceMonitor
from wnn.training.sequence_monitor_budget import validate_monitor_event_budget
from wnn.training.sequence_monitor_checkpoints import (
    advance_early_stop,
    write_checkpoint_pointers,
)
from wnn.training.sequence_monitor_diagnostics import DiagnosticProbe


class _MonitorDataset:
    def __init__(self, groups: list[int], mask_name: str) -> None:
        self.members = [
            ("train" if mask_name == "all_after_warmup" else "valid", group) for group in groups
        ]
        self.training_mask = mask_name
        self.records: list[dict[str, torch.Tensor]] = []
        for group in groups:
            rows = torch.arange(8, dtype=torch.float32)
            x = torch.zeros(8, 112)
            x[:, 0] = torch.sin(rows * 0.3 + group)
            x[:, 1] = torch.cos(rows * 0.4 - group)
            y = x[:, :2] * 0.7 + 0.1
            mask = torch.ones(8, dtype=torch.bool)
            mask[:2] = False
            if mask_name == "is_scored":
                mask[-1] = False
            self.records.append(
                {
                    "x": x,
                    "y": y,
                    "mask": mask,
                    "teacher": torch.zeros(8, 2),
                    "hints": torch.zeros(8, 0),
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.records[index]


class _PassThrough(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_batch_sizes: list[int] = []

    def forward(
        self, values: torch.Tensor, state: object = None
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        self.seen_batch_sizes.append(values.shape[0])
        return values[..., :2], values.new_empty((*values.shape[:-1], 0)), state


def _expected_weighted_loss(dataset: _MonitorDataset) -> float:
    numerator = torch.zeros((), dtype=torch.float64)
    denominator = torch.zeros((), dtype=torch.float64)
    for record in dataset.records:
        clipped = record["y"].clamp(-2.0, 2.0)
        weight = clipped.abs() * record["mask"][:, None]
        difference = record["x"][:, :2] - clipped
        numerator += (weight.double() * difference.double().square()).sum()
        denominator += weight.double().sum()
    return float(numerator / denominator)


def _policy() -> dict[str, int | float]:
    return {"min_epoch": 5, "patience": 4, "min_delta": 0.0005}


def test_train_eval_and_full_valid_keep_distinct_masks_and_batch_size(tmp_path: Path) -> None:
    train_eval = _MonitorDataset([1, 2], "all_after_warmup")
    valid = _MonitorDataset([1, 2], "is_scored")
    settings = {
        "diagnostics_every_steps": 10,
        "training_telemetry_every_steps": 5,
        "eval_batch_size": 2,
        "train_eval_mask": "all_after_warmup",
        "early_stopping": _policy(),
    }
    monitor = SequenceMonitor(
        tmp_path,
        settings,
        train_eval.members,
        valid.members,
        "c" * 64,
        "d" * 64,
        resume=False,
    )
    model = _PassThrough().train()
    metrics = monitor.evaluate_epoch(
        model,
        train_eval,
        valid,
        2,
        2,
        "fp32",
        DistributedContext(0, 0, 1, torch.device("cpu"), False),
        0,
        1,
    )

    assert metrics["train_eval/selected_rows"] == 12
    assert metrics["valid/selected_rows"] == 10
    assert metrics["train_eval/loss"] == pytest.approx(_expected_weighted_loss(train_eval))
    assert metrics["valid/loss"] == pytest.approx(_expected_weighted_loss(valid))
    assert metrics["train_eval/loss"] != metrics["valid/loss"]
    assert model.seen_batch_sizes == [2] * 8
    assert model.training
    assert "timing/train_eval_seconds" in metrics
    assert "timing/full_valid_seconds" in metrics
    assert "timing/paired_evaluation_seconds" in metrics
    event = json.loads((tmp_path / "monitor_events.jsonl").read_text().splitlines()[0])
    assert event["kind"] == "evaluation"
    assert event["context"]["train_eval"]["row_mask"] == "all_after_warmup"
    assert event["context"]["valid"]["row_mask"] == "is_scored"
    assert "masks differ" in event["context"]["mask_comparison"]


def test_early_stop_keeps_strict_best_separate_from_min_delta_patience(tmp_path: Path) -> None:
    history: list[dict[str, object]] = []
    scores = [0.5, 0.5002, 0.5006, 0.5005, 0.5005, 0.5005]
    decisions = []
    entries = []
    for completed, score in enumerate(scores, start=1):
        decision = advance_early_stop(history, completed, score, _policy())
        decisions.append(decision)
        history.append(
            {
                "epoch": completed - 1,
                "valid_wp": score,
                "early_stop_best_raw_wp": decision.best_wp,
                "early_stop_best_epoch": decision.best_epoch,
                "early_stop_patience_reference_wp": decision.reference_wp,
                "early_stop_bad_checks": decision.bad_checks,
            }
        )
        entries.append(
            {
                "checkpoint_path": f"epoch_checkpoints/epoch-{completed:06d}-{'a' * 64}.pt",
                "checkpoint_sha256": "a" * 64,
            }
        )

    assert not decisions[-1].should_stop
    assert decisions[-1].best_wp == pytest.approx(0.5006)
    assert decisions[-1].best_epoch == 3
    assert decisions[-1].reference_wp == pytest.approx(0.5006)
    assert decisions[-1].bad_checks == 3
    pointer = write_checkpoint_pointers(
        tmp_path,
        "b" * 64,
        "running",
        history,
        {"entries": entries},
        _policy(),
    )
    saved = json.loads((tmp_path / "checkpoint_pointers.json").read_text())
    assert pointer == saved
    assert saved["best"]["epoch"] == 3
    assert saved["best_raw_wp"] == pytest.approx(0.5006)
    assert saved["last"]["epoch"] == 6
    assert saved["early_stopping"]["patience_reference_wp"] == pytest.approx(0.5006)
    assert saved["early_stopping"]["bad_count"] == 3

    resumed = advance_early_stop(history, 7, 0.5005, _policy())
    assert resumed.should_stop
    assert resumed.bad_checks == 4
    history.append(
        {
            "epoch": 6,
            "valid_wp": 0.5005,
            "early_stop_best_raw_wp": resumed.best_wp,
            "early_stop_best_epoch": resumed.best_epoch,
            "early_stop_patience_reference_wp": resumed.reference_wp,
            "early_stop_bad_checks": resumed.bad_checks,
        }
    )
    entries.append(
        {
            "checkpoint_path": f"epoch_checkpoints/epoch-000007-{'a' * 64}.pt",
            "checkpoint_sha256": "a" * 64,
        }
    )
    pointer = write_checkpoint_pointers(
        tmp_path,
        "b" * 64,
        "early_stopped",
        history,
        {"entries": entries},
        _policy(),
    )
    saved = json.loads((tmp_path / "checkpoint_pointers.json").read_text())
    assert pointer == saved
    assert saved["best"]["epoch"] == 3
    assert saved["last"]["epoch"] == 7
    assert saved["early_stopping"]["bad_count"] == 4


def test_monitor_budget_preserves_full_eval_journal_under_sparse_training_logs() -> None:
    event_count = validate_monitor_event_budget(360_000, 20, 100, 1_000)
    assert event_count < 8_192
    with pytest.raises(ValueError, match="1 GiB event journal bound"):
        validate_monitor_event_budget(360_000, 20, 1, 1_000)


def test_sparse_telemetry_keeps_epoch_boundaries_and_full_eval_events(tmp_path: Path) -> None:
    sampled = [step for step in range(1, 334) if _training_telemetry_due(step, step, 333, 100)]
    assert sampled == [100, 200, 300, 333]

    path = tmp_path / "monitor_events.jsonl"
    writer = MonitorEventWriter(path, resume=False)
    for epoch in range(1, 21):
        writer.append("evaluation", epoch, epoch, {"valid/wp": 0.1 + epoch / 1000}, {})
    assert len(path.read_text().splitlines()) == 20
    assert len((tmp_path / "monitor_evaluations.jsonl").read_text().splitlines()) == 20


def test_plain_gru_diagnostics_do_not_require_galam_branches() -> None:
    model = build_sequence_model(
        {
            "architecture": "gru",
            "hidden": 8,
            "layers": 1,
            "heads": 1,
            "ff_multiplier": 2,
            "context_rows": 8,
            "dropout": 0.0,
            "checkpoint_blocks": False,
            "hint_dim": 0,
            "hint_projection": "learned",
            "attention_backend": "math",
            "compile_blocks": False,
            "precision_mode": "fp32",
        }
    ).train()
    values = torch.randn(2, 4, 112)
    with DiagnosticProbe(model) as probe:
        prediction, _, _ = model(values)
        probe.observe_prediction(prediction, torch.ones(2, 4, dtype=torch.bool))
        prediction.square().mean().backward()
    metrics = probe.metrics()
    assert "local/hidden_state_rms" in metrics
    assert "local/gate/r/layer_0/mean" in metrics
    assert not any(key.startswith("global/") for key in metrics)
