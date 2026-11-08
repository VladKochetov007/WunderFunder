from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from wnn.training import sequence_fit_epoch
from wnn.training.sequence_config_validation import validate_monitoring_config
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_artifacts import write_final_outputs
from wnn.training.sequence_fit_epoch import FitEpochRuntime, run_epochs
from wnn.training.sequence_loop import EpochResult
from wnn.training.sequence_monitor import SequenceMonitor
from wnn.training.sequence_monitor_checkpoints import advance_early_stop


class _StatusMonitor:
    def __init__(self) -> None:
        self.statuses: list[tuple[str, int, int, dict[str, object]]] = []
        self.progress = self

    def phase(self, *_args: object) -> None:
        return None

    def status(
        self, state: str, step: int, epoch: int, context: dict[str, object] | None = None
    ) -> None:
        self.statuses.append((state, step, epoch, context or {}))


def _policy() -> dict[str, int | float]:
    return {"min_epoch": 5, "patience": 4, "min_delta": 0.0005}


@pytest.mark.parametrize(
    ("policy", "configured_epochs", "expected_epochs", "expected_status"),
    [
        (_policy(), 20, 5, "early_stopped"),
        (
            {"min_epoch": 2, "patience": 2, "min_delta": 0.0001, "action": "report_only"},
            6,
            6,
            "completed",
        ),
    ],
)
def test_fit_stop_and_report_only_actions_keep_best_separate_from_fixed_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: dict[str, int | float | str],
    configured_epochs: int,
    expected_epochs: int,
    expected_status: str,
) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    config: dict[str, Any] = {
        "epochs": configured_epochs,
        "chunk_rows": 2,
        "batch_size": 2,
        "kd_weight": 0.0,
        "hint_weight": 0.0,
        "checkpoint_retention": "all_epochs",
        "model": {"architecture": "gru"},
        "monitoring": {"early_stopping": policy, "eval_batch_size": 2},
    }
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    monitor = _StatusMonitor()
    trained_epochs: list[int] = []
    saved_histories: list[list[dict[str, object]]] = []
    final_outputs: dict[str, object] = {}

    def train(*args: object, **_kwargs: object) -> EpochResult:
        epoch = args[7]
        if not isinstance(epoch, int):
            raise TypeError("fit loop did not pass an integer epoch")
        trained_epochs.append(epoch)
        return EpochResult(1, True, torch.tensor([0, 0, 0, 0, 0, 0, 1, 1.0]), 0.1)

    scores = iter([0.5] * 20)

    def evaluate(*_args: object, **_kwargs: object) -> dict[str, int | float]:
        score = next(scores)
        return {
            "train_eval/wp": score,
            "train_eval/wp/t0": score,
            "train_eval/wp/t1": score,
            "valid/wp": score,
            "valid/wp/t0": score,
            "valid/wp/t1": score,
        }

    def publish_snapshot(
        _output: Path,
        _model: nn.Module,
        _config: dict[str, Any],
        _identity: dict[str, object],
        _normalization: dict[str, list[float]],
        _provenance: dict[str, object],
        _receipt: Path,
        _profile_hash: str,
        _history: list[dict[str, object]],
        epoch: int,
        global_step: int,
        previous: dict[str, object] | None,
        _context: DistributedContext,
        _label: str,
    ) -> dict[str, object]:
        entries = list(previous["entries"]) if previous is not None else []
        completed = epoch + 1
        entries.append(
            {
                "epoch": epoch,
                "completed_epochs": completed,
                "global_step": global_step,
                "checkpoint_path": f"epoch_checkpoints/epoch-{completed:06d}-{'a' * 64}.pt",
                "checkpoint_sha256": "a" * 64,
            }
        )
        return {"entries": entries, "index_sha256": "a" * 64}

    def save_state(*args: object, **_kwargs: object) -> None:
        history = args[9]
        saved_histories.append([dict(row) for row in history])

    def write_final(
        _output: Path,
        _model: nn.Module,
        final_config: dict[str, Any],
        final_history: list[dict[str, object]],
        *_args: object,
    ) -> bool:
        final_outputs.update(
            final=True, config=final_config, history=[dict(row) for row in final_history]
        )
        return True

    monkeypatch.setattr(
        sequence_fit_epoch.torch.cuda, "reset_peak_memory_stats", lambda _device: None
    )
    monkeypatch.setattr(sequence_fit_epoch, "train_epoch", train)
    monkeypatch.setattr(sequence_fit_epoch, "evaluate_full_valid", evaluate)
    monkeypatch.setattr(sequence_fit_epoch, "publish_epoch_snapshot", publish_snapshot)
    monkeypatch.setattr(sequence_fit_epoch, "current_monitor_offsets", lambda *_args: None)
    monkeypatch.setattr(sequence_fit_epoch, "save_safe_state", save_state)
    monkeypatch.setattr(sequence_fit_epoch, "write_final_outputs", write_final)

    runtime = FitEpochRuntime(
        dataset=object(),
        model=model,
        base_model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        context=context,
        rank_log=object(),
        recorder=None,
        identity={"run": "monitor-stop"},
        normalization={"mean": [0.0], "std": [1.0]},
        data_provenance={},
        output=tmp_path,
        profile_receipt=tmp_path / "receipt.json",
        profile_hash="c" * 64,
        save_state=lambda *_args: None,
        monitor=cast(SequenceMonitor, monitor),
        evaluation_dataset=object(),
        train_eval_dataset=object(),
    )
    run_epochs(runtime, 0, 0, [], None, monitoring_enabled=True)

    assert trained_epochs == list(range(expected_epochs))
    assert len(saved_histories[-1]) == expected_epochs
    final_row = saved_histories[-1][-1]
    assert final_row["early_stopped"] is (expected_status == "early_stopped")
    assert final_row["early_stopping_action"] == policy.get("action", "stop")
    assert final_outputs["final"] is True
    assert len(final_outputs["history"]) == expected_epochs
    assert final_outputs["config"]["epochs"] == configured_epochs
    assert monitor.statuses[-1][0] == expected_status
    pointers = json.loads((tmp_path / "checkpoint_pointers.json").read_text())
    assert pointers["status"] == expected_status
    assert pointers["completed_epochs"] == expected_epochs
    assert pointers["best"]["epoch"] == 1
    assert pointers["last"]["epoch"] == expected_epochs
    assert pointers["early_stopping"]["action"] == policy.get("action", "stop")
    assert pointers["early_stopping"]["plateau_reached"] is True
    assert pointers["early_stopping"]["bad_count"] == expected_epochs - 1
    assert "exploratory pointer only" in pointers["best_checkpoint_policy"]
    if policy.get("action") == "report_only":
        assert final_row["early_stop_plateau_reached"] is True
        assert final_row["early_stopped"] is False
        resumed = advance_early_stop(saved_histories[-1], expected_epochs + 1, 0.49, policy)
        assert resumed.action == "report_only"
        assert resumed.plateau_reached is True and resumed.should_stop is False
        assert resumed.decline_from_observed_peak == pytest.approx(0.01)


def test_report_only_overlay_requires_retained_primary_valid_schedule() -> None:
    preset_path = (
        Path(__file__).parents[1]
        / "research/experiment_drafts/training-repair-20261108/l4-report-only-overlay-v1.json"
    )
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    assert "epochs" not in preset
    assert preset["checkpoint_retention"] == "all_epochs"
    validate_monitoring_config(
        preset["monitoring"],
        {"architecture": "gru"},
        6,
        preset["checkpoint_retention"],
    )
    legacy_stop = {
        **preset["monitoring"],
        "early_stopping": {"min_epoch": 2, "patience": 2, "min_delta": 0.0001},
    }
    with pytest.raises(ValueError, match="fixed 5/4/0.0005"):
        validate_monitoring_config(
            legacy_stop,
            {"architecture": "gru"},
            6,
            preset["checkpoint_retention"],
        )


def test_control_role_cannot_drive_primary_valid_plateau_policy() -> None:
    evaluation = {
        "train_eval/wp": 0.4,
        "train_eval/wp/t0": 0.4,
        "train_eval/wp/t1": 0.4,
        "control/wp": 0.4,
        "control/wp/t0": 0.4,
        "control/wp/t1": 0.4,
    }
    with pytest.raises(ValueError, match="primary valid"):
        sequence_fit_epoch._record_evaluation_history(
            [],
            {},
            evaluation,
            "control",
            1,
            {"min_epoch": 2, "patience": 2, "min_delta": 0.0001, "action": "report_only"},
        )


def test_final_model_stays_at_configured_endpoint_and_labels_best_as_exposed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic history proves a six-epoch fixed endpoint remains epoch six."""
    receipt = tmp_path / "profile.json"
    receipt.write_text("{}", encoding="utf-8")
    model = nn.Linear(1, 1)
    config: dict[str, Any] = {
        "epochs": 6,
        "checkpoint_retention": "all_epochs",
        "model": {"architecture": "synthetic_test_stub"},
    }
    history = [
        {
            "epoch": epoch,
            "valid_wp": score,
            "early_stop_best_raw_wp": 0.6,
            "early_stop_best_epoch": 2,
            "early_stopping_action": "report_only",
            "early_stop_plateau_reached": True,
            "decline_from_observed_peak": max(0.0, 0.6 - score),
            "early_stopped": False,
        }
        for epoch, score in enumerate((0.4, 0.6, 0.55, 0.58, 0.59, 0.58))
    ]

    def portable_save_final(
        path: Path,
        model_state: dict[str, torch.Tensor],
        saved_config: dict[str, object],
        epoch: int,
        completed_epochs: int,
        data: dict[str, object],
        provenance: dict[str, object],
        *,
        allow_early_stop: bool = False,
    ) -> None:
        assert epoch == 5 and completed_epochs == 6
        assert allow_early_stop is False
        torch.save(
            {
                "checkpoint_schema": "wnn.sequence-fit.v1",
                "model": model_state,
                "config": saved_config,
                "epoch": epoch,
                "completed_epochs": completed_epochs,
                "data": data,
                "provenance": provenance,
            },
            path,
        )

    monkeypatch.setattr("wnn.training.sequence_fit_artifacts.save_final_model", portable_save_final)
    monkeypatch.setattr(
        "wnn.training.sequence_fit_artifacts.atomic_write_json",
        lambda value, path: path.write_text(json.dumps(value), encoding="utf-8"),
    )

    assert write_final_outputs(
        tmp_path,
        model,
        config,
        history,
        {"run": "synthetic"},
        {"mean": [0.0], "std": [1.0]},
        {
            "evaluation_role": "valid",
            "monitoring_plan": {
                "evaluation_reporting": "full official valid after each epoch; exposed report"
            },
        },
        receipt,
        "b" * 64,
        4,
        "c" * 64,
    )
    saved = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    assert saved["epoch"] == 5
    assert saved["completed_epochs"] == config["epochs"]
    assert saved["provenance"]["fixed_checkpoint_policy"] == "configured-final-epoch"
    best = saved["provenance"]["best_observed_evaluation"]
    assert best["epoch"] == 2 and best["wp"] == pytest.approx(0.6)
    assert "exposed" in best["interpretation"]
    assert best["pointer"] == "checkpoint_pointers.json#best"
    terminal = json.loads((tmp_path / "terminal_status.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "completed"
    assert terminal["completion_kind"] == "max_epochs"
