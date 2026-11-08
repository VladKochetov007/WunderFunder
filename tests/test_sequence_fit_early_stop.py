from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from wnn.training import sequence_fit_epoch
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_epoch import FitEpochRuntime, run_epochs
from wnn.training.sequence_loop import EpochResult
from wnn.training.sequence_monitor import SequenceMonitor


class _StatusMonitor:
    def __init__(self) -> None:
        self.statuses: list[tuple[str, int, int, dict[str, object]]] = []

    def status(
        self, state: str, step: int, epoch: int, context: dict[str, object] | None = None
    ) -> None:
        self.statuses.append((state, step, epoch, context or {}))


def _policy() -> dict[str, int | float]:
    return {"min_epoch": 5, "patience": 4, "min_delta": 0.0005}


def test_fit_stops_only_after_full_valid_patience_and_retains_best_last_pointers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    config: dict[str, Any] = {
        "epochs": 20,
        "chunk_rows": 2,
        "batch_size": 2,
        "kd_weight": 0.0,
        "hint_weight": 0.0,
        "checkpoint_retention": "all_epochs",
        "monitoring": {"early_stopping": _policy(), "eval_batch_size": 2},
    }
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    monitor = _StatusMonitor()
    trained_epochs: list[int] = []
    saved_histories: list[list[dict[str, object]]] = []
    output: dict[str, object] = {}

    def train(*args: object, **_kwargs: object) -> EpochResult:
        epoch = args[7]
        if not isinstance(epoch, int):
            raise TypeError("fit loop did not pass an integer epoch")
        trained_epochs.append(epoch)
        return EpochResult(1, True, torch.tensor([0, 0, 0, 0, 0, 0, 1, 1.0]), 0.1)

    scores = iter([0.5, 0.5, 0.5, 0.5, 0.5])

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
        return {"entries": entries, "index_sha256": str(completed) * 64}

    def save_state(*args: object, **_kwargs: object) -> None:
        history = args[9]
        saved_histories.append([dict(row) for row in history])

    monkeypatch.setattr(
        sequence_fit_epoch.torch.cuda, "reset_peak_memory_stats", lambda _device: None
    )
    monkeypatch.setattr(sequence_fit_epoch, "train_epoch", train)
    monkeypatch.setattr(sequence_fit_epoch, "evaluate_full_valid", evaluate)
    monkeypatch.setattr(sequence_fit_epoch, "publish_epoch_snapshot", publish_snapshot)
    monkeypatch.setattr(sequence_fit_epoch, "current_monitor_offsets", lambda *_args: None)
    monkeypatch.setattr(sequence_fit_epoch, "save_safe_state", save_state)
    monkeypatch.setattr(
        sequence_fit_epoch, "write_final_outputs", lambda *_args: output.update(final=True) or True
    )

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

    assert trained_epochs == [0, 1, 2, 3, 4]
    assert len(saved_histories[-1]) == 5
    assert saved_histories[-1][-1]["early_stopped"] is True
    assert output["final"] is True
    assert monitor.statuses[-1][0] == "early_stopped"
    pointers = json.loads((tmp_path / "checkpoint_pointers.json").read_text())
    assert pointers["status"] == "early_stopped"
    assert pointers["completed_epochs"] == 5
    assert pointers["best"]["epoch"] == 1
    assert pointers["last"]["epoch"] == 5
    assert pointers["early_stopping"]["bad_count"] == 4
