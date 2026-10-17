from __future__ import annotations

import pytest
import torch

from wnn.resources import configure_strict_fp32
from wnn.training.cli import build_parser
from wnn.training.environment import TrainingEnvironment
from wnn.training.epoch import fit_epoch_summary, record_fit_window
from wnn.training.setup_lineage import configure_runtime
from wnn.training.state import TrainingRun


@pytest.mark.parametrize(
    ("precision", "expected"), [("fp32", False), ("fp16", True), ("bf16", True)]
)
def test_fit_runtime_disables_tf32_only_for_fp32(precision: str, expected: bool) -> None:
    original = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        parser = build_parser()
        run = TrainingRun(TrainingEnvironment())
        run.options.ap = parser
        run.options.args = parser.parse_args(["--name", "precision-test", "--precision", precision])

        configure_runtime(run)

        assert torch.backends.cuda.matmul.allow_tf32 is expected
        assert torch.backends.cudnn.allow_tf32 is expected
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original


def test_shared_strict_fp32_helper_disables_both_tf32_backends() -> None:
    original = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        configure_strict_fp32()

        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original


def test_fit_epoch_summary_reports_window_objectives_rates_and_update_counts() -> None:
    class Scheduler:
        rates = iter(([0.1, 0.01], [0.05, 0.005]))

        def get_last_lr(self) -> list[float]:
            return next(self.rates)

    run = TrainingRun(TrainingEnvironment())
    run.optimization.sched = Scheduler()
    run.batch.loss = torch.tensor(1.0, requires_grad=True)
    record_fit_window(run)
    run.batch.loss = torch.tensor(3.0, requires_grad=True)
    record_fit_window(run)
    counts = {"optimizer_attempts": 2, "optimizer_completed_updates": 2}

    summary = fit_epoch_summary(run, counts)

    assert summary["objective_mean"] == 2.0
    assert summary["optimizer_windows"] == 2
    assert summary["optimizer_updates"] == counts
    assert summary["learning_rate_by_group"] == [
        {"group": 0, "first": 0.1, "last": 0.05, "mean": 0.07500000000000001},
        {"group": 1, "first": 0.01, "last": 0.005, "mean": 0.0075},
    ]
    assert run.progress.fit_loss_sum is not None
    assert not run.progress.fit_loss_sum.requires_grad
