from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from wnn.research_reporting import RunRecorder
from wnn.training.sequence_checkpoint import save_training_state
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_fit_full import run_full_fit_impl
from wnn.training.sequence_fit_utils import RankJsonlLog, SequenceDataset
from wnn.training.sequence_monitor import SequenceMonitor


def run_full_fit(
    dataset: SequenceDataset,
    model: torch.nn.Module,
    base_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    config: dict[str, Any],
    context: DistributedContext,
    rank_log: RankJsonlLog,
    recorder: RunRecorder | None,
    identity: dict[str, object],
    normalization: dict[str, list[float]],
    data_provenance: Mapping[str, object],
    output: Path,
    resume_path: Path | None,
    profile_receipt: Path,
    profile_hash: str,
    *,
    evaluation_dataset: SequenceDataset | None = None,
    train_eval_dataset: SequenceDataset | None = None,
    monitor: SequenceMonitor | None = None,
) -> None:
    """Keep the public fit seam stable while preserving caller-monkeypatched safe saves."""
    run_full_fit_impl(
        dataset,
        model,
        base_model,
        optimizer,
        scheduler,
        config,
        context,
        rank_log,
        recorder,
        identity,
        normalization,
        data_provenance,
        output,
        resume_path,
        profile_receipt,
        profile_hash,
        evaluation_dataset=evaluation_dataset,
        train_eval_dataset=train_eval_dataset,
        monitor=monitor,
        save_state=save_training_state,
    )
