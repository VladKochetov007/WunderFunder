"""Typed boundary between PR81 epoch engines and shared fit lifecycle code."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch


class EpochResult(Protocol):
    """Canonical epoch telemetry plus optional experiment-specific totals."""

    updates: int
    complete: bool
    totals: torch.Tensor
    elapsed_seconds: float
    objective_totals: torch.Tensor | None
    auxiliary_totals: torch.Tensor | None


EpochDriver = Callable[..., EpochResult]
EpochSummarizer = Callable[[EpochResult, Mapping[str, Any]], Mapping[str, Any]]
ProfileFields = Callable[[list[EpochResult], Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class TrainingEngine:
    """Supply objective math while preserving shared checkpoint/profile lifecycles.

    ``train_epoch`` must accept the canonical train_epoch arguments, including
    keyword-only ``progress``.  ``summarize_epoch`` returns the canonical epoch
    summary keys and a correct combined ``total_loss``.  ``profile_fields``
    returns JSON-safe diagnostics and a combined ``total_loss`` over the actual
    profile updates.  The shared lifecycle never interprets auxiliary tensor
    layouts itself.
    """

    train_epoch: EpochDriver
    summarize_epoch: EpochSummarizer
    profile_fields: ProfileFields
