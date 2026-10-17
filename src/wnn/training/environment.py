from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import torch
from attrs import define, field

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, SEQUENCE_LENGTH, WARMUP, SequenceFile
from wnn.propensity import propensity
from wnn.splits import dev_halves, valid_row_groups
from wnn.training.forward import forward_chunk
from wnn.training.models import make_model
from wnn.training_dynamics import build_optimizer


@define
class InputConfiguration:
    """Causal input transforms shared explicitly with this run's lazy datasets."""

    propensity_model: torch.nn.Module | None = None
    online_norm: bool = False
    heuristics: bool = False
    raw_only: bool = False
    research_features: str = "none"
    research_feature_prior: dict[str, Any] | None = None
    research_feature_control: str = "none"
    privileged_mode: str = "none"
    privileged_horizon: int = 8
    sentinel_preprocessor: Any = None


@define
class ReadoutConfiguration:
    """Optional outcome-code moment tables used by prediction and snapshots."""

    tables: tuple[torch.Tensor, ...] | None = None
    tilted: bool = False


@define
class TrainingEnvironment:
    """Filesystem/schema identities and replaceable I/O/backend dependencies."""

    source_root: Path = field(factory=lambda: Path(__file__).resolve().parents[3])
    root: Path = field(
        factory=lambda: Path(
            os.environ.get("WNN_PROJECT_ROOT", Path(__file__).resolve().parents[3])
        ).resolve()
    )
    argv: list[str] = field(factory=lambda: sys.argv[1:])
    inputs: InputConfiguration = field(factory=InputConfiguration)
    readout: ReadoutConfiguration = field(factory=ReadoutConfiguration)
    sequence_length: int = SEQUENCE_LENGTH
    warmup: int = WARMUP
    corrupt_train_groups: frozenset[int] = field(
        factory=lambda: frozenset(CORRUPT_TRAIN_ROW_GROUPS)
    )
    sequence_file: Callable[..., SequenceFile] = SequenceFile
    valid_row_groups: Callable[..., Any] = valid_row_groups
    dev_halves: Callable[..., Any] = dev_halves
    propensity: Callable[..., torch.Tensor] = propensity
    forward_chunk: Callable[..., Any] = forward_chunk
    build_optimizer: Callable[[torch.nn.Module, argparse.Namespace], torch.optim.Optimizer] = (
        build_optimizer
    )
    make_model: Callable[[dict[str, Any], int, int], torch.nn.Module] = make_model
    prediction: Callable[..., Any] | None = None

    @property
    def entrypoint(self) -> Path:
        """Keep source provenance bound to the public CLI, not a moved helper."""
        return self.source_root / "scripts/train_rnn.py"
