"""Full-sequence dataset records for the canonical recurrent runtime."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from wnn.data import N_FEATURES, SEQUENCE_LENGTH, WARMUP
from wnn.training.datasets import Sequences
from wnn.training.environment import InputConfiguration, TrainingEnvironment
from wnn.training.sequence_cache import TeacherCache
from wnn.training.sequence_features import (
    RAW_INPUT_MODE,
    sequence_input_width,
    sequence_inputs_from_raw,
)


class SequenceRecords(Dataset):
    """Wrap complete sequences with source keys, causal inputs, and DDP padding."""

    def __init__(
        self,
        root: Path,
        members: list[tuple[str, int]],
        hint_dim: int,
        cache: TeacherCache | None = None,
        hint_members: list[tuple[str, int]] | None = None,
        training_mask: str = "is_scored",
        input_feature_mode: str = RAW_INPUT_MODE,
    ) -> None:
        self.members = members
        self.hint_dim = hint_dim
        self.cache = cache
        self.hint_members = hint_members
        if training_mask not in ("is_scored", "all_after_warmup"):
            raise ValueError("unsupported sequence training mask")
        self.input_feature_mode = input_feature_mode
        self.input_width = sequence_input_width(input_feature_mode)
        self.training_mask = training_mask
        env = TrainingEnvironment(root=root, inputs=InputConfiguration(raw_only=True))
        self.sources = {
            split: Sequences(
                split, [group for source, group in members if source == split], environment=env
            )
            for split in ("train", "valid")
        }
        self.positions = {}
        for split in ("train", "valid"):
            for position, group in enumerate(self.sources[split].indices):
                self.positions[(split, group)] = position

    def __len__(self) -> int:
        """Return the number of real source-qualified sequences."""
        return len(self.members)

    def validate_source_schema(self) -> None:
        """Require labels and the scored-row mask when the fitting recipe needs it."""
        for source in self.sources.values():
            if not source.indices:
                continue
            if source._file is None:
                source._file = source.environment.sequence_file(source.path)
            if not source._file.has_targets:
                raise ValueError("fitting sources require t0/t1 target columns")
            if self.training_mask == "is_scored" and not source._file.has_mask:
                raise ValueError("fitting sources require t0/t1 and is_scored columns")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one complete sequence with assembled inputs and optional teacher values."""
        if index == -1:
            return {
                "x": torch.zeros(SEQUENCE_LENGTH, self.input_width),
                "y": torch.zeros(SEQUENCE_LENGTH, 2),
                "mask": torch.zeros(SEQUENCE_LENGTH, dtype=torch.bool),
                "teacher": torch.zeros(SEQUENCE_LENGTH, 2),
                "hints": torch.zeros(SEQUENCE_LENGTH, self.hint_dim),
            }
        split, group = self.members[index]
        x, y, mask = self.sources[split][self.positions[(split, group)]]
        if x.shape != (SEQUENCE_LENGTH, N_FEATURES) or y.shape != (SEQUENCE_LENGTH, 2):
            raise ValueError("raw112/full-sequence source schema mismatch")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f"nonfinite source sequence: {split}/{group}")
        configured_x = sequence_inputs_from_raw(x, self.input_feature_mode)
        if not isinstance(configured_x, torch.Tensor):
            raise TypeError("sequence input assembly must return a torch tensor")
        if self.training_mask == "all_after_warmup":
            mask = torch.ones_like(mask, dtype=torch.bool)
        else:
            mask = mask.clone()
        mask[:WARMUP] = False
        teacher = torch.zeros_like(y)
        hints = torch.zeros(SEQUENCE_LENGTH, self.hint_dim)
        if self.cache is not None:
            teacher, hints = self.cache.read(split, group, self.hint_dim)
            if self.hint_members is not None:
                hint_source, hint_group = self.hint_members[index]
                _, hints = self.cache.read(hint_source, hint_group, self.hint_dim)
        return {
            "x": configured_x.float(),
            "y": y.float(),
            "mask": mask,
            "teacher": teacher,
            "hints": hints,
        }
