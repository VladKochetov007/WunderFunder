from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from wnn.data import N_FEATURES, SEQUENCE_LENGTH, WARMUP
from wnn.training.datasets import Sequences
from wnn.training.environment import InputConfiguration, TrainingEnvironment
from wnn.training.sequence_cache import TeacherCache
from wnn.training.sequence_disagreement import DisagreementCache


class SequenceRecords(Dataset):
    """Wrap canonical full sequences with explicit source keys and zero-weight DDP padding."""

    def __init__(
        self,
        root: Path,
        members: list[tuple[str, int]],
        hint_dim: int,
        cache: TeacherCache | DisagreementCache | None = None,
        hint_members: list[tuple[str, int]] | None = None,
        training_mask: str = "is_scored",
        disagreement_mode: str | None = None,
    ) -> None:
        self.members = members
        if (disagreement_mode is not None) != isinstance(cache, DisagreementCache):
            raise ValueError("disagreement mode and source cache must be used together")
        self.hint_dim = hint_dim
        self.cache = cache
        self.disagreement_mode = disagreement_mode
        self.hint_members = hint_members
        if training_mask not in ("is_scored", "all_after_warmup"):
            raise ValueError("unsupported sequence training mask")
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
        """Require targets and, for scored fitting, the explicit evaluation mask."""
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
        """Return raw inputs, labels, masks and optional cached teacher values."""
        if index == -1:
            empty = {
                "x": torch.zeros(SEQUENCE_LENGTH, N_FEATURES),
                "y": torch.zeros(SEQUENCE_LENGTH, 2),
                "mask": torch.zeros(SEQUENCE_LENGTH, dtype=torch.bool),
                "teacher": torch.zeros(SEQUENCE_LENGTH, 2),
                "hints": torch.zeros(SEQUENCE_LENGTH, self.hint_dim),
            }
            if self.disagreement_mode is not None:
                empty["kd_row_weight"] = torch.ones(SEQUENCE_LENGTH)
            return empty
        split, group = self.members[index]
        x, y, mask = self.sources[split][self.positions[(split, group)]]
        if x.shape != (SEQUENCE_LENGTH, N_FEATURES) or y.shape != (SEQUENCE_LENGTH, 2):
            raise ValueError("raw112/full-sequence training schema mismatch")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f"nonfinite source sequence: {split}/{group}")
        if self.training_mask == "all_after_warmup":
            mask = torch.ones_like(mask, dtype=torch.bool)
        else:
            mask = mask.clone()
        mask[:WARMUP] = False
        teacher = torch.zeros_like(y)
        hints = torch.zeros(SEQUENCE_LENGTH, self.hint_dim)
        if isinstance(self.cache, DisagreementCache):
            if self.disagreement_mode is None:
                raise ValueError("disagreement cache requires an explicit mode")
            teacher, hints, kd_row_weight = self.cache.read_weighted(
                split, group, self.disagreement_mode
            )
        elif self.cache is not None:
            teacher, hints = self.cache.read(split, group, self.hint_dim)
            if self.hint_members is not None:
                hint_source, hint_group = self.hint_members[index]
                _, hints = self.cache.read(hint_source, hint_group, self.hint_dim)
        result = {"x": x.float(), "y": y.float(), "mask": mask, "teacher": teacher, "hints": hints}
        if self.disagreement_mode is not None:
            result["kd_row_weight"] = kd_row_weight
        return result
