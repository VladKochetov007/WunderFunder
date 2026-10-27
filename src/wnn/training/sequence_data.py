from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from attrs import define
from torch.utils.data import Dataset

from wnn.data import N_FEATURES, SEQUENCE_LENGTH, WARMUP
from wnn.splits import (
    CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    MAX_CAMPAIGN_SPLIT_BYTES,
    CampaignSplit,
    load_campaign_split,
)
from wnn.training.datasets import Sequences
from wnn.training.environment import InputConfiguration, TrainingEnvironment
from wnn.training.sequence_cache import TeacherCache
from wnn.training.sequence_normalization import (
    NormalizationMoments,
    build_normalization_identity,
    compute_population_moments,
    load_or_compute_full_fit_moments,
    moments_sha256,
    normalization_identity_sha256,
)

# Heterogeneous JSON recipes/provenance are restricted to the public runtime boundary.


class SequenceRecords(Dataset):
    """Wrap canonical full sequences with explicit source keys and zero-weight DDP padding."""

    def __init__(
        self,
        root: Path,
        members: list[tuple[str, int]],
        hint_dim: int,
        cache: TeacherCache | None = None,
        hint_members: list[tuple[str, int]] | None = None,
        training_mask: str = "is_scored",
    ) -> None:
        self.members = members
        self.hint_dim = hint_dim
        self.cache = cache
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
            return {
                "x": torch.zeros(SEQUENCE_LENGTH, N_FEATURES),
                "y": torch.zeros(SEQUENCE_LENGTH, 2),
                "mask": torch.zeros(SEQUENCE_LENGTH, dtype=torch.bool),
                "teacher": torch.zeros(SEQUENCE_LENGTH, 2),
                "hints": torch.zeros(SEQUENCE_LENGTH, self.hint_dim),
            }
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
        if self.cache is not None:
            teacher, hints = self.cache.read(split, group, self.hint_dim)
            if self.hint_members is not None:
                hint_source, hint_group = self.hint_members[index]
                _, hints = self.cache.read(hint_source, hint_group, self.hint_dim)
        return {"x": x.float(), "y": y.float(), "mask": mask, "teacher": teacher, "hints": hints}


@define
class SequenceData:
    """Validated membership shared by fitting, teacher exports and official scoring."""

    fit: SequenceRecords
    evaluation: SequenceRecords
    split: CampaignSplit
    provenance: dict[str, Any]
    normalization_sequences: int

    def normalization(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve source-bound raw-input moments from fitting members only.

        Float64 moment accumulation is preprocessing, not mixed-precision training.
        All model buffers and input tensors are serialized and consumed as FP32.
        """
        normalization = self.provenance["normalization"]
        scope = normalization["scope"]
        if scope == "all_fit":
            members = list(self.fit.members)
        elif scope == "prefix":
            members = self.fit.members[: min(len(self.fit), self.normalization_sequences)]
        else:
            raise ValueError("unsupported normalization scope")
        if len(members) != normalization["sequence_count"]:
            raise ValueError("normalization membership changed after data construction")

        def read_features(source_name: str, group: int) -> np.ndarray:
            """Read one selected fitting row group without labels or scoring masks."""
            source = self.fit.sources[source_name]
            if source._file is None:
                source._file = source.environment.sequence_file(source.path)
            return source._file.read(group, targets=False, mask=False).features

        source_root = self.fit.sources[members[0][0]].environment.root.resolve()
        source_fingerprints = normalization["source_fingerprints"]
        if scope == "all_fit":
            result = load_or_compute_full_fit_moments(
                source_root / "cache" / "sequence_normalization",
                source_fingerprints,
                self.provenance["split_sha256"],
                members,
                read_features,
                feature_count=N_FEATURES,
            )
        else:
            identity = build_normalization_identity(
                source_fingerprints,
                self.provenance["split_sha256"],
                members,
                scope,
                N_FEATURES,
            )
            mean, std = compute_population_moments(members, read_features, feature_count=N_FEATURES)
            result = NormalizationMoments(
                mean,
                std,
                normalization_identity_sha256(identity),
                moments_sha256(mean, std),
                len(members),
            )
        normalization["identity_sha256"] = result.identity_sha256
        normalization["moments_sha256"] = result.moments_sha256
        return torch.from_numpy(result.mean), torch.from_numpy(result.std)


def permuted_hint_members(
    members: list[tuple[str, int]],
    split: CampaignSplit,
    seed: int,
) -> list[tuple[str, int]]:
    """Rotate fitting components, retaining row position and excluding the source component."""
    component_of = {(s, g): c for s, g, c in split.components}
    groups: dict[int, list[tuple[str, int]]] = {}
    for member in members:
        groups.setdefault(component_of[member], []).append(member)
    ordered = list(np.random.default_rng(seed).permutation(sorted(groups)))
    if len(ordered) < 2:
        raise ValueError("component-permuted hints need at least two fitting components")
    mapping = {}
    for index, component in enumerate(ordered):
        donor = groups[ordered[(index + 1) % len(ordered)]]
        for position, member in enumerate(groups[component]):
            mapping[member] = donor[position % len(donor)]
    return [mapping[member] for member in members]


def build_data(
    config: dict[str, Any],
    data_root: Path,
    split_manifest: Path,
    teacher_cache: Path | None = None,
) -> SequenceData:
    """Validate mandatory chronological membership before opening training streams."""
    split = load_campaign_split(split_manifest, data_root)
    if split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA or split.valid_groups:
        raise ValueError(
            "new sequence fits require the chronological primary manifest with no fit-valid rows"
        )
    members = [("train", group) for group in split.train_groups]
    hint_dim = config["model"]["hint_dim"]
    needed = config["kd_weight"] > 0 or config["hint_weight"] > 0
    if needed and teacher_cache is None:
        raise ValueError("distillation requires a source-bound teacher cache")
    if teacher_cache is not None and not needed:
        raise ValueError("label-only control must not receive a teacher cache")
    cache = TeacherCache(teacher_cache, split, members) if teacher_cache else None
    if cache is not None and config["hint_weight"] > 0 and cache.metadata["hint_dim"] != hint_dim:
        raise ValueError("teacher/student hint widths differ")
    with split_manifest.open("rb") as stream:
        raw_bytes = stream.read(MAX_CAMPAIGN_SPLIT_BYTES + 1)
    if hashlib.sha256(raw_bytes).hexdigest() != split.sha256:
        raise ValueError("split manifest changed after membership validation")
    raw = json.loads(raw_bytes)
    hint_members = (
        permuted_hint_members(members, split, config["seed"])
        if config["hint_mode"] == "component_permuted"
        else None
    )
    training_mask = config.get("training_mask", "is_scored")
    normalization_scope = config.get("normalization_scope", "prefix")
    if training_mask not in ("is_scored", "all_after_warmup"):
        raise ValueError("unsupported sequence training mask")
    if normalization_scope not in ("prefix", "all_fit"):
        raise ValueError("unsupported normalization scope")
    normalization_members = (
        members
        if normalization_scope == "all_fit"
        else members[: min(len(members), config.get("normalization_sequences", 32))]
    )
    normalization_sources = {
        f"datasets/{source}.parquet": raw["sources"][f"datasets/{source}.parquet"]
        for source in sorted({source for source, _ in normalization_members})
    }
    provenance = {
        "split_sha256": split.sha256,
        "fold": split.fold,
        "sources": raw["sources"],
        "fit_members": [list(member) for member in members],
        "evaluation_members": [["valid", group] for group in split.evaluation_groups],
        "input_schema": "raw112-v1",
        "mask": (
            "all_after_warmup AND row_index>=99"
            if training_mask == "all_after_warmup"
            else "is_scored AND row_index>=99"
        ),
        "training_mask": training_mask,
        "normalization": {
            "scope": normalization_scope,
            "sequence_count": len(normalization_members),
            "source_fingerprints": normalization_sources,
            **(
                {"fit_prefix_sequences": len(normalization_members)}
                if normalization_scope == "prefix"
                else {}
            ),
        },
        "teacher": None if cache is None else cache.identity,
        "hint_mode": config["hint_mode"],
        "hint_members": None if hint_members is None else [list(member) for member in hint_members],
    }
    fit = SequenceRecords(
        data_root,
        members,
        hint_dim,
        cache,
        hint_members,
        training_mask=training_mask,
    )
    fit.validate_source_schema()
    evaluation = SequenceRecords(data_root, [("valid", g) for g in split.evaluation_groups], 0)
    evaluation.validate_source_schema()
    return SequenceData(
        fit,
        evaluation,
        split,
        provenance,
        config["normalization_sequences"],
    )
