from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.data import SEQUENCE_LENGTH
from wnn.splits import CampaignSplit, file_sha256
from wnn.training.sequence_reference_contract import (
    is_sha256,
    members_sha256,
    read_object,
    validate_reference_identity,
)

# Manifest values are heterogeneous JSON; tensor payloads have an exact FP32 schema.
__all__ = ["ReferenceCache", "validate_reference_identity"]


class ReferenceCache:
    """Read a frozen full-row TL3 reference for one explicit fit/evaluation cohort."""

    def __init__(
        self,
        root: Path,
        split: CampaignSplit,
        members: list[tuple[str, int]],
        membership: str = "fit",
    ) -> None:
        self.root = Path(root)
        if membership not in ("fit", "evaluation"):
            raise ValueError("unsupported reference cache membership")
        meta, manifest_hash = read_object(self.root / "manifest.json")
        self.metadata = meta
        if (
            meta.get("schema") != "sequence-reference-cache-v1"
            or meta.get("complete") is not True
            or meta.get("membership") != membership
            or meta.get("sequence_length") != SEQUENCE_LENGTH
            or meta.get("dtype") != "float32"
            or meta.get("prediction_width") != 2
            or meta.get("hint_dim") != 0
            or type(meta.get("hint_dim")) is not int
            or meta.get("split_sha256") != split.sha256
        ):
            raise ValueError("reference cache schema, membership or split mismatch")
        fitting = [("train", g) for g in split.train_groups]
        fitting += [("valid", g) for g in split.valid_groups]
        evaluation = [("valid", g) for g in split.evaluation_groups]
        expected = fitting if membership == "fit" else evaluation
        if set(members) != set(expected) or len(members) != len(expected):
            raise ValueError("reference consumer must use the full declared cohort")
        self.identity = {
            "manifest_sha256": manifest_hash,
            "reference_sha256": meta.get("reference_sha256"),
            "producer": meta.get("producer"),
        }
        validate_reference_identity(self.identity, split.sha256, fitting)
        if self.identity["producer"]["evaluation_members_sha256"] != members_sha256(evaluation):
            raise ValueError("reference evaluation membership differs from the actual split")
        entries = meta.get("sequences")
        if not isinstance(entries, list) or len(entries) != len(expected):
            raise ValueError("reference cache must contain exactly the declared sequence set")
        self.entries: dict[tuple[str, int], dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("reference sequence metadata must be an object")
            source, group = entry.get("split"), entry.get("row_group")
            if source not in ("train", "valid") or type(group) is not int or group < 0:
                raise ValueError("reference cache sequence identity is malformed")
            key = (source, group)
            if key in self.entries or not is_sha256(entry.get("sha256")):
                raise ValueError("reference cache repeats a sequence or lacks a payload digest")
            if not isinstance(entry.get("file"), str):
                raise ValueError("reference cache array path must be a string")
            self.entries[key] = entry
        if set(self.entries) != set(expected):
            raise ValueError("reference cache cohort differs from the declared membership")
        if len({entry.get("file") for entry in entries}) != len(entries):
            raise ValueError("reference sequences must reference distinct files")
        self.verify_unchanged()

    def _entry_path(self, entry: dict[str, Any]) -> Path:
        """Resolve a flat payload filename without traversals or symlinks."""
        name = entry.get("file")
        if not isinstance(name, str) or not name.endswith(".npy"):
            raise ValueError("reference array filename must be a string ending in.npy")
        relative = Path(name)
        path = self.root / relative
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name in (".", "..")
            or path.is_symlink()
            or path.resolve().parent != self.root.resolve()
        ):
            raise ValueError("reference cache path escapes its root")
        return path

    def verify_unchanged(self) -> None:
        """Bind immutable metadata and every referenced array to their recorded bytes."""
        if file_sha256(self.root / "manifest.json") != self.identity["manifest_sha256"]:
            raise ValueError("reference cache manifest changed")
        for entry in self.entries.values():
            if file_sha256(self._entry_path(entry)) != entry["sha256"]:
                raise ValueError("reference array differs from its recorded digest")

    def read(self, source: str, group: int, hint_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two raw prediction channels and no privileged representation hints."""
        if type(hint_dim) is not int or hint_dim != 0:
            raise ValueError("TL3 reference caches do not contain hidden hints")
        entry = self.entries[(source, group)]
        # Verify the exact bytes consumed, even if storage changes after admission.
        maximum_bytes = SEQUENCE_LENGTH * 2 * 4 + 4096
        with self._entry_path(entry).open("rb") as stream:
            raw = stream.read(maximum_bytes + 1)
        if len(raw) > maximum_bytes or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("reference payload changed after cache admission")
        values = np.load(io.BytesIO(raw), allow_pickle=False)
        if values.dtype != np.float32 or values.shape != (SEQUENCE_LENGTH, 2):
            raise ValueError("reference array must have shape20000x2 and float32 dtype")
        if not np.isfinite(values).all():
            raise ValueError("reference array contains nonfinite predictions")
        return torch.from_numpy(np.array(values, copy=True)), torch.zeros(
            SEQUENCE_LENGTH, 0, dtype=torch.float32
        )
