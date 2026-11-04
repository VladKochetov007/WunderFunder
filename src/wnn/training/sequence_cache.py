from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.data import SEQUENCE_LENGTH
from wnn.splits import CampaignSplit, file_sha256
from wnn.training.sequence_cache_ensemble_lineage import (
    ENSEMBLE_KIND,
    is_sha256,
    validate_producer,
)
from wnn.training.sequence_cache_weighted_pool_lineage import WEIGHTED_POOL_KIND

MAX_CACHE_MANIFEST_BYTES = 32_000_000


def validate_export_schema(meta: dict[str, Any], membership: str) -> None:
    """Check the shared full-row FP32 export schema before loading payloads."""
    if not isinstance(meta, dict):
        raise ValueError("prediction metadata must be an object")
    if meta.get("input_schema") != "raw112-v1" or meta.get("membership") != membership:
        raise ValueError("unexpected prediction inputs or membership")
    if meta.get("sequence_length") != SEQUENCE_LENGTH:
        raise ValueError("prediction sequence length mismatch")
    hint_dim = meta.get("hint_dim")
    if type(hint_dim) is not int or not 0 <= hint_dim <= 4096:
        raise ValueError("unsupported hint width")


class TeacherCache:
    """Validate fitting-only producer identity and lazily read per-sequence targets."""

    def __init__(
        self,
        root: Path,
        split: CampaignSplit,
        members: list[tuple[str, int]],
    ) -> None:
        self.root = root
        manifest = root / "manifest.json"
        size = manifest.stat().st_size
        if size > MAX_CACHE_MANIFEST_BYTES:
            raise ValueError("teacher cache manifest exceeds bounded metadata size")
        with manifest.open("rb") as stream:
            raw = stream.read(MAX_CACHE_MANIFEST_BYTES + 1)
        if len(raw) != size or len(raw) > MAX_CACHE_MANIFEST_BYTES:
            raise ValueError("teacher cache manifest changed while being read")
        self.metadata = json.loads(raw)
        meta = self.metadata
        if not isinstance(meta, dict) or meta.get("schema") != "sequence-teacher-cache-v1":
            raise ValueError("incomplete or unsupported teacher cache")
        if meta.get("complete") is not True:
            raise ValueError("incomplete or unsupported teacher cache")
        validate_export_schema(meta, "fit")
        if meta.get("split_sha256") != split.sha256:
            raise ValueError("teacher cache split/input ancestry mismatch")
        sequences = meta.get("sequences")
        if not isinstance(sequences, list):
            raise ValueError("teacher cache sequence entries are malformed")
        observed: list[tuple[str, int]] = []
        for entry in sequences:
            if not isinstance(entry, dict):
                raise ValueError("teacher cache sequence entry is malformed")
            source, group = entry.get("split"), entry.get("row_group")
            if source not in ("train", "valid") or type(group) is not int or group < 0:
                raise ValueError("teacher cache sequence identity is malformed")
            if not isinstance(entry.get("file"), str) or not is_sha256(entry.get("sha256")):
                raise ValueError("teacher cache file reference is malformed")
            observed.append((source, group))
        if len(set(observed)) != len(observed) or set(observed) != set(members):
            raise ValueError("teacher cache must cover exactly fitting membership")
        if len({entry["file"] for entry in sequences}) != len(sequences):
            raise ValueError("teacher cache members must reference distinct array files")
        producer = meta.get("producer")
        if not isinstance(producer, dict):
            raise ValueError("teacher cache producer is malformed")
        expected_sources = None
        if split.path.is_file():
            split_record = json.loads(split.path.read_text(encoding="utf-8"))
            raw_sources = split_record.get("sources") if isinstance(split_record, dict) else None
            expected_sources = raw_sources if isinstance(raw_sources, dict) else None
        validate_producer(
            producer,
            split.sha256,
            members,
            evaluation_members=[
                (split.evaluation_source, group) for group in split.evaluation_groups
            ],
            primary_split_sha256=split.primary_split_sha256,
            expected_sources=expected_sources,
        )
        self.entries = dict(zip(observed, sequences, strict=True))
        manifest_hash = hashlib.sha256(raw).hexdigest()
        if producer.get("kind") in (ENSEMBLE_KIND, WEIGHTED_POOL_KIND):
            self.identity = {
                "manifest_sha256": manifest_hash,
                "composition_sha256": producer["composition_sha256"],
                "producer": producer,
            }
        else:
            self.identity = {
                "manifest_sha256": manifest_hash,
                "checkpoint_sha256": producer["checkpoint_sha256"],
                "producer": producer,
            }
        for entry in self.entries.values():
            path = self._entry_path(entry)
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"teacher cache hash mismatch: {entry['file']}")

    def _entry_path(self, entry: dict[str, Any]) -> Path:
        """Resolve one canonical flat cache filename without following escapes."""
        relative = Path(entry["file"])
        root = self.root.resolve()
        path = self.root / relative
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name in (".", ".."):
            raise ValueError("teacher cache path escapes its root")
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError("teacher cache path escapes its root")
        return path

    def verify_unchanged(self) -> None:
        """Recheck the manifest and every source array after a streaming consumer pass."""
        manifest_hash = file_sha256(self.root / "manifest.json")
        if manifest_hash != self.identity["manifest_sha256"]:
            raise ValueError("teacher cache manifest changed during merge")
        for entry in self.entries.values():
            path = self._entry_path(entry)
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"teacher cache changed during merge: {entry['file']}")

    def read(self, split: str, group: int, hint_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one hash-checked cache entry without materializing the full cache."""
        entry = self.entries[(split, group)]
        values = np.load(self._entry_path(entry), mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float32 or values.shape != (
            SEQUENCE_LENGTH,
            2 + self.metadata["hint_dim"],
        ):
            raise ValueError("teacher target shape/dtype mismatch")
        if not np.isfinite(values).all():
            raise ValueError("nonfinite teacher cache")
        predictions = torch.from_numpy(np.array(values[:, :2], copy=True))
        hints = torch.zeros(SEQUENCE_LENGTH, hint_dim)
        if hint_dim and self.metadata["hint_dim"] == hint_dim:
            hints = torch.from_numpy(np.array(values[:, 2:], copy=True))
        return predictions, hints
