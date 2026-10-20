from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.data import SEQUENCE_LENGTH
from wnn.splits import CampaignSplit, file_sha256

# Cache manifests contain heterogeneous JSON fields; arrays are strictly FP32.


def validate_producer(
    producer: dict[str, Any],
    split_hash: str,
    members: list[tuple[str, int]],
) -> None:
    """Require same-fold fitting ancestry through the entire distillation chain."""
    depth = 0
    current = producer
    while current is not None:
        if depth > 32:
            raise ValueError("teacher ancestry exceeds supported depth")
        if current["split_sha256"] != split_hash:
            raise ValueError("teacher ancestry crosses evaluation folds")
        if current["fit_members"] != [list(member) for member in members]:
            raise ValueError("teacher fitted membership differs from the student's clean split")
        if re.fullmatch(r"[0-9a-f]{64}", current.get("checkpoint_sha256", "")) is None:
            raise ValueError("invalid teacher checkpoint hash")
        ancestor = current.get("teacher")
        current = None if ancestor is None else ancestor["producer"]
        depth += 1


def validate_export_schema(meta: dict[str, Any], membership: str) -> None:
    """Check the shared full-row FP32 export schema before loading payloads."""
    if meta["input_schema"] != "raw112-v1" or meta["membership"] != membership:
        raise ValueError("unexpected prediction inputs or membership")
    if meta["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError("prediction sequence length mismatch")
    if type(meta["hint_dim"]) is not int or not 0 <= meta["hint_dim"] <= 4096:
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
        if manifest.stat().st_size > 32_000_000:
            raise ValueError("teacher cache manifest exceeds bounded metadata size")
        self.metadata: dict[str, Any] = json.loads(manifest.read_text())
        meta = self.metadata
        if meta.get("schema") != "sequence-teacher-cache-v1" or meta.get("complete") is not True:
            raise ValueError("incomplete or unsupported teacher cache")
        validate_export_schema(meta, "fit")
        if meta["split_sha256"] != split.sha256 or meta["input_schema"] != "raw112-v1":
            raise ValueError("teacher cache split/input ancestry mismatch")
        observed = [(entry["split"], entry["row_group"]) for entry in meta["sequences"]]
        if len(set(observed)) != len(observed) or set(observed) != set(members):
            raise ValueError("teacher cache must cover exactly fitting membership")
        producer = meta["producer"]
        validate_producer(producer, split.sha256, members)
        self.entries = dict(zip(observed, meta["sequences"], strict=True))
        self.identity = {
            "manifest_sha256": file_sha256(manifest),
            "checkpoint_sha256": producer["checkpoint_sha256"],
            "producer": producer,
        }
        for entry in self.entries.values():
            path = root / entry["file"]
            if path.parent.resolve() != root.resolve() or path.is_symlink():
                raise ValueError("teacher cache path escapes its root")
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"teacher cache hash mismatch: {entry['file']}")

    def read(self, split: str, group: int, hint_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one hash-checked cache entry without materializing the full cache."""
        entry = self.entries[(split, group)]
        values = np.load(self.root / entry["file"], mmap_mode="r", allow_pickle=False)
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
