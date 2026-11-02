from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from wnn.splits import CampaignSplit, file_sha256
from wnn.training.sequence_cache import TeacherCache
from wnn.training.sequence_cache_ensemble import _parent_records


class DisagreementCache:
    """Three source-qualified predictions and bounded per-row KD weights."""

    def __init__(self, root: Path, split: CampaignSplit, members: list[tuple[str, int]]) -> None:
        descriptor = root / "manifest.json"
        if descriptor.is_symlink() or descriptor.stat().st_size > 16384:
            raise ValueError("invalid disagreement cache descriptor")
        size = descriptor.stat().st_size
        with descriptor.open("rb") as stream:
            raw = stream.read(16385)
        if len(raw) != size or len(raw) > 16384:
            raise ValueError("invalid disagreement cache descriptor")
        meta = json.loads(raw)
        if not isinstance(meta, dict) or meta.get("schema") != "sequence-disagreement-input-v1":
            raise ValueError("invalid disagreement cache schema")
        inputs = meta.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != 3:
            raise ValueError("exactly three teacher caches are required")
        caches = []
        for item in inputs:
            if not isinstance(item, dict) or set(item) != {"path", "manifest_sha256"}:
                raise ValueError("each input must contain path and manifest_sha256")
            path = Path(item["path"])
            if not path.is_absolute() or path.is_symlink():
                raise ValueError("teacher roots must be absolute non-symlink paths")
            cache = TeacherCache(path, split, members)
            if cache.identity["manifest_sha256"] != item["manifest_sha256"]:
                raise ValueError("teacher manifest identity mismatch")
            if cache.metadata["hint_dim"] != 0:
                raise ValueError("disagreement inputs must contain only predictions")
            caches.append(cache)
        if len({str(cache.root.resolve()) for cache in caches}) != 3:
            raise ValueError("teacher cache roots must be distinct")
        self.caches = caches
        self.descriptor = descriptor
        self.metadata: dict[str, Any] = {"hint_dim": 0}
        producer = _parent_records(caches)
        self.identity = {
            "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "composition_sha256": producer["composition_sha256"],
            "producer": producer,
        }

    def read_weighted(
        self, source: str, group: int, mode: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute raw FP32 mean and a bounded disagreement weight per row."""
        values = []
        for cache in self.caches:
            entry = cache.entries[(source, group)]
            if file_sha256(cache._entry_path(entry)) != entry["sha256"]:
                raise ValueError("teacher prediction changed before read")
            value, _ = cache.read(source, group, 0)
            if file_sha256(cache._entry_path(entry)) != entry["sha256"]:
                raise ValueError("teacher prediction changed during read")
            values.append(value)
        mean = ((values[0] + values[1]) + values[2]) / 3.0
        variance = sum((value - mean).square().mean(dim=-1) for value in values) / 3.0
        fraction = variance / (variance + 0.01)
        if not torch.isfinite(fraction).all():
            raise ValueError("nonfinite teacher disagreement")
        if mode == "constant":
            weights = torch.ones_like(fraction)
        elif mode == "up":
            weights = 1.0 + 0.5 * fraction
        elif mode == "down":
            weights = 1.0 - 0.5 * fraction
        else:
            raise ValueError("unknown disagreement mode")
        return mean, torch.zeros(mean.shape[0], 0), weights

    def verify_unchanged(self) -> None:
        """Ensure descriptor and parent caches remain bound after fitting."""
        if file_sha256(self.descriptor) != self.identity["manifest_sha256"]:
            raise ValueError("disagreement descriptor changed during fitting")
        for cache in self.caches:
            cache.verify_unchanged()
