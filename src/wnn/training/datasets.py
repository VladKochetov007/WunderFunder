from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from wnn.data import N_FEATURES
from wnn.spread_aux import PackedSpreadCache
from wnn.training.environment import TrainingEnvironment
from wnn.training.row_views import thin_rows


class Sequences(Dataset):
    def __init__(
        self,
        split: str,
        indices: Iterable[int],
        teachers: list[tuple[str, float]] | None = None,
        prop_cache: str | Path | None = None,
        quote_path: bool = False,
        quote_cache_root: str | Path | None = None,
        spread_cache_root: str | Path | None = None,
        thin: tuple[float, float, float] | tuple[float, float, float, str] | None = None,
        distill_cache_root: str | Path | None = None,
        distill_cache_variant: str = "source",
        distill_cache_metadata: dict[str, Any] | None = None,
        include_family_targets: bool = False,
        *,
        environment: TrainingEnvironment | None = None,
    ) -> None:
        """Init."""
        self.environment = environment if environment is not None else TrainingEnvironment()
        self.path = self.environment.root / "datasets" / f"{split}.parquet"
        self.thin = thin
        self.indices = list(indices)
        self._file = None
        self.prop_cache, self._prop = (prop_cache, None)
        self.teachers = teachers
        self.split = split
        self._teach = None
        self.quote_path = quote_path
        self.quote_cache_root = (
            Path(quote_cache_root) if quote_cache_root else self.environment.root / "cache"
        )
        self._quote = None
        self.spread_cache_root = Path(spread_cache_root) if spread_cache_root else None
        self._spread = None
        self.distill_cache_root = Path(distill_cache_root) if distill_cache_root else None
        self.distill_cache_variant = distill_cache_variant
        self.distill_cache_metadata = distill_cache_metadata
        self.include_family_targets = include_family_targets
        self._distill_main = None
        self._distill_families = None
        self._distill_pos = None
        if (teachers or self.distill_cache_root) and quote_path:
            raise ValueError("distillation and quote-path auxiliaries share target columns")
        if self.distill_cache_root and teachers:
            raise ValueError("use either a legacy teacher list or a preweighted distillation cache")
        if self.distill_cache_root:
            if self.distill_cache_metadata is None:
                raise ValueError("preweighted distillation cache metadata was not validated")
            files = self.distill_cache_metadata["metadata"]["files"]
            key = f"main_{self.distill_cache_variant}_{split}"
            rows = list(map(int, files[key]["row_groups"]))
            self._distill_pos = {group: position for position, group in enumerate(rows)}
            absent = set(map(int, self.indices)) - self._distill_pos.keys()
            if absent:
                raise ValueError(
                    f"distillation cache does not contain selected {split} fitting rows"
                )

    def __len__(self) -> int:
        """Len."""
        return len(self.indices)

    def batch_tensor_bytes(self, batch_size: int, *, heuristics: bool | None = None) -> int:
        """Size returned tensors from their schema without reading a sample or RNG.

        Cache headers supply auxiliary target widths; teacher predictions are
        averaged, so their column count is included once, not per teacher.
        Event/normalization features computed on device are not loader tensors.
        """
        if batch_size <= 0:
            raise ValueError("loader batch size must be positive")
        features = N_FEATURES
        use_heuristics = self.environment.inputs.heuristics if heuristics is None else heuristics
        if use_heuristics:
            from wnn.heuristics import K

            features += K
        targets = 2
        if self.teachers:
            widths = set()
            for name, _ in self.teachers:
                values = np.load(
                    self.environment.root / "cache" / f"pred_{name}_{self.split}.npy", mmap_mode="r"
                )
                if values.ndim != 3 or values.shape[1] != self.environment.sequence_length:
                    raise ValueError("teacher cache does not contain full sequences")
                widths.add(values.shape[2])
            if len(widths) != 1:
                raise ValueError("teacher target widths differ")
            targets += widths.pop()
        elif self.distill_cache_root:
            targets += 2 + (8 if self.include_family_targets else 0)
        elif self.quote_path:
            values = np.load(self.quote_cache_root / f"quote_path_{self.split}.npy", mmap_mode="r")
            if values.ndim != 3 or values.shape[1] != self.environment.sequence_length:
                raise ValueError("quote cache does not contain full sequences")
            targets += values.shape[2]
        row_bytes = (features + targets) * np.dtype(np.float32).itemsize
        row_bytes += np.dtype(np.bool_).itemsize
        if self.prop_cache is not None:
            row_bytes += np.dtype(np.float32).itemsize
        if self.spread_cache_root is not None:
            row_bytes += np.dtype(np.uint8).itemsize
        sequence_bytes = self.environment.sequence_length * row_bytes
        if self.thin is not None:
            sequence_bytes += (
                self.environment.sequence_length
                if len(self.thin) == 4 and self.thin[3] == "retained-mass"
                else np.dtype(np.int64).itemsize
            )
        return batch_size * sequence_bytes

    def __getitem__(self, i: Any) -> tuple[torch.Tensor, ...]:
        """Getitem."""
        if self._file is None:
            self._file = self.environment.sequence_file(self.path)
        seq = self._file.read(self.indices[i])
        scored = seq.scored if seq.scored is not None else np.zeros(len(seq.features), bool)
        x = seq.features
        if self.environment.inputs.heuristics:
            from wnn.heuristics import heuristic_features

            x = np.concatenate([x, heuristic_features(x)], axis=1)
        y = seq.targets
        if self.teachers:
            if self._teach is None:
                rg = np.load(self.environment.root / "cache" / f"{self.split}_targets.npz")[
                    "row_group"
                ]
                self._pos = {int(g): i for i, g in enumerate(rg)}
                self._teach = [
                    (
                        np.load(
                            self.environment.root / "cache" / f"pred_{n}_{self.split}.npy",
                            mmap_mode="r",
                        ),
                        sc,
                    )
                    for n, sc in self.teachers
                ]
            j = self._pos[int(self.indices[i])]
            t = sum((sc * np.asarray(arr[j], dtype=np.float32) for arr, sc in self._teach)) / len(
                self._teach
            )
            y = np.concatenate([y, t.astype(np.float32)], axis=1)
        if self.distill_cache_root:
            j = self._distill_pos[int(self.indices[i])]
            if self._distill_main is None:
                self._distill_main = np.load(
                    self.distill_cache_root / f"main_{self.distill_cache_variant}_{self.split}.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
            y = np.concatenate([y, np.asarray(self._distill_main[j], dtype=np.float32)], axis=1)
            if self.include_family_targets:
                if self._distill_families is None:
                    self._distill_families = np.load(
                        self.distill_cache_root / f"families_{self.split}.npy",
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                y = np.concatenate(
                    [y, np.asarray(self._distill_families[j], dtype=np.float32)], axis=1
                )
        if self.quote_path:
            if self._quote is None:
                self._quote = np.load(
                    self.quote_cache_root / f"quote_path_{self.split}.npy", mmap_mode="r"
                )
            y = np.concatenate(
                [y, np.asarray(self._quote[self.indices[i]], dtype=np.float32)], axis=1
            )
        if self.prop_cache is not None:
            if self._prop is None:
                self._prop = np.load(self.prop_cache, mmap_mode="r")
            p = np.array(self._prop[self.indices[i]], dtype=np.float32)
            result = (
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(scored),
                torch.from_numpy(p),
            )
        else:
            result = (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(scored))
        if self.spread_cache_root is not None:
            if self._spread is None:
                self._spread = PackedSpreadCache(self.spread_cache_root, self.split)
            result += (torch.from_numpy(self._spread.read(self.indices[i])),)
        if self.thin is not None:
            result = thin_rows(result, self.thin, environment=self.environment)
        return result
