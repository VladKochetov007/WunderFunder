from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class RecordedOrder(Sampler[int]):
    """Record the actual sampler traversal once without consuming any extra RNG."""

    def __init__(self, sampler: Sampler[int]) -> None:
        self.sampler = sampler
        self.indices: list[int] = []

    def __iter__(self) -> Iterator[int]:
        self.indices.clear()
        for index in self.sampler:
            self.indices.append(int(index))
            yield index

    def __len__(self) -> int:
        return len(self.sampler)

    def identity(self, batch_size: int) -> dict[str, object]:
        """Bind retained and dropped positions within the exact constructed fit dataset."""
        retained = len(self.indices) // batch_size * batch_size
        values = np.asarray(self.indices[:retained], dtype="<i8")
        return {
            "retained_dataset_indices_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "retained_count": retained,
            "unique_retained_count": len(set(self.indices[:retained])),
            "dropped_dataset_indices": self.indices[retained:],
        }


def summarize_masses(masses: Sequence[float]) -> dict[str, float | int]:
    """Describe a frozen empirical fit-batch reference, including sampling variability."""
    values = np.asarray(masses, dtype=np.float64)
    if not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("calibration needs finite nonnegative update masses")
    reference = float(values.mean())
    if reference <= 0:
        raise ValueError("calibration mass must be positive")
    return {
        "reference_mass": reference,
        "updates": len(values),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std()),
        "zero_mass_updates": int((values == 0).sum()),
    }
