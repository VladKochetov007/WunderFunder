"""Synthetic inputs and value comparisons for the PR81 receiver loop gate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

# Model/optimizer state and JSON telemetry have heterogeneous nested value types.
TOLERANCE = 1e-5
IGNORED_RECORD_FIELDS = {
    "window_seconds",
    "valid_rows_per_second",
    "forwarded_rows_per_second",
    "peak_memory_bytes",
    "optimizer_seconds",
}


class SyntheticDataset:
    """Two deterministic full-length sequences with 99 warmup rows each."""

    def __init__(self) -> None:
        from wnn.training.sequence_fit_utils import FEATURE_COUNT, SEQUENCE_ROWS

        self._records = [
            self._make_record(index, FEATURE_COUNT, SEQUENCE_ROWS) for index in range(2)
        ]
        self._padding = {
            "x": torch.zeros((SEQUENCE_ROWS, FEATURE_COUNT), dtype=torch.float32),
            "y": torch.zeros((SEQUENCE_ROWS, 2), dtype=torch.float32),
            "mask": torch.zeros(SEQUENCE_ROWS, dtype=torch.bool),
            "teacher": torch.zeros((SEQUENCE_ROWS, 2), dtype=torch.float32),
            "hints": torch.zeros((SEQUENCE_ROWS, 0), dtype=torch.float32),
        }

    @staticmethod
    def _make_record(index: int, feature_count: int, row_count: int) -> dict[str, Tensor]:
        rows = torch.arange(row_count, dtype=torch.float32)
        columns = torch.arange(feature_count, dtype=torch.float32)
        phase = (rows[:, None] + index * 3.0) * (columns[None, :] + 1.0) * 0.00017
        features = torch.sin(phase) + 0.25 * torch.cos(phase * 0.37)
        label_phase = rows + index * 17.0
        labels = torch.stack(
            (
                0.8 * torch.sin(label_phase * 0.017) + 0.15,
                0.7 * torch.cos(label_phase * 0.013) - 0.12,
            ),
            dim=-1,
        )
        return {
            "x": features.contiguous(),
            "y": labels.contiguous(),
            "mask": rows >= 99,
            "teacher": torch.zeros_like(labels),
            "hints": torch.zeros((row_count, 0), dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if index == -1:
            return self._padding
        if not 0 <= index < len(self._records):
            raise IndexError(index)
        return self._records[index]

    @property
    def valid_rows(self) -> int:
        """Return the total post-warmup row count used to validate the fixture."""
        return sum(int(record["mask"].sum()) for record in self._records)


class MemoryRankLog:
    """Capture one receiver update without creating a filesystem artifact."""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def write(self, record: Mapping[str, object]) -> None:
        """Retain an independent shallow copy of one update record."""
        self.records.append(dict(record))


def compare(actual: Any, expected: Any, label: str) -> float:
    """Require equal structure and finite numeric values within absolute tolerance."""
    if isinstance(actual, Tensor) and isinstance(expected, Tensor):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError(f"{label} tensor shape or dtype differs")
        if actual.is_floating_point():
            if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
                raise AssertionError(f"{label} contains nonfinite tensor values")
            difference = float((actual - expected).abs().max().item()) if actual.numel() else 0.0
            if difference > TOLERANCE:
                raise AssertionError(f"{label} differs by {difference:.8g}")
            return difference
        if not torch.equal(actual, expected):
            raise AssertionError(f"{label} integer or Boolean tensor differs")
        return 0.0
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        if set(actual) != set(expected):
            raise AssertionError(f"{label} mapping keys differ")
        return max(
            (compare(actual[key], expected[key], f"{label}.{key}") for key in actual),
            default=0.0,
        )
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"{label} sequence structure differs")
        return max(
            (
                compare(left, right, f"{label}[{index}]")
                for index, (left, right) in enumerate(zip(actual, expected, strict=True))
            ),
            default=0.0,
        )
    if isinstance(actual, bool) or isinstance(expected, bool):
        if actual is not expected:
            raise AssertionError(f"{label} Boolean value differs")
        return 0.0
    if isinstance(actual, (int, str, type(None))) or isinstance(expected, (int, str, type(None))):
        if actual != expected:
            raise AssertionError(f"{label} value differs: {actual!r} != {expected!r}")
        return 0.0
    if isinstance(actual, (float, int)) and isinstance(expected, (float, int)):
        if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
            raise AssertionError(f"{label} is nonfinite")
        difference = abs(float(actual) - float(expected))
        if difference > TOLERANCE:
            raise AssertionError(f"{label} differs by {difference:.8g}")
        return difference
    if actual != expected:
        raise AssertionError(f"{label} differs")
    return 0.0


def compare_update_record(stock: dict[str, object], experiment: dict[str, object]) -> float:
    """Compare canonical update fields while excluding timing-only telemetry."""
    expected = {key: value for key, value in stock.items() if key not in IGNORED_RECORD_FIELDS}
    missing = set(expected) - set(experiment)
    if missing:
        raise AssertionError(f"experiment update record omits canonical fields: {sorted(missing)}")
    observed = {key: experiment[key] for key in expected}
    extra = set(experiment) - set(stock)
    if extra != {"primary_loss", "impact81_auxiliary"}:
        raise AssertionError(f"unexpected control-engine telemetry fields: {sorted(extra)}")
    difference = compare(observed, expected, "rank log")
    difference = max(
        difference,
        compare(experiment["primary_loss"], stock["total_loss"], "primary loss"),
        compare(
            experiment["impact81_auxiliary"]["weighted_auxiliary_loss"],
            0.0,
            "control auxiliary loss",
        ),
    )
    return difference
