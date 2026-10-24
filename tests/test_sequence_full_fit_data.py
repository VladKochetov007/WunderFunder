from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wnn.data import N_FEATURES, SEQUENCE_LENGTH, WARMUP
from wnn.training.sequence_data import SequenceRecords
from wnn.training.sequence_normalization import load_or_compute_full_fit_moments


class _FakeSequenceFile:
    def __init__(self, *, has_targets: bool, has_mask: bool) -> None:
        self.has_targets = has_targets
        self.has_mask = has_mask


class _FakeSource:
    def __init__(self, *, has_targets: bool, has_mask: bool, scored: torch.Tensor) -> None:
        self.indices = [7]
        self._file = _FakeSequenceFile(has_targets=has_targets, has_mask=has_mask)
        self.environment = SimpleNamespace(sequence_file=lambda _: self._file)
        self.path = Path("unused.parquet")
        self.scored = scored

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert index == 0
        return (
            torch.ones(SEQUENCE_LENGTH, N_FEATURES),
            torch.ones(SEQUENCE_LENGTH, 2),
            self.scored,
        )


def _records(tmp_path: Path, *, training_mask: str, source: _FakeSource) -> SequenceRecords:
    records = SequenceRecords(tmp_path, [("train", 7)], 0, training_mask=training_mask)
    records.sources["train"] = source
    records.positions[("train", 7)] = 0
    return records


def test_all_after_warmup_accepts_training_source_without_scoring_column(tmp_path: Path) -> None:
    source = _FakeSource(
        has_targets=True,
        has_mask=False,
        scored=torch.zeros(SEQUENCE_LENGTH, dtype=torch.bool),
    )
    records = _records(tmp_path, training_mask="all_after_warmup", source=source)
    records.validate_source_schema()

    sample = records[0]

    assert not sample["mask"][:WARMUP].any()
    assert sample["mask"][WARMUP:].all()


def test_default_scored_training_mask_still_requires_is_scored(tmp_path: Path) -> None:
    source = _FakeSource(
        has_targets=True,
        has_mask=False,
        scored=torch.zeros(SEQUENCE_LENGTH, dtype=torch.bool),
    )
    records = _records(tmp_path, training_mask="is_scored", source=source)

    with pytest.raises(ValueError, match="is_scored columns"):
        records.validate_source_schema()


def _source_fingerprints() -> dict[str, dict[str, int]]:
    return {
        "datasets/train.parquet": {"bytes": 101, "mtime_ns": 202},
        "datasets/valid.parquet": {"bytes": 303, "mtime_ns": 404},
    }


def _fit_members() -> list[tuple[str, int]]:
    return [("train", 2), ("valid", 5)]


def _reader_for(calls: list[tuple[str, int]]) -> Callable[[str, int], np.ndarray]:
    def read(source: str, group: int) -> np.ndarray:
        calls.append((source, group))
        base = 2.0 if source == "train" else 10.0
        return np.full((3, 2), base + group, dtype=np.float32)

    return read


def test_all_fit_normalization_reads_only_source_qualified_fitting_members(tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []
    result = load_or_compute_full_fit_moments(
        tmp_path / "cache" / "sequence_normalization",
        _source_fingerprints(),
        "a" * 64,
        _fit_members(),
        _reader_for(calls),
        feature_count=2,
    )

    assert calls == _fit_members()
    assert not {("valid", 9)}.intersection(calls)
    assert result.sequence_count == 2
    np.testing.assert_allclose(result.mean, np.asarray([9.5, 9.5], dtype=np.float32))


def test_all_fit_cache_rejects_stale_identity(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache" / "sequence_normalization"
    split_sha256 = "b" * 64
    load_or_compute_full_fit_moments(
        cache_dir,
        _source_fingerprints(),
        split_sha256,
        _fit_members(),
        _reader_for([]),
        feature_count=2,
    )
    cache_path = next(cache_dir.glob("*.json"))
    payload = json.loads(cache_path.read_text())
    payload["identity"]["split_sha256"] = "c" * 64
    cache_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="identity does not match"):
        load_or_compute_full_fit_moments(
            cache_dir,
            _source_fingerprints(),
            split_sha256,
            _fit_members(),
            _reader_for([]),
            feature_count=2,
        )


def test_all_fit_cache_rejects_corrupt_moments_sha(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache" / "sequence_normalization"
    arguments = (
        cache_dir,
        _source_fingerprints(),
        "d" * 64,
        _fit_members(),
    )
    load_or_compute_full_fit_moments(*arguments, _reader_for([]), feature_count=2)
    cache_path = next(cache_dir.glob("*.json"))
    payload = json.loads(cache_path.read_text())
    payload["mean"][0] += 1.0
    cache_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="moments SHA-256 mismatch"):
        load_or_compute_full_fit_moments(
            *arguments,
            _reader_for([]),
            feature_count=2,
        )
