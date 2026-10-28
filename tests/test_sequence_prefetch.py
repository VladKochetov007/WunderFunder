"""CPU checks for bounded sequence prefetch ordering, failures, and cleanup."""

from __future__ import annotations

import json
import random
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from wnn.training.sequence_config import DEFAULTS, load_config
from wnn.training.sequence_fit_utils import SequenceItem, collate_sequences
from wnn.training.sequence_prefetch import SequenceBatchPrefetcher


class _DeterministicRecords:
    """Small RNG-free dataset double with optional bounded producer failure."""

    def __init__(self, fail_index: int | None = None) -> None:
        self.fail_index = fail_index
        self.accessed: list[int] = []
        self.active = 0
        self._lock = threading.Lock()
        self.failure_seen = threading.Event()

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> SequenceItem:
        with self._lock:
            self.accessed.append(index)
            self.active += 1
        try:
            if index == self.fail_index:
                self.failure_seen.set()
                raise RuntimeError(f"synthetic producer failure at {index}")
            value = 0 if index < 0 else index
            return {
                "x": torch.full((3, 4), value, dtype=torch.float32),
                "y": torch.full((3, 2), value + 1, dtype=torch.float32),
                "mask": torch.tensor([True, False, True]),
                "teacher": torch.zeros((3, 2), dtype=torch.float32),
                "hints": torch.zeros((3, 0), dtype=torch.float32),
            }
        finally:
            with self._lock:
                self.active -= 1


def _write_recipe(path: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """Load one tiny JSON recipe through the production config validator."""
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_config(path)


def _assert_batches_equal(left: SequenceItem, right: SequenceItem) -> None:
    """Require every complete collated training tensor to match synchronously."""
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def _assert_numpy_rng_equal(left: tuple[object, ...], right: tuple[object, ...]) -> None:
    """Compare NumPy's array-containing RNG state without ambiguous array equality."""
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_prefetch_matches_synchronous_batches_order_and_rng_on_cpu() -> None:
    """One worker preserves complete batch tensors, sentinels, order, and caller RNG state."""
    dataset = _DeterministicRecords()
    indices = [4, 2, -1, 7, 1]
    starts = range(0, len(indices), 2)
    expected = [
        (tuple(indices[start : start + 2]), collate_sequences(dataset, indices[start : start + 2]))
        for start in starts
    ]
    torch_state = torch.get_rng_state()
    python_state = random.getstate()
    numpy_state = np.random.get_state()  # noqa: NPY002 - check caller RNG state is preserved

    prefetched: list[tuple[tuple[int, ...], SequenceItem]] = []
    with SequenceBatchPrefetcher(dataset, indices, 2, torch.device("cpu")) as batches:
        for item in batches:
            item.wait_on_current_stream()
            prefetched.append((item.indices, item.batch))

    assert len(prefetched) == len(expected)
    for (actual_ids, actual), (expected_ids, expected_batch) in zip(
        prefetched, expected, strict=True
    ):
        assert actual_ids == expected_ids
        _assert_batches_equal(actual, expected_batch)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    _assert_numpy_rng_equal(
        np.random.get_state(),  # noqa: NPY002 - check caller RNG state is preserved
        numpy_state,
    )


def test_prefetch_propagates_producer_failure_and_joins_worker() -> None:
    """A failed future reaches the consumer and close joins the sole producer."""
    dataset = _DeterministicRecords(fail_index=2)
    batches = SequenceBatchPrefetcher(dataset, [0, 1, 2, 3], 2, torch.device("cpu"))
    try:
        assert next(batches).indices == (0, 1)
        with pytest.raises(RuntimeError, match="synthetic producer failure at 2"):
            next(batches)
    finally:
        batches.close()
    assert batches._closed
    assert batches._future is None
    assert dataset.active == 0
    assert all(not worker.is_alive() for worker in batches._executor._threads)


def test_partial_prefetch_close_cancels_lookahead_and_releases_future() -> None:
    """Closing after one batch bounds source reads to the active and single next batch."""
    dataset = _DeterministicRecords()
    batches = SequenceBatchPrefetcher(dataset, list(range(8)), 2, torch.device("cpu"))
    first = next(batches)
    assert first.indices == (0, 1)
    batches.close()
    batches.close()
    assert batches._future is None
    assert dataset.active == 0
    assert len(dataset.accessed) <= 4
    assert all(not worker.is_alive() for worker in batches._executor._threads)


def test_partial_close_propagates_failure_from_unconsumed_lookahead() -> None:
    """Early profile cleanup reports an already-running failed producer."""
    dataset = _DeterministicRecords(fail_index=2)
    batches = SequenceBatchPrefetcher(dataset, [0, 1, 2, 3], 2, torch.device("cpu"))
    try:
        assert next(batches).indices == (0, 1)
        assert dataset.failure_seen.wait(timeout=2)
        with pytest.raises(RuntimeError, match="synthetic producer failure at 2"):
            batches.close()
    finally:
        batches.close()
    assert dataset.active == 0
    assert all(not worker.is_alive() for worker in batches._executor._threads)


def test_prefetch_config_is_opt_in_and_accepts_only_one_batch(tmp_path: Path) -> None:
    """Omission keeps the legacy config shape; explicit prefetch is bounded to one."""
    assert "input_prefetch_batches" not in DEFAULTS
    legacy = _write_recipe(tmp_path / "legacy.json", {"epochs": 2})
    assert "input_prefetch_batches" not in legacy
    enabled = _write_recipe(tmp_path / "prefetch.json", {"epochs": 2, "input_prefetch_batches": 1})
    assert enabled["input_prefetch_batches"] == 1
    disabled = _write_recipe(tmp_path / "disabled.json", {"input_prefetch_batches": 0})
    assert disabled["input_prefetch_batches"] == 0
    for invalid in (True, 2, -1, 1.0, None):
        with pytest.raises(ValueError, match="input_prefetch_batches"):
            _write_recipe(tmp_path / "invalid.json", {"input_prefetch_batches": invalid})
