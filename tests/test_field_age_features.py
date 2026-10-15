"""Independent checks for causal per-field and broadcast group-age features."""

from __future__ import annotations

import numpy as np

from wnn.research_features import (
    FIELD_AGE_CAP,
    RAW_FIELD_GROUPS,
    StreamingFieldAgeFeatures,
    StreamingResearchFeatures,
    research_features_numpy,
)


def _oracle(raw: np.ndarray, *, grouped: bool) -> np.ndarray:
    """Small, deliberately row-wise specification independent of production code."""
    result = np.zeros(raw.shape, dtype=np.float32)
    ages = np.zeros(112, dtype=np.int64)
    previous: np.ndarray | None = None
    for row_index, row in enumerate(raw):
        if previous is not None:
            if grouped:
                for start, end in RAW_FIELD_GROUPS:
                    if np.any(row[start:end] != previous[start:end]):
                        ages[start:end] = 0
                    else:
                        ages[start:end] = np.minimum(ages[start:end] + 1, FIELD_AGE_CAP)
            else:
                for column in range(112):
                    if row[column] != previous[column]:
                        ages[column] = 0
                    else:
                        ages[column] = min(ages[column] + 1, FIELD_AGE_CAP)
        result[row_index] = np.log1p(ages.astype(np.float32))
        previous = row.copy()
    return result


def _raw_panel(batch: int = 2, rows: int = 257) -> np.ndarray:
    row = np.arange(rows, dtype=np.float32)[:, None]
    col = np.arange(112, dtype=np.float32)[None, :]
    values = (col + 1.0) * np.ones((rows, 1), dtype=np.float32)
    values = np.broadcast_to(values, (rows, 112)).copy()
    # Different reset-local schedules exercise every fixed field group and
    # ensure the field-level representation is not accidentally group-broadcast.
    for field in range(112):
        period = 3 + field % 17
        values[:, field] += (row[:, 0] // period).astype(np.float32)
    values[1::29, 0] += 0.25
    values[7::31, 23] += 0.5
    values[4::37, 55] += 0.75
    values[2::41, 111] += 1.0
    return np.stack([values, values[::-1].copy()], axis=0)[:batch]


def test_vectorized_age_features_match_independent_row_oracle() -> None:
    raw = _raw_panel()
    for mode, grouped in (("field_age", False), ("group_age", True)):
        actual = research_features_numpy(raw, mode)
        expected = np.stack([_oracle(sequence, grouped=grouped) for sequence in raw])
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == raw.shape
        assert np.isfinite(actual).all()
        assert np.count_nonzero(actual[:, 0]) == 0
        if grouped:
            for start, end in RAW_FIELD_GROUPS:
                np.testing.assert_array_equal(
                    actual[..., start:end],
                    np.broadcast_to(actual[..., start : start + 1], actual[..., start:end].shape),
                )
    field_ages = research_features_numpy(raw, "field_age")
    group_ages = research_features_numpy(raw, "group_age")
    assert np.any(field_ages != group_ages)


def test_streaming_age_features_match_oracle_across_chunks_and_resets() -> None:
    raw = _raw_panel(rows=257)
    cuts = (1, 2, 99, 100, 128, 256, len(raw[0]))
    for mode, grouped in (("field_age", False), ("group_age", True)):
        for sequence in raw:
            stream = StreamingFieldAgeFeatures(mode)
            parts = []
            start = 0
            for stop in cuts:
                parts.append(stream.update_chunk(sequence[start:stop]))
                start = stop
            actual = np.concatenate(parts, axis=0)
            np.testing.assert_array_equal(actual, _oracle(sequence, grouped=grouped))
            integrated = StreamingResearchFeatures(mode)
            integrated_rows = np.stack([integrated.update(row) for row in sequence])
            np.testing.assert_array_equal(integrated_rows, _oracle(sequence, grouped=grouped))
            stream.reset()
            reset_replay = stream.update_chunk(sequence[:2])
            np.testing.assert_array_equal(reset_replay, _oracle(sequence[:2], grouped=grouped))


def test_future_raw_changes_do_not_rewrite_past_ages() -> None:
    raw = _raw_panel(rows=257)
    changed_future = raw.copy()
    changed_future[:, 150:, 17] += 100.0
    for mode in ("group_age", "field_age"):
        original = research_features_numpy(raw, mode)
        changed = research_features_numpy(changed_future, mode)
        np.testing.assert_array_equal(original[:, :150], changed[:, :150])


def test_age_caps_at_declared_history_limit() -> None:
    raw = np.zeros((FIELD_AGE_CAP + 2, 112), dtype=np.float32)
    actual = research_features_numpy(raw, "field_age")
    capped_ages = np.minimum(np.arange(len(raw)), FIELD_AGE_CAP).astype(np.float32)
    expected = np.log1p(capped_ages)
    np.testing.assert_array_equal(actual[:, 0], expected)
    np.testing.assert_array_equal(actual[:, -1], expected)
