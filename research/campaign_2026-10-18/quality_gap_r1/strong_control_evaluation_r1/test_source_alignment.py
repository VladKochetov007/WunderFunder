"""Tiny synthetic regressions for same-row source extraction safety checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from source_alignment import (
    collect_evaluation_rows,
    extract_group,
    prediction_row_map,
    require_sha256,
    write_npy_create_only,
)


class SourceAlignmentTests(unittest.TestCase):
    """Exercise masks, source order, finite values, hashes, and create-only outputs."""

    def _fixture(self) -> tuple[np.ndarray, ...]:
        seq = np.full(5, 17, dtype=np.int64)
        steps = np.arange(5, dtype=np.int64)
        need = np.array([False, True, True, True, True], dtype=bool)
        scored = np.array([False, True, True, False, True], dtype=bool)
        target0 = np.array([0, 1, 2, 3, 4], dtype=np.float32)
        target1 = target0 + np.float32(0.5)
        prediction = np.column_stack((target0, target1)).astype(np.float32)
        return seq, steps, need, scored, target0, target1, prediction

    def test_explicit_sidecar_map_handles_and_rejects_row_order(self) -> None:
        groups = [2, 0, 3, 1]
        mapping = prediction_row_map(groups, expected_group_count=4)
        self.assertEqual(mapping, {2: 0, 0: 1, 3: 2, 1: 3})
        with self.assertRaisesRegex(ValueError, "unique full-valid permutation"):
            prediction_row_map([2, 0, 0, 1], expected_group_count=4)
        with self.assertRaisesRegex(ValueError, "full validation source"):
            prediction_row_map([2, 0, 3, 4], expected_group_count=4)

    def test_scoring_mask_selects_only_need_and_scored_rows(self) -> None:
        seq, steps, need, scored, y0, y1, pred = self._fixture()
        labels, values, positions, mask = extract_group(
            seq, steps, need, scored, y0, y1, pred, 17, warmup=1
        )
        np.testing.assert_array_equal(positions, np.array([1, 2, 4], dtype=np.int64))
        np.testing.assert_array_equal(labels, np.column_stack((y0, y1))[[1, 2, 4]])
        np.testing.assert_array_equal(values, pred[[1, 2, 4]])
        self.assertEqual(int(mask.sum()), 3)

    def test_wrong_step_order_fails_closed(self) -> None:
        seq, steps, need, scored, y0, y1, pred = self._fixture()
        steps[[1, 2]] = steps[[2, 1]]
        with self.assertRaisesRegex(ValueError, "step order"):
            extract_group(seq, steps, need, scored, y0, y1, pred, 17, warmup=1)

    def test_selected_nonfinite_prediction_fails(self) -> None:
        seq, steps, need, scored, y0, y1, pred = self._fixture()
        pred[2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            extract_group(seq, steps, need, scored, y0, y1, pred, 17, warmup=1)

    def test_arrival_hash_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "differs from the arrival pin"):
            require_sha256("a" * 64, "b" * 64, "test artifact")

    def test_output_collision_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.npy"
            path.write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                write_npy_create_only(path, np.zeros((2, 2), dtype=np.float32))
            self.assertEqual(path.read_bytes(), b"keep")

    def test_tiny_parquet_uses_explicit_prediction_row_map_and_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.parquet"
            steps = np.tile(np.arange(5, dtype=np.int64), 2)
            need = np.tile(np.array([False, True, True, True, True]), 2)
            scored = np.tile(np.array([False, True, True, False, True]), 2)
            target = np.arange(10, dtype=np.float32)
            table = pa.table(
                {
                    "seq_ix": np.repeat(np.array([101, 202], dtype=np.int64), 5),
                    "step_in_seq": steps,
                    "need_prediction": need,
                    "is_scored": scored,
                    "t0": target,
                    "t1": target + np.float32(0.25),
                }
            )
            pq.write_table(table, path, row_group_size=5)
            predictions = np.zeros((2, 5, 2), dtype=np.float32)
            predictions[0] = np.float32(200.0)
            predictions[1] = np.float32(100.0)
            row_map = prediction_row_map([1, 0], expected_group_count=2)
            arrays, audit = collect_evaluation_rows(
                pq.ParquetFile(path),
                [0, 1],
                {0: {"seq_ix": 101, "component": 11}, 1: {"seq_ix": 202, "component": 12}},
                predictions,
                row_map,
                rows_per_group=5,
                warmup=1,
            )
            np.testing.assert_array_equal(arrays["row_group"], [0, 0, 0, 1, 1, 1])
            np.testing.assert_array_equal(arrays["component"], [11, 11, 11, 12, 12, 12])
            np.testing.assert_array_equal(
                arrays["prediction"][:, 0], [100, 100, 100, 200, 200, 200]
            )
            self.assertEqual([item["prediction_row_index"] for item in audit], [1, 0])


if __name__ == "__main__":
    unittest.main()
