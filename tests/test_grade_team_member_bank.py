from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.grade_team_member_bank import GlobalAccumulator
from scripts.team_member_grade_support import (
    CovarianceAccumulator,
    _load_prediction_map,
    _plan_records,
    blend_predictions,
    read_bounded_bytes,
)


class FixedMixtureTest(unittest.TestCase):
    def test_blends_raw_fp32_predictions_before_official_clipping(self) -> None:
        target_side = np.array([-2, -1, 1, 2, -0.5, 0.5, 1.5, -1.5], dtype=np.float32)
        targets = np.column_stack((target_side, target_side))
        first_side = np.array([4, 4, 0, 0, -4, -4, 2, -2], dtype=np.float32)
        second_side = np.array([0, 4, 0, -4, 0, -4, -2, 2], dtype=np.float32)
        first = np.column_stack((first_side, first_side))
        second = np.column_stack((second_side, second_side))
        weights = {"first": 0.5, "second": 0.5}

        mixture = blend_predictions({"first": first, "second": second}, weights)
        np.testing.assert_array_equal(
            mixture[:, 0], np.array([2, 4, 0, -2, -2, -4, 0, 0], dtype=np.float32)
        )
        mask = np.ones(targets.shape[0], dtype=np.bool_)
        official = GlobalAccumulator()
        official.add(targets, mixture, mask)

        clipped_members = {
            "first": np.clip(first, -2, 2).astype(np.float32),
            "second": np.clip(second, -2, 2).astype(np.float32),
        }
        wrong_order = blend_predictions(clipped_members, weights)
        wrong = GlobalAccumulator()
        wrong.add(targets, wrong_order, mask)
        official_wp = float(official.result()["wp"])
        wrong_wp = float(wrong.result()["wp"])
        self.assertAlmostEqual(
            official_wp,
            -0.7348469228349533,
            places=12,
        )
        self.assertNotAlmostEqual(official_wp, wrong_wp, places=4)

    def test_plan_requires_unique_members_and_normalized_frozen_weights(self) -> None:
        members = [{"id": f"m{index}", "sha256": f"{index + 1:064x}"} for index in range(14)]
        mixtures = {f"mix{index}": {"m0": 0.25, "m1": 0.75} for index in range(9)}
        parsed_members, parsed_mixtures = _plan_records({"members": members, "mixtures": mixtures})
        self.assertEqual(len(parsed_members), 14)
        self.assertEqual(len(parsed_mixtures), 9)
        original_mix0 = mixtures.pop("mix0")
        mixtures["m0"] = {"m0": 1.0}
        with self.assertRaisesRegex(ValueError, "distinct from member IDs"):
            _plan_records({"members": members, "mixtures": mixtures})
        mixtures.pop("m0")
        mixtures["mix0"] = original_mix0
        mixtures["mix0"] = {"m0": 0.25, "m1": 0.5}
        with self.assertRaisesRegex(ValueError, "sum to one"):
            _plan_records({"members": members, "mixtures": mixtures})


class StreamingArtifactsTest(unittest.TestCase):
    def test_bounded_metadata_reader_rejects_large_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.json"
            path.write_bytes(b"12345")
            with self.assertRaisesRegex(ValueError, "byte limit"):
                read_bounded_bytes(path, max_bytes=4)

    def test_incremental_npy_digest_covers_header_and_every_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.npy"
            predictions = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(2, 3, 2)
            )
            predictions[:] = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
            predictions.flush()
            del predictions

            mapped, hasher = _load_prediction_map(path, (2, 3, 2))
            try:
                for group in range(2):
                    hasher.update(memoryview(mapped[group]).cast("B"))
                with path.open("rb") as stream:
                    expected = hashlib.file_digest(stream, "sha256").hexdigest()
                self.assertEqual(hasher.hexdigest(), expected)
            finally:
                mapped._mmap.close()  # pyright: ignore[reportAttributeAccessIssue]

    def test_centered_correlation_merges_blocks(self) -> None:
        values = np.array([[1.0, 2.0, -1.0], [2.0, 1.0, -2.0], [3.0, 4.0, 0.0], [5.0, 3.0, 1.0]])
        accumulator = CovarianceAccumulator(3)
        accumulator.add(values[:2])
        accumulator.add(values[2:])
        np.testing.assert_allclose(
            accumulator.correlation(), np.corrcoef(values, rowvar=False), atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
