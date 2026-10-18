"""Focused synthetic tests for the standalone native panel checker."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from check_native_panel import (
    SEQUENCE_LENGTH,
    DataPoint,
    OfficialWPAccumulator,
    callback_prediction,
    extract_zip,
    official_wp,
)


class _WarmupModel:
    def predict(self, point: DataPoint) -> np.ndarray | None:
        if not point.need_prediction:
            return None
        return point.state[:2].copy()


class _InvalidWarmupModel:
    def predict(self, point: DataPoint) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)


class NativePanelCheckerTests(unittest.TestCase):
    def test_streaming_wp_matches_direct_global_formula(self) -> None:
        rows = np.arange(SEQUENCE_LENGTH, dtype=np.float32)
        base = np.stack((np.sin(rows / 90), np.cos(rows / 73)), axis=-1).astype(np.float32)
        targets = np.stack((base, base * 0.7 + np.array([0.3, -0.2], dtype=np.float32)))
        predictions = np.stack((base * 0.8 + 0.1, base * -0.2 + 0.3))
        mask = np.broadcast_to(rows[None, :] >= 99, (2, SEQUENCE_LENGTH)).copy()
        mask[:, ::7] = False
        expected = official_wp(targets, predictions, mask)
        accumulator = OfficialWPAccumulator()
        for index in range(2):
            accumulator.add(targets[index], predictions[index], mask[index])
        actual = accumulator.result()
        self.assertEqual(expected["selected_rows"], actual["selected_rows"])
        self.assertAlmostEqual(float(expected["t0"]), float(actual["t0"]), places=12)
        self.assertAlmostEqual(float(expected["t1"]), float(actual["t1"]), places=12)
        self.assertAlmostEqual(float(expected["wp"]), float(actual["wp"]), places=12)

    def test_callback_requires_none_during_warmup_and_finite_pair_after(self) -> None:
        features = np.array([1.0, -2.0], dtype=np.float32)
        self.assertIsNone(callback_prediction(_WarmupModel(), 3, 0, False, features))
        np.testing.assert_array_equal(
            callback_prediction(_WarmupModel(), 3, 99, True, features), features
        )
        with self.assertRaisesRegex(ValueError, "warm-up"):
            callback_prediction(_InvalidWarmupModel(), 3, 0, False, features)

    def test_zip_extraction_accepts_root_solution_and_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "safe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("solution.py", "class PredictionModel: pass\n")
                archive.writestr("v2/wnn_native.abi3.so", b"fixture")
            destination = root / "safe"
            destination.mkdir()
            extract_zip(archive_path, destination)
            self.assertTrue((destination / "solution.py").is_file())
            self.assertTrue((destination / "v2/wnn_native.abi3.so").is_file())

            unsafe_path = root / "unsafe.zip"
            with zipfile.ZipFile(unsafe_path, "w") as archive:
                archive.writestr("solution.py", "pass\n")
                archive.writestr("../escape", b"no")
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                extract_zip(unsafe_path, root / "unsafe")


if __name__ == "__main__":
    unittest.main()
