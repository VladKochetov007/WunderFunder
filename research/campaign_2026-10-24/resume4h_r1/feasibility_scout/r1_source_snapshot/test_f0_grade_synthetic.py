"""Small synthetic checks for F0-grade streaming and fixed contrast structure."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from f0_grade_scoring import (
    _extract_npy_member,
    build_contrasts,
    stream_global_score,
)

from wnn.gmetric import GlobalAccumulator

SCRATCH = Path("/home/vlad/development/wunder/research_tmp")


class SyntheticGradeTests(unittest.TestCase):
    """Exercise bounded helpers on synthetic arrays without campaign labels."""

    def test_fixed_contrasts_preserve_requested_primary_pairs(self) -> None:
        contrasts = build_contrasts()
        primary = [contrast for contrast in contrasts if contrast["role"] == "primary"]
        self.assertEqual(len(contrasts), 15)
        self.assertEqual(len(primary), 4)
        self.assertEqual(
            {(item["candidate"], item["reference"]) for item in primary},
            {
                ("C_lag1", "B_lag1"),
                ("B_lag1", "A_lag1"),
                ("C_lag16", "B_lag16"),
                ("B_lag16", "A_lag16"),
            },
        )
        self.assertEqual(len({item["name"] for item in contrasts}), len(contrasts))

    def test_sequence_stream_matches_single_full_accumulator(self) -> None:
        generator = np.random.default_rng(17)
        targets = generator.normal(size=(3, 19, 2)).astype(np.float32)
        predictions = generator.normal(size=targets.shape).astype(np.float32)
        mask = generator.random(size=targets.shape[:2]) > 0.2
        expected = GlobalAccumulator()
        expected.add(targets, predictions, mask)
        actual = stream_global_score(targets, predictions, mask)
        for name in ("wp", "t0", "t1"):
            self.assertAlmostEqual(actual[name], expected.result()[name], places=12)
        self.assertEqual(actual["selected_rows"], expected.result()["selected_rows"])

    def test_npy_member_copy_is_bounded_and_exact(self) -> None:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="resume4h-synthetic-", dir=SCRATCH) as temp_name:
            root = Path(temp_name)
            targets = np.arange(20, dtype=np.float32).reshape(2, 5, 2)
            masks = np.array([[True, False, True, True, False], [False, True, True, False, True]])
            archive_path = root / "synthetic.npz"
            np.savez(archive_path, row_group=np.array([7, 11]), y=targets, mask=masks)
            output_path = root / "synthetic-y.npy"
            coverage = _extract_npy_member(archive_path, "y.npy", output_path)
            self.assertEqual(coverage["shape"], [2, 5, 2])
            self.assertEqual(coverage["dtype"], "<f4")
            np.testing.assert_array_equal(
                np.load(output_path, mmap_mode="r", allow_pickle=False), targets
            )


if __name__ == "__main__":
    unittest.main()
