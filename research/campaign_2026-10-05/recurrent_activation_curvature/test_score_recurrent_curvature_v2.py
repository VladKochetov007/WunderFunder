"""Small synthetic tests for the E481 v2 grader repair; never read score labels."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import score_recurrent_curvature_v2 as scorer


class CurvatureScorerV2Tests(unittest.TestCase):
    def test_local_file_sha256_matches_standard_library(self) -> None:
        payload = b"E481 actual fit-only proof hash fixture\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.bin"
            path.write_bytes(payload)
            self.assertEqual(scorer.file_sha256(path), hashlib.sha256(payload).hexdigest())

    def test_uses_local_hash_api_not_missing_e470_helper(self) -> None:
        self.assertFalse(hasattr(scorer.e470, "file_hash"))
        self.assertTrue(callable(scorer.file_sha256))

    def test_candidate_transforms_are_finite_and_zero_anchored(self) -> None:
        values = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        tensor = scorer.torch.from_numpy(values)
        for shape_code in range(3):
            actual = scorer.transform_candidate(tensor, shape_code).numpy()
            self.assertTrue(np.isfinite(actual).all())
            self.assertAlmostEqual(float(actual[2]), 0.0, places=7)
            self.assertAlmostEqual(float(actual[0]), -1.0, places=6)
            self.assertAlmostEqual(float(actual[-1]), 1.0, places=6)

    def test_registered_contrasts_have_all_ten_paired_effects(self) -> None:
        names = [
            f"{optimizer}__{shape}" for optimizer in scorer.OPTIMIZERS for shape in scorer.SHAPES
        ]
        scores = {name: float(index) / 10.0 for index, name in enumerate(names)}
        draws = {name: np.full(scorer.BOOTSTRAP_COUNT, score) for name, score in scores.items()}
        result = scorer.registered_contrasts(scores, draws)
        self.assertEqual(len(result), 10)
        self.assertAlmostEqual(result["adamw__sine_minus_adamw__baseline"]["mean_diff"], 0.1)
        self.assertAlmostEqual(
            result["curvature_cubic_minus_sine_x_muon_minus_adamw"]["mean_diff"], 0.0
        )
        self.assertTrue(
            all(item["ci95"] == [item["mean_diff"], item["mean_diff"]] for item in result.values())
        )


if __name__ == "__main__":
    unittest.main()
