"""Pure numerical checks; no model, inference, fitting, or optimizer calls."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from derive_sequence_decomposition import decompose  # noqa: E402

from wnn.gmetric import global_wp, wp_sums  # noqa: E402


def test_decomposition_matches_public_centered_scorer() -> None:
    """Check the weighted decomposition on 100 independent masked fixtures."""
    rng = np.random.default_rng(20261017)
    for _ in range(100):
        target = np.clip(rng.normal(size=(9, 37)), -2, 2).astype(np.float32)
        prediction = np.clip(0.6 * target + rng.normal(size=target.shape), -2, 2).astype(np.float32)
        mask = rng.uniform(size=target.shape) > 0.25
        sums = np.stack([wp_sums(target[j], prediction[j], mask[j]) for j in range(9)])
        result = decompose(sums)
        reference = global_wp(
            np.stack([target, target], axis=-1),
            np.stack([prediction, prediction], axis=-1),
            mask,
        )
        assert abs(result["pooled_wp_reconstructed"] - reference["wp"]) < 1e-10
        parts = (
            result["within_covariance_over_global_denominator"]
            + result["between_covariance_over_global_denominator"]
        )
        assert abs(parts - reference["wp"]) < 1e-10
