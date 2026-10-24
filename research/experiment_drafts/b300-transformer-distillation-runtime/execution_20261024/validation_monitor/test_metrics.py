"""Numerical checks for streaming validation metrics against canonical formulas."""

from __future__ import annotations

import unittest

import numpy as np
from metrics import WMSE_METRIC_ID, StreamingMetrics

from wnn.gmetric import METRIC_ID, global_wp


def reference_wmse(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    """Evaluate raw-prediction weighted MSE in float64 for an independent oracle."""
    y = np.clip(target[mask].astype(np.float64), -2.0, 2.0)
    p = prediction[mask].astype(np.float64)
    weight = np.abs(y)
    total_weight = float(weight.sum())
    value = 0.0 if total_weight == 0 else float((weight * (p - y) ** 2).sum() / total_weight)
    return value, total_weight


class StreamingMetricsTests(unittest.TestCase):
    """Test exact chunk aggregation, masks, clipping, and failure behavior."""

    def setUp(self) -> None:
        rng = np.random.default_rng(20261024)
        self.y = rng.normal(size=(3, 140, 2)).astype(np.float32)
        self.p = (rng.normal(size=self.y.shape) * 2.5).astype(np.float32)
        self.scored = rng.random(self.y.shape[:2]) > 0.2
        self.scored[:, :99] = False
        self.after_warmup = np.ones(self.y.shape[:2], dtype=bool)
        self.after_warmup[:, :99] = False

    def test_streaming_matches_canonical_wp_and_raw_prediction_wmse(self) -> None:
        metrics = StreamingMetrics()
        metrics.update(self.y, self.p, self.scored, wmse_mask=self.after_warmup)
        result = metrics.result()

        expected_wp = global_wp(self.y, self.p, self.scored)
        expected_wmse, expected_weight = reference_wmse(self.y, self.p, self.after_warmup)
        self.assertAlmostEqual(result["wp"], expected_wp["wp"], places=14)
        self.assertAlmostEqual(result["t0"], expected_wp["t0"], places=14)
        self.assertAlmostEqual(result["t1"], expected_wp["t1"], places=14)
        self.assertAlmostEqual(result["wmse"], expected_wmse, places=12)
        self.assertAlmostEqual(result["wmse_weight"], expected_weight, places=10)
        self.assertEqual(result["selected_rows"], int(self.scored.sum()))
        self.assertEqual(result["wmse_rows"], int(self.after_warmup.sum()))
        self.assertEqual(result["metric_id"], METRIC_ID)
        self.assertEqual(result["wmse_metric_id"], WMSE_METRIC_ID)

    def test_streaming_is_invariant_to_batch_and_row_partitioning(self) -> None:
        whole = StreamingMetrics()
        whole.update(self.y, self.p, self.scored, wmse_mask=self.after_warmup)

        chunks = StreamingMetrics()
        for start, stop in ((0, 43), (43, 101), (101, 140)):
            chunks.update(
                self.y[:, start:stop],
                self.p[:, start:stop],
                self.scored[:, start:stop],
                wmse_mask=self.after_warmup[:, start:stop],
            )
        expected, actual = whole.result(), chunks.result()
        for key in ("wp", "t0", "t1", "wmse", "wmse_weight"):
            self.assertAlmostEqual(actual[key], expected[key], places=12)
        self.assertEqual(actual["selected_rows"], expected["selected_rows"])
        self.assertEqual(actual["wmse_rows"], expected["wmse_rows"])

    def test_default_wmse_uses_scored_mask(self) -> None:
        metrics = StreamingMetrics()
        metrics.update(self.y, self.p, self.scored)
        result = metrics.result()
        expected, weight = reference_wmse(self.y, self.p, self.scored)
        self.assertAlmostEqual(result["wmse"], expected, places=12)
        self.assertAlmostEqual(result["wmse_weight"], weight, places=10)
        self.assertEqual(result["wmse_rows"], int(self.scored.sum()))

    def test_raw_prediction_is_not_clipped_for_wmse(self) -> None:
        target = np.ones((1, 3, 2), dtype=np.float32)
        prediction = np.full_like(target, 3.0)
        mask = np.ones((1, 3), dtype=bool)
        result = StreamingMetrics()
        result.update(target, prediction, mask)
        self.assertEqual(result.result()["wmse"], 4.0)

    def test_unselected_nonfinite_values_are_ignored_but_selected_ones_fail(self) -> None:
        target = np.ones((1, 4, 2), dtype=np.float32)
        prediction = np.ones_like(target)
        mask = np.array([[True, False, True, False]])
        prediction[0, 1] = np.nan
        metrics = StreamingMetrics()
        metrics.update(target, prediction, mask)
        self.assertEqual(metrics.result()["wmse"], 0.0)

        prediction[0, 2] = np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            StreamingMetrics().update(target, prediction, mask)

    def test_invalid_shapes_and_empty_official_mask_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "aligned"):
            StreamingMetrics().update(self.y, self.p, np.ones((2, 140), dtype=bool))
        empty = StreamingMetrics()
        empty.update(self.y, self.p, np.zeros_like(self.scored))
        with self.assertRaisesRegex(ValueError, "no selected rows"):
            empty.result()

    def test_zero_target_weights_produce_zero_wmse(self) -> None:
        target = np.zeros((2, 5, 2), dtype=np.float32)
        prediction = np.ones_like(target)
        mask = np.ones((2, 5), dtype=bool)
        metrics = StreamingMetrics()
        metrics.update(target, prediction, mask)
        result = metrics.result()
        self.assertEqual(result["wmse"], 0.0)
        self.assertEqual(result["wmse_weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
