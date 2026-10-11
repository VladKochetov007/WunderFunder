"""Unit checks for the Q4-grid transformation boundaries."""

from __future__ import annotations

import unittest

import numpy as np
from build_q4_variant import _is_target_tensor, rowwise_q4_round


class RowwiseQ4RoundingTests(unittest.TestCase):
    """Verify row scaling, signed levels, zero rows, and selection boundaries."""

    def test_nonzero_rows_use_per_row_maximum_and_signed_levels(self) -> None:
        matrix = np.array([[0.0, 1.0, -0.5, 0.25], [2.0, -1.0, 0.0, 0.5]], dtype=np.float32)
        reconstructed, codes, scales = rowwise_q4_round(matrix)

        np.testing.assert_array_equal(codes[:, 0], np.array([0, 7], dtype=np.int8))
        self.assertEqual(float(scales[0]), float(np.float32(1.0 / 7.0)))
        self.assertEqual(float(scales[1]), float(np.float32(2.0 / 7.0)))
        self.assertEqual(int(codes.min()), -3)
        self.assertEqual(int(codes.max()), 7)
        self.assertLessEqual(float(np.max(np.abs(reconstructed))), 2.0)
        self.assertTrue(np.isfinite(reconstructed).all())

    def test_zero_rows_remain_exact_zero(self) -> None:
        reconstructed, codes, scales = rowwise_q4_round(np.zeros((2, 3), dtype=np.float32))

        np.testing.assert_array_equal(reconstructed, np.zeros((2, 3), dtype=np.float32))
        np.testing.assert_array_equal(codes, np.zeros((2, 3), dtype=np.int8))
        np.testing.assert_array_equal(scales, np.zeros(2, dtype=np.float32))

    def test_ties_follow_numpy_round_to_even(self) -> None:
        matrix = np.array([[7.0, 2.5]], dtype=np.float32)

        _, codes, _ = rowwise_q4_round(matrix)

        self.assertEqual(int(codes[0, 1]), 2)

    def test_only_dense_recurrent_and_first_head_matrices_are_selected(self) -> None:
        matrix = np.zeros((2, 2), dtype=np.float32)

        self.assertTrue(_is_target_tensor("gru.weight_ih_l0", matrix))
        self.assertTrue(_is_target_tensor("gru.weight_hh_l1", matrix))
        self.assertTrue(_is_target_tensor("head.0.weight", matrix))
        self.assertFalse(_is_target_tensor("inp.weight", matrix))
        self.assertFalse(_is_target_tensor("head.2.weight", matrix))
        self.assertFalse(_is_target_tensor("gru.bias_ih_l0", matrix))

    def test_invalid_dtype_or_nonfinite_values_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            rowwise_q4_round(np.ones((2, 2), dtype=np.float64))
        with self.assertRaises(ValueError):
            rowwise_q4_round(np.array([[np.nan]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
