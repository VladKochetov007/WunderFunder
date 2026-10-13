from __future__ import annotations

import unittest

import numpy as np

from scripts.benchmark_team_runtimes import abba_order, parity_summary


class AbbaScheduleTest(unittest.TestCase):
    def test_repeated_order_is_a_b_b_a(self) -> None:
        schedule = abba_order(2)
        self.assertEqual(
            [(cycle, runtime) for cycle, _, runtime in schedule],
            [
                (0, "cpp"),
                (0, "rust"),
                (0, "rust"),
                (0, "cpp"),
                (1, "cpp"),
                (1, "rust"),
                (1, "rust"),
                (1, "cpp"),
            ],
        )
        self.assertEqual([phase for _, phase, _ in schedule], [0, 1, 2, 3] * 2)

    def test_schedule_rejects_unbounded_cycle_counts(self) -> None:
        for cycles in (0, 11):
            with self.subTest(cycles=cycles), self.assertRaises(ValueError):
                abba_order(cycles)


class PairedOutputSummaryTest(unittest.TestCase):
    def test_bitwise_repeat_and_numeric_difference_are_both_reported(self) -> None:
        first = np.array([[0.0, 1.0], [-2.0, 0.5]], dtype=np.float32)
        repeat = first.copy()
        difference = first.copy()
        difference[1, 1] = 0.75
        self.assertTrue(parity_summary(first, repeat)["bitwise_equal"])
        summary = parity_summary(first, difference)
        self.assertFalse(summary["bitwise_equal"])
        self.assertEqual(summary["different_elements"], 1)
        self.assertEqual(summary["max_abs"], 0.25)

    def test_shape_mismatch_and_nonfinite_outputs_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "different shapes"):
            parity_summary(np.zeros((1, 2)), np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            parity_summary(np.array([np.inf]), np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
