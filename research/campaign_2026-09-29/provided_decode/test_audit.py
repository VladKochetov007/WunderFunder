from __future__ import annotations

import unittest

import numpy as np
from audit import common_law, event_rows, horizon_predictions, invert_pair, price_keys


class IndependentLawChecks(unittest.TestCase):
    """Check payoff branches and uncertainty without old decoder code."""

    def test_law_and_inverse_are_algebraic(self) -> None:
        future = np.array([-2, 0, 1, 2, 4])
        expected = np.array([[-4, 2], [-2, 0], [-1, -1], [0, -2], [2, -4]])
        pairs = common_law(future, np.zeros(5), np.full(5, 2))
        np.testing.assert_array_equal(pairs, expected)
        spread, implied_future = invert_pair(expected)
        np.testing.assert_array_equal(spread, np.full(5, 2))
        np.testing.assert_array_equal(implied_future, future)

    def test_full_block_event_not_representative_column(self) -> None:
        prices = np.zeros((4, 22), dtype=np.float32)
        prices[2:, 7] = 1
        np.testing.assert_array_equal(event_rows(prices), [0, 2])

    def test_unknown_coordinates_invalidate_future_horizon(self) -> None:
        prices = np.repeat(np.arange(12, dtype=np.float32), 22).reshape(12, 22)
        keys = price_keys(prices)
        moves = {(keys[index - 1], keys[index]): (1, 1) for index in range(1, 12)}
        spreads = dict.fromkeys(keys, 1)
        rules, _ = horizon_predictions(prices, 3, moves, spreads)
        self.assertTrue(rules["event4/common_mean"][1][0])
        moves.pop((keys[1], keys[2]))
        rules, _ = horizon_predictions(prices, 3, moves, spreads)
        self.assertFalse(rules["event4/common_mean"][1][0])

    def test_barrier_and_common_reference_are_distinct(self) -> None:
        prices = np.repeat(np.arange(12, dtype=np.float32), 22).reshape(12, 22)
        keys = price_keys(prices)
        moves = {(keys[index - 1], keys[index]): (1, 1) for index in range(1, 12)}
        spreads = dict.fromkeys(keys, 1)
        rules, _ = horizon_predictions(prices, 3, moves, spreads)
        self.assertFalse(
            np.array_equal(
                rules["event4/common_mean"][0][0], rules["event4/independent_stop_end"][0][0]
            )
        )


if __name__ == "__main__":
    unittest.main()
