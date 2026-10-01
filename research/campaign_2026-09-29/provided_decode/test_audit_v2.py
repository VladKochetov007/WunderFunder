from __future__ import annotations

import unittest

import numpy as np
from audit import price_keys
from audit_v2 import predictions, quote_coverage


class QuoteClockChecks(unittest.TestCase):
    """Check the corrected quote clock independently of old decoder code."""

    def test_zero_quote_moves_do_not_advance_quote_horizon(self) -> None:
        prices = np.repeat(np.arange(20, dtype=np.float32), 22).reshape(20, 22)
        keys = price_keys(prices)
        moves = {(keys[j - 1], keys[j]): (int(j % 2 == 0), int(j % 2 == 0)) for j in range(1, 20)}
        spreads = dict.fromkeys(keys, 1)
        rules, event, coverage = predictions(prices, moves, spreads)
        np.testing.assert_array_equal(rules["quote4/common_mean"][0][0], [2, -3])
        self.assertEqual(coverage["known_decoded_quote_changes"], 9)
        self.assertFalse(event[1])
        self.assertTrue(event[2])

    def test_unknown_zero_candidate_also_invalidates_quote_window(self) -> None:
        prices = np.repeat(np.arange(20, dtype=np.float32), 22).reshape(20, 22)
        keys = price_keys(prices)
        moves = {(keys[j - 1], keys[j]): (int(j % 2 == 0), int(j % 2 == 0)) for j in range(1, 20)}
        spreads = dict.fromkeys(keys, 1)
        self.assertTrue(quote_coverage(prices, moves, spreads, 4)[0])
        moves.pop((keys[0], keys[1]))
        self.assertFalse(quote_coverage(prices, moves, spreads, 4)[0])

    def test_flat_row_clock_includes_quiet_future_rows(self) -> None:
        prices = np.zeros((120, 22), dtype=np.float32)
        spreads = {price_keys(prices[:1])[0]: 1}
        rules, _, _ = predictions(prices, {}, spreads)
        self.assertTrue(rules["row50/common_mean"][1][0])
        np.testing.assert_array_equal(rules["row50/common_mean"][0][0], [0, -1])
        self.assertFalse(rules["quote4/common_mean"][1][0])


if __name__ == "__main__":
    unittest.main()
