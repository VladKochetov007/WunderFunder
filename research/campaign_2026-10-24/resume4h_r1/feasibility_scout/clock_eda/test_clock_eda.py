"""Small no-data invariants for update-event overlap and censored ages."""

from __future__ import annotations

import unittest

import numpy as np
from clock_eda import feature_layout, finalize, summarize_sequence


class ClockEdaTests(unittest.TestCase):
    """Check exact live-notebook fields and sequence-local clock semantics."""

    def test_live_feature_order_has_112_fields_and_nine_families(self) -> None:
        names, group_names, groups = feature_layout()
        self.assertEqual(len(names), 112)
        self.assertEqual(
            group_names,
            ["i0_p", "i0_v", "i0_dp", "i0_dv", "i1_p", "i1_v", "i1_dp", "i1_dv", "a"],
        )
        self.assertEqual(
            [len(groups[name]) for name in group_names], [22, 22, 4, 4, 22, 22, 4, 4, 8]
        )
        self.assertEqual(names[:3], ["i0_p0", "i0_p1", "i0_p2"])
        self.assertEqual(names[-1], "a7")

    def test_change_events_use_exact_adjacent_rows_and_ignore_row_zero(self) -> None:
        names, _, groups = feature_layout()
        del names
        values = np.zeros((8, 112), dtype=np.float32)
        values[3:, 0] = 1
        values[6:, 0] = 2
        values[3:, 1] = 1
        values[5:, 2] = 1
        need = np.ones(8, dtype=bool)
        scored = np.ones(8, dtype=bool)
        need[5] = False
        stats = summarize_sequence(
            values,
            np.full(8, 42, dtype=np.int64),
            need,
            scored,
            groups,
            position_edges=(0, 2, 5, 8),
            age_thresholds=(1, 2, 4),
        )
        self.assertEqual(stats["eligible_rows"], 7)
        self.assertEqual(stats["selected_change_events"], 3)
        self.assertEqual(int(stats["cochange_counts"][0, 1]), 1)
        self.assertEqual(int(stats["cochange_counts"][0, 2]), 0)
        self.assertEqual(stats["change_counts_by_position"][:, 0].tolist(), [0, 1, 1])

    def test_position_age_keeps_start_censoring_until_first_update(self) -> None:
        _, _, groups = feature_layout()
        values = np.zeros((8, 112), dtype=np.float32)
        values[3:, 0] = 1
        values[6:, 0] = 2
        stats = summarize_sequence(
            values,
            np.full(8, 7, dtype=np.int64),
            np.ones(8, dtype=bool),
            np.ones(8, dtype=bool),
            groups,
            position_edges=(0, 2, 5, 8),
            age_thresholds=(1, 2, 4),
        )
        self.assertEqual(stats["censored_counts_by_position"][:, 0].tolist(), [2, 1, 0])
        self.assertEqual(stats["age_ge_counts_by_position"][:, 0, 0].tolist(), [1, 2, 2])

    def test_final_matrices_are_symmetric_and_plot_ready(self) -> None:
        names, group_names, groups = feature_layout()
        values = np.zeros((8, 112), dtype=np.float32)
        values[3:, 0] = 1
        values[6:, 0] = 2
        values[3:, 1] = 1
        stats = summarize_sequence(
            values,
            np.full(8, 3, dtype=np.int64),
            np.ones(8, dtype=bool),
            np.ones(8, dtype=bool),
            groups,
            position_edges=(0, 2, 5, 8),
            age_thresholds=(1, 2, 4),
        )
        totals = {
            "eligible_rows": stats["eligible_rows"],
            "eligible_rows_by_position": stats["eligible_rows_by_position"],
            "change_counts_by_position": stats["change_counts_by_position"],
            "censored_counts_by_position": stats["censored_counts_by_position"],
            "age_ge_counts_by_position": stats["age_ge_counts_by_position"],
            "cochange_counts": stats["cochange_counts"],
        }
        result = finalize(
            totals,
            [{key: stats[key] for key in ("seq_ix", "eligible_rows", "selected_change_events")}],
            names,
            group_names,
            groups,
            (0, 2, 5, 8),
            (1, 2, 4),
        )
        pairwise = result["pairwise_update_events"]
        matrix = np.asarray(pairwise["jaccard"])
        self.assertTrue(np.allclose(matrix, matrix.T))
        self.assertEqual(matrix[0, 1], 0.5)
        self.assertEqual(len(pairwise["group_pair_mean_jaccard"]), 45)
        position = result["position_activity_age"]
        self.assertEqual(np.asarray(position["age_ge_rates_by_feature"]).shape, (3, 112, 3))


if __name__ == "__main__":
    unittest.main()
