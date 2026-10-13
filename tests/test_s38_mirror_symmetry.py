"""Correctness checks for the frozen S38 signed-coordinate maps."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.s38_mirror_symmetry_diagnostic import (
    analyze_sequence,
    build_maps,
    transform,
    validate_group_membership,
    validate_involution,
)


class MirrorMapTests(unittest.TestCase):
    """Check exact schema coordinates and involution before reading data."""

    def test_all_maps_are_involutions(self) -> None:
        for mapping in build_maps().values():
            validate_involution(*mapping)

    def test_side_swap_changes_only_documented_bid_ask_blocks(self) -> None:
        values = np.arange(112, dtype=np.float64)[None, :]
        transformed = transform(values, build_maps()["side_swap"])
        self.assertEqual(transformed[0, 0], values[0, 11])
        self.assertEqual(transformed[0, 11], values[0, 0])
        self.assertEqual(transformed[0, 22], values[0, 33])
        self.assertEqual(transformed[0, 52], values[0, 63])
        self.assertTrue(np.array_equal(transformed[0, 44:52], values[0, 44:52]))
        self.assertTrue(np.array_equal(transformed[0, 96:], values[0, 96:]))

    def test_reflection_and_negative_control_signs_are_frozen(self) -> None:
        values = np.ones((1, 112), dtype=np.float64)
        mappings = build_maps()
        reflected = transform(values, mappings["price_axis_reflection"])[0]
        sign_only = transform(values, mappings["price_sign_only_negative_control"])[0]
        self.assertTrue(np.all(reflected[0:22] == -1))
        self.assertTrue(np.all(reflected[22:44] == 1))
        self.assertTrue(np.all(reflected[44:48] == -1))
        self.assertTrue(np.all(reflected[48:52] == 1))
        self.assertTrue(np.all(sign_only[0:22] == -1))
        self.assertTrue(np.all(sign_only[11:22] == -1))
        self.assertEqual(sign_only[0], -1)
        self.assertTrue(np.all(sign_only[104:] == 1))

    def test_every_map_squares_to_identity_on_sample_rows(self) -> None:
        values = np.arange(3 * 112, dtype=np.float64).reshape(3, 112)
        for mapping in build_maps().values():
            self.assertTrue(np.array_equal(transform(transform(values, mapping), mapping), values))

    def test_identity_has_zero_support_and_quantile_displacement(self) -> None:
        rng = np.random.default_rng(317)
        values = rng.normal(size=(20_000, 112)).astype(np.float32)
        identity = (np.arange(112, dtype=np.int64), np.ones(112, dtype=np.float64))
        metrics = analyze_sequence(values, identity)
        self.assertAlmostEqual(
            metrics["outside_empirical_0_1_99_9_percent_fraction"], 0.002, places=4
        )
        self.assertEqual(metrics["outside_observed_min_max_fraction"], 0.0)
        self.assertEqual(metrics["max_feature_displacement"], 0.0)

    def test_group_membership_allows_known_manifest_exclusions(self) -> None:
        split = {
            "sequences": {"train": [{"row_group": group} for group in (0, 1, 3)]},
            "fit": {"train": [0, 3]},
        }
        validate_group_membership(split, row_group_count=4)
        split["fit"]["train"] = [0, 2]
        with self.assertRaisesRegex(ValueError, "absent from train records"):
            validate_group_membership(split, row_group_count=4)


if __name__ == "__main__":
    unittest.main()
