"""Synthetic invariants for the bounded F0 residual diagnostic."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from residual_analysis import analyze_arrays, write_csv  # noqa: E402

from wnn.gmetric import global_wp  # noqa: E402


class ResidualAnalysisTests(unittest.TestCase):
    """Check additive decomposition, paired deltas, and zero-weight rows."""

    def setUp(self) -> None:
        self.targets = np.array(
            [
                [0.0, 0.1],
                [0.0, -0.2],
                [0.0, 0.0],
                [0.05, 0.3],
                [-0.1, -0.5],
                [0.2, 0.8],
                [-0.4, -1.0],
                [0.7, 1.3],
                [-1.2, -1.8],
                [1.8, 2.1],
                [-2.2, 0.0],
                [0.3, 0.6],
            ],
            dtype=np.float32,
        )
        offsets = np.array(
            [
                [0.0, 0.2],
                [0.1, -0.1],
                [0.5, 0.0],
                [-0.1, 0.3],
                [0.2, -0.2],
                [-0.3, 0.1],
                [0.4, -0.4],
                [-0.1, 0.2],
                [0.3, -0.1],
                [0.9, 0.7],
                [-0.8, 0.2],
                [0.1, -0.3],
            ],
            dtype=np.float32,
        )
        self.wide = self.targets * np.float32(0.82) + offsets
        self.deep = self.targets * np.float32(0.91) - offsets * np.float32(0.35)
        self.q8 = self.targets * np.float32(0.88) + offsets * np.float32(0.6)
        self.q8[9, 0] = np.float32(2.7)
        self.q8[10, 0] = np.float32(-1.9)
        self.mix = np.add(
            np.multiply(self.wide, np.float32(0.75)),
            np.multiply(self.deep, np.float32(0.25)),
            dtype=np.float32,
        )
        self.predictions = {"wide": self.wide, "mix": self.mix, "q8": self.q8}
        self.row_group = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)
        self.component = np.array([10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40], dtype=np.int64)
        self.comparisons = [
            {"baseline": "wide", "candidate": "mix"},
            {"baseline": "mix", "candidate": "q8"},
        ]

    def test_pooled_contributions_reproduce_official_wp_and_paired_delta(self) -> None:
        """Each shared partition sums to the official metric and endpoint delta."""
        report = analyze_arrays(
            self.targets,
            self.predictions,
            self.row_group,
            self.component,
            None,
            self.comparisons,
        )
        models = report["models"]
        for name, prediction in self.predictions.items():
            score = global_wp(self.targets, prediction, np.ones(len(self.targets), dtype=bool))
            self.assertAlmostEqual(models[name]["score"]["wp"], score["wp"], places=12)
            for regime in ("retrospective_label_abs_clipped", "component"):
                contributions = models[name]["regimes"][regime]
                self.assertAlmostEqual(
                    sum(row["global_wp_contribution"] for row in contributions),
                    score["wp"],
                    places=12,
                )
                masses = [
                    sum(row["weight_mass"] for row in contributions if row["target"] == target)
                    for target in range(2)
                ]
                for target in range(2):
                    clipped = np.clip(self.targets[:, target], -2, 2).astype(np.float64)
                    expected_mass = np.abs(clipped).sum()
                    self.assertAlmostEqual(masses[target], float(expected_mass), places=12)

        for pair in report["comparisons"]:
            endpoint_delta = pair["candidate_minus_baseline_wp"]
            for regime_rows in pair["shared_membership_regime_contributions"].values():
                self.assertAlmostEqual(
                    sum(row["global_wp_contribution_delta"] for row in regime_rows),
                    endpoint_delta,
                    places=12,
                )
            self.assertFalse(pair["predicted_magnitude_bins_paired"])
            self.assertNotIn(
                "prediction_abs_clipped", pair["shared_membership_regime_contributions"]
            )

    def test_zero_weight_label_bin_and_raw_clipping_are_explicit(self) -> None:
        """Zero-weight labels keep zero mass; model-specific bins are descriptive only."""
        report = analyze_arrays(
            self.targets,
            self.predictions,
            self.row_group,
            self.component,
            None,
            self.comparisons,
        )
        label_rows = report["models"]["wide"]["regimes"]["retrospective_label_abs_clipped"]
        zero_bin = next(
            row for row in label_rows if row["target"] == 0 and row["category"] == "[0,0.05)"
        )
        self.assertEqual(zero_bin["rows"], 3)
        self.assertEqual(zero_bin["weight_mass"], 0.0)
        self.assertEqual(zero_bin["global_wp_contribution"], 0.0)
        for pair in report["comparisons"]:
            row = next(
                row
                for row in pair["shared_membership_regime_contributions"][
                    "retrospective_label_abs_clipped"
                ]
                if row["target"] == 0 and row["category"] == "[0,0.05)"
            )
            self.assertIsNone(row["weighted_bias_delta"])
            self.assertIsNone(row["within_regime_rho_delta_descriptive_only"])
        prediction_rows = report["models"]["q8"]["regimes"]["prediction_raw_clipping"]
        clipped_target_zero = next(
            row
            for row in prediction_rows
            if row["target"] == 0 and row["category"] == "clipped_gt_2"
        )
        self.assertEqual(clipped_target_zero["rows"], 1)
        self.assertFalse(clipped_target_zero["membership_shared_across_models"])
        self.assertIn("no parquet scan", report["position_status"])

    def test_csv_writer_projects_every_emitted_record_field(self) -> None:
        """The flat table includes additive numerators and calibration diagnostics."""
        report = analyze_arrays(
            self.targets,
            self.predictions,
            self.row_group,
            self.component,
            None,
            self.comparisons,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic.csv"
            write_csv(path, report)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
        self.assertGreater(len(rows), 0)
        self.assertIn("pooled_covariance_numerator", rows[0])
        self.assertIn("pooled_covariance_share", rows[0])
        self.assertIn("weighted_prediction_on_target_slope", rows[0])


if __name__ == "__main__":
    unittest.main()
