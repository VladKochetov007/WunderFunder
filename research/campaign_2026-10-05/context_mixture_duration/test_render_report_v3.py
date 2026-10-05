"""Synthetic-only schema and output checks for the E488 report renderer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import render_report_v3 as renderer


def summary(point: float) -> dict:
    return {
        "point_diff": point,
        "bootstrap_sd": 0.01,
        "ci95": [point - 0.02, point + 0.02],
        "p_le_0": 0.4,
        "components": 16,
        "bootstrap_count": 2000,
        "bootstrap_seed": 271839,
    }


def fixture() -> tuple[dict, dict]:
    arm_scores = {}
    for seed_index, seed in enumerate(renderer.SEEDS):
        for epoch in renderer.EPOCHS:
            for arm_index, arm in enumerate(renderer.ARMS):
                key = f"seed{seed}__{arm}__epoch{epoch}"
                arm_scores[key] = {
                    "seed": seed,
                    "arm": arm,
                    "epoch": epoch,
                    "parameters": 61058 if arm in ("gelu", "fixed_mix") else 61122,
                    "updates": epoch * 474,
                    "wp": 0.4 + 0.01 * seed_index + 0.001 * arm_index + 0.0001 * epoch,
                    "t0": 0.3 + 0.01 * seed_index + 0.001 * arm_index,
                    "t1": 0.5 + 0.01 * seed_index - 0.001 * arm_index,
                    "metric_id": "global_wp_clip2_v1",
                    "selected_rows": 26990,
                }
    global_rows = {}
    target_rows = {}
    for seed in renderer.SEEDS:
        for epoch in renderer.EPOCHS:
            for contrast_index, contrast in enumerate(renderer.CONTRASTS):
                key = f"seed{seed}__epoch{epoch}__{contrast}"
                row = summary(0.001 * (contrast_index - 1) * epoch)
                global_rows[key] = {
                    **row,
                    "seed": seed,
                    "epoch": epoch,
                    "contrast": contrast,
                }
                target_rows[key] = {
                    "seed": seed,
                    "epoch": epoch,
                    "contrast": contrast,
                    "targets": {"t0": summary(0.0005), "t1": summary(-0.0003)},
                }
        key = f"seed{seed}__duration_interaction"
        global_rows[key] = {
            **summary(0.002),
            "seed": seed,
            "epoch": None,
            "contrast": "duration_interaction",
        }
        target_rows[key] = {
            "seed": seed,
            "epoch": None,
            "contrast": "duration_interaction",
            "targets": {"t0": summary(0.001), "t1": summary(-0.001)},
        }
    score = {
        "metric_id": "global_wp_clip2_v1",
        "trajectory_count": 12,
        "snapshot_count": 24,
        "registered_global_contrast_count": 27,
        "registered_target_contrast_count": 27,
        "no_seed_pooling": True,
        "test_labels_used_for_fitting_or_selection": False,
        "bootstrap": {
            "draws": 2000,
            "seed": 271839,
            "multiplicities_shared_across_all_24_snapshots_and_contrasts": True,
        },
        "arm_scores": arm_scores,
        "registered_global_contrasts": global_rows,
        "registered_target_contrasts": target_rows,
    }
    reproduction = {
        "study": "E488",
        "status": "all_checked_values_match",
        "result_sha256": "r" * 64,
        "export_sha256": "e" * 64,
        "arms": 24,
        "global_contrasts": 27,
        "target_contrasts": 54,
        "maximum_absolute_error": 1e-14,
    }
    return score, reproduction


class E488ReportRendererTests(unittest.TestCase):
    def test_registered_matrix_and_root_binding_pass(self) -> None:
        scores, reproduction = fixture()
        renderer.validate_numeric_summary(scores, reproduction, "r" * 64, "e" * 64)

    def test_missing_arm_fails_closed(self) -> None:
        scores, reproduction = fixture()
        scores["arm_scores"].pop(next(iter(scores["arm_scores"])))
        with self.assertRaisesRegex(ValueError, "arm snapshot inventory"):
            renderer.validate_numeric_summary(scores, reproduction, "r" * 64, "e" * 64)

    def test_synthetic_outputs_preserve_full_matrices(self) -> None:
        scores, _reproduction = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            renderer.render_arm_scores(scores, output / "arms.png")
            renderer.render_global_contrasts(scores, output / "global.png")
            renderer.render_target_contrasts(scores, output / "targets.png")
            renderer.write_csvs(scores, output)
            self.assertEqual(len((output / "all24_arm_snapshots.csv").read_text().splitlines()), 25)
            self.assertEqual(
                len((output / "all27_global_contrasts.csv").read_text().splitlines()), 28
            )
            self.assertEqual(
                len((output / "all54_target_contrasts.csv").read_text().splitlines()), 55
            )
            self.assertTrue(
                all(
                    (output / name).stat().st_size > 0
                    for name in ("arms.png", "global.png", "targets.png")
                )
            )


if __name__ == "__main__":
    unittest.main()
