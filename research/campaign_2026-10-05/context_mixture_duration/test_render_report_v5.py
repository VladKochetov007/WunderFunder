"""Synthetic-only schema and output checks for the E488 report renderer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import render_report_v5 as renderer


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
        "status": "all_raw_export_scores_and_draw_summaries_match",
        "score_sha256": "r" * 64,
        "export_sha256": "e" * 64,
        "arm_snapshots": 24,
        "global_contrasts": 27,
        "target_contrasts": 54,
        "bootstrap_draws_per_summary": 2000,
        "bootstrap_draw_vectors_recomputed": {
            "arm_global_scores": 24,
            "arm_target_scores": 48,
            "global_contrast_arrays": 27,
            "target_contrast_scores": 54,
        },
        "no_seed_pooling": True,
        "max_abs_error": 1e-14,
        "tolerance": 1e-10,
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

    def test_incomplete_root_draw_inventory_fails_closed(self) -> None:
        scores, reproduction = fixture()
        reproduction["bootstrap_draw_vectors_recomputed"]["target_contrast_scores"] = 53
        with self.assertRaisesRegex(ValueError, "draw-vector inventory"):
            renderer.validate_numeric_summary(scores, reproduction, "r" * 64, "e" * 64)

    def test_plot_handles_interval_that_does_not_contain_point(self) -> None:
        scores, _reproduction = fixture()
        row = scores["registered_global_contrasts"][
            f"seed{renderer.SEEDS[0]}__epoch1__{renderer.CONTRASTS[0]}"
        ]
        row["point_diff"] = 0.0
        row["ci95"] = [0.01, 0.02]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "global.png"
            renderer.render_global_contrasts(scores, output)
            self.assertGreater(output.stat().st_size, 0)

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

    def test_complete_result_only_render_pipeline(self) -> None:
        scores, reproduction = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proof_path = root / "proof.json"
            fit_path = root / "fit.json"
            grade_path = root / "grade.json"
            score_path = root / "scores.json"
            export_path = root / "selected_predictions.npz"
            reproduction_path = root / "raw_reproduction.json"
            wrapper_path = (
                root
                / "research/campaign_2026-10-05/checks/context_duration_export_root_reproduction.json"
            )
            output_dir = root / "rendered"
            report_path = root / "REPORT.md"
            proof = {
                "status": "all_fit_only_evidence_passed_test_rows_unread",
                "snapshot_count": 24,
            }
            fit_path.write_text(
                renderer.json.dumps(
                    {
                        "status": "finished",
                        "exit_code": 0,
                        "teardown_verified": True,
                        "signals": [],
                        "elapsed_seconds": 114.0,
                        "peak_ram_bytes": 1000,
                        "peak_observed_vram_bytes": 500,
                    }
                )
            )
            proof["fit_attempt_sha256"] = renderer.sha256_file(fit_path)
            proof_path.write_text(renderer.json.dumps(proof))
            export_path.write_bytes(b"synthetic saved export placeholder")
            export_sha = renderer.sha256_file(export_path)
            scores.update(
                {
                    "proof_sha256": renderer.sha256_file(proof_path),
                    "selected_export_sha256": export_sha,
                    "selected_export_bytes": export_path.stat().st_size,
                }
            )
            score_path.write_text(renderer.json.dumps(scores))
            score_sha = renderer.sha256_file(score_path)
            reproduction.update({"score_sha256": score_sha, "export_sha256": export_sha})
            reproduction_path.write_text(renderer.json.dumps(reproduction))
            wrapper_path.parent.mkdir(parents=True)
            wrapper = {
                "study": "E488",
                "status": "all_checked_values_match",
                "arms": 24,
                "global_contrasts": 27,
                "target_contrasts": 54,
                "result_sha256": score_sha,
                "export_sha256": export_sha,
                "raw_arithmetic_receipt_sha256": renderer.sha256_file(reproduction_path),
                "raw_bootstrap_arrays_all_draws_verified": True,
            }
            wrapper_path.write_text(renderer.json.dumps(wrapper))
            grade_path.write_text(
                renderer.json.dumps(
                    {
                        "status": "finished",
                        "exit_code": 0,
                        "teardown_verified": True,
                        "signals": [],
                        "elapsed_seconds": 5.0,
                        "peak_ram_bytes": 900,
                        "artifacts": [
                            {
                                "path": "scores.json",
                                "sha256": score_sha,
                                "size": score_path.stat().st_size,
                            },
                            {
                                "path": "selected_predictions.npz",
                                "sha256": export_sha,
                                "size": export_path.stat().st_size,
                            },
                        ],
                    }
                )
            )
            with patch.object(renderer, "ROOT", root):
                renderer.render(
                    score_path,
                    proof_path,
                    export_path,
                    fit_path,
                    grade_path,
                    reproduction_path,
                    output_dir,
                    report_path,
                )
            report = report_path.read_text()
            self.assertIn("maximum absolute difference was 1.000e-14", report)
            self.assertNotIn("difference was nan", report.lower())
            self.assertTrue((output_dir / "all24_arm_snapshots.csv").is_file())
            self.assertTrue((output_dir / "all27_global_contrasts.png").is_file())


if __name__ == "__main__":
    unittest.main()
