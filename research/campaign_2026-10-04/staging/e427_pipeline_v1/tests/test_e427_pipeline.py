"""Stdlib-only controller gates for actual E427 stats and future runner records."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
sys.path.insert(0, str(CAMPAIGN / "staging/e427_pipeline_v1/scripts"))

import run_e427_pipeline as queue  # noqa: E402


def smoke_payload(arm: str) -> dict:
    return {
        "schema": "wnn.e427-production-smoke-record-v1",
        "arm": arm,
        "status": "pass",
        "exit_code": 0,
        "model_checkpoint_written": False,
        "scientific_fit": False,
        "no_holdout_evaluation": True,
        "training_result": {
            "status": "pass",
            "arm": arm,
            "completed_loader_batches": 2,
            "optimizer_attempts": 40,
            "optimizer_completed_updates": 39,
            "amp_skipped_updates": 1,
            "ema_updates": 40,
            "student_projection_audits": 40,
            "ema_projection_audits": 40,
            "loader_batch_shape": [32, 20000, 211],
            "peak_cuda_allocated_bytes": 2_000_000_000,
            "peak_cuda_reserved_bytes": 2_500_000_000,
            "peak_host_rss_bytes": 4_000_000_000,
            "host_rss_limit_bytes": 24 * (1 << 30),
            "allocator_limit_bytes": 4200000000,
            "stop_limit_bytes": 4800000000,
            "failures": [],
            "initial_parent_cuda_parity": {
                "parent_parity_pass": True,
                "causal_prefix_future_suffix_pass": True,
            },
        },
    }


def e464_payload() -> tuple[dict, dict]:
    completed = [*queue.EXPECTED_E464_FITS, *queue.EXPECTED_E464_EXPORTS,
                 queue.EXPECTED_E464_SCORE]
    report_sha = "a" * 64
    sequence = {
        "status": "complete",
        "completed": completed,
        "paired_comparisons": 16,
        "bootstrap_replicates": 2000,
        "paired_score_path": "research/campaign_2026-10-04/staging/e464/results/e464_report_all_fp32_v1.json",
        "paired_score_sha256": report_sha,
    }
    report = {
        "metric_id": "global_wp_clip2_v1",
        "n_sequences": 368,
        "screen": {"folds": ["holdout"], "comparisons": [[str(i), str(i + 1)] for i in range(16)]},
    }
    return sequence, report


class E427PipelineTests(unittest.TestCase):
    def test_actual_context_stats_are_fit_only_and_have_expected_memberships(self) -> None:
        stats = json.loads((CAMPAIGN / "checks/e427_context_stats_s3_v1.json").read_text())
        parent = json.loads((ROOT / "artifacts/runs/c20261004_combo_s3_final_r1/provenance.json").read_text())
        self.assertEqual(stats["format"], "wnn.e427-volume28-fit-stats")
        self.assertIs(stats["scoring_rows_used"], False)
        self.assertEqual(stats["row_count"], 242_220_000)
        self.assertEqual(stats["fit_row_groups"]["train"], parent["fit_row_groups"]["train"])
        self.assertEqual(set(stats["fit_row_groups"]["valid"]), set(parent["fit_row_groups"]["valid"]))
        self.assertEqual(len(stats["fit_row_groups"]["train"]), 10606)
        self.assertEqual(len(stats["fit_row_groups"]["valid"]), 1505)
        self.assertEqual(len(stats["evaluation_row_groups_excluded"]), 368)
        self.assertFalse(set(stats["evaluation_row_groups_excluded"]) & set(stats["fit_row_groups"]["valid"]))

    def test_synthetic_e464_success_record_passes_fixed_design_gate(self) -> None:
        sequence, report = e464_payload()
        queue.validate_e464_payload(sequence, report, "a" * 64)

    def test_e464_must_match_metric_panel_and_all_sixteen_comparisons(self) -> None:
        sequence, report = e464_payload()
        bad_panel = json.loads(json.dumps(report))
        bad_panel["screen"]["folds"] = ["fold0"]
        with self.assertRaises(ValueError):
            queue.validate_e464_payload(sequence, bad_panel, "a" * 64)
        bad_metric = json.loads(json.dumps(report))
        bad_metric["metric_id"] = "legacy_sequence_mean"
        with self.assertRaises(ValueError):
            queue.validate_e464_payload(sequence, bad_metric, "a" * 64)
        bad_comparisons = json.loads(json.dumps(report))
        bad_comparisons["screen"]["comparisons"].pop()
        with self.assertRaises(ValueError):
            queue.validate_e464_payload(sequence, bad_comparisons, "a" * 64)

    def test_e464_success_is_rejected_for_attempt_order_or_score_hash_changes(self) -> None:
        sequence, report = e464_payload()
        sequence["completed"].reverse()
        with self.assertRaises(ValueError):
            queue.validate_e464_payload(sequence, report, "a" * 64)
        sequence, report = e464_payload()
        with self.assertRaises(ValueError):
            queue.validate_e464_payload(sequence, report, "b" * 64)

    def test_smoke_record_requires_full_shape_causal_and_attempt_accounting(self) -> None:
        for arm in queue.ARMS:
            marker = queue.validate_smoke_payload(smoke_payload(arm), arm)
            self.assertEqual(marker["optimizer_attempts"], 40)
        for field, value in (("optimizer_attempts", 20), ("ema_updates", 39),
                             ("loader_batch_shape", [16, 20000, 211])):
            bad = smoke_payload("aligned")
            bad["training_result"][field] = value
            with self.assertRaises(ValueError):
                queue.validate_smoke_payload(bad, "aligned")
        bad = smoke_payload("stale")
        bad["training_result"]["initial_parent_cuda_parity"]["causal_prefix_future_suffix_pass"] = False
        with self.assertRaises(ValueError):
            queue.validate_smoke_payload(bad, "stale")
        bad = smoke_payload("noop")
        bad["model_checkpoint_written"] = True
        with self.assertRaises(ValueError):
            queue.validate_smoke_payload(bad, "noop")

    def test_prediction_group_domain_is_exact_ordered_and_typed(self) -> None:
        self.assertTrue(queue.prediction_domain_is_exact(list(range(1873))))
        self.assertFalse(queue.prediction_domain_is_exact(list(range(1872)) + ["1872"]))
        self.assertFalse(queue.prediction_domain_is_exact(list(range(1872)) + [True]))
        self.assertFalse(queue.prediction_domain_is_exact(list(range(1, 1874))))

    def test_runtime_fit_specs_preserve_recipe_and_withhold_fit_time_evaluation(self) -> None:
        for arm in queue.ARMS:
            path = CAMPAIGN / "staging/e427_pipeline_v1/execution_specs" / (
                f"c20261004_e427_{arm}_s3_1ep_v2.draft.json"
            )
            spec = json.loads(path.read_text())
            argv = spec["argv"]
            self.assertEqual(spec["kind"], "gpu")
            self.assertEqual(spec["limits"]["timeout_seconds"], 3600)
            self.assertEqual(spec["limits"]["gpu_memory_bytes"], 4200000000)
            self.assertEqual(spec["limits"]["gpu_stop_bytes"], 4800000000)
            self.assertIn("--no-eval-during-fit", argv)
            self.assertEqual(argv[argv.index("--e427-arm") + 1], arm)
            self.assertEqual(argv[argv.index("--init") + 1], "c20261004_combo_s3_final_r1@ep3")
            self.assertEqual(argv[argv.index("--checkpoint-epoch") + 1], "0")
            self.assertEqual(argv[argv.index("--epochs") + 1], "1")

    def test_registered_smokes_are_full_shape_and_do_not_create_checkpoints(self) -> None:
        for arm in queue.ARMS:
            path = CAMPAIGN / "staging/e427_smoke_review/specs" / (
                f"e427_smoke_{arm}_s3_b32x2_v1.draft.json"
            )
            spec = json.loads(path.read_text())
            argv = spec["argv"]
            self.assertEqual(argv[argv.index("--arm") + 1], arm)
            self.assertEqual(argv[argv.index("--profile-steps") + 1], "2")
            self.assertEqual(argv[argv.index("--batch") + 1], "32")
            self.assertIn("--no-eval-during-fit", argv)
            self.assertNotIn("--save-training-state", argv)
            self.assertEqual(spec["limits"]["timeout_seconds"], 900)

    def test_three_panel_design_freezes_overlap_counts_before_any_score(self) -> None:
        design = json.loads((CAMPAIGN / "staging/e427_pipeline_v1/specs/e427_screen_v1.json").read_text())
        self.assertEqual(design["metric_id"], "global_wp_clip2_v1")
        self.assertEqual(design["bootstrap_replicates"], 2000)
        self.assertEqual(design["bootstrap_seed"], 271828)
        self.assertEqual(design["panels"]["fold0_primary_historically_exposed"]["expected_n"], 468)
        self.assertEqual(design["panels"]["fold0_primary_historically_exposed"]["expected_fit_overlap"], 374)
        self.assertEqual(design["panels"]["stats_excluded_368_historically_exposed"]["expected_fit_overlap"], 0)
        self.assertEqual(design["panels"]["fold0_nonfit_94_diagnostic"]["expected_n"], 94)
        self.assertIn("no independent", design["qualification"].lower())

    def test_next_timeout_reservation_counts_prior_gpu_charge(self) -> None:
        # The complete three-fit reservation is 10,800 seconds before any fit starts.
        self.assertLessEqual(0 + 3 * FIT_TIMEOUT_LOCAL, queue.MAX_GPU_SECONDS)
        self.assertGreater(24 * 3600 - 3599 + 3600, queue.MAX_GPU_SECONDS)


FIT_TIMEOUT_LOCAL = 3600


if __name__ == "__main__":
    unittest.main()
