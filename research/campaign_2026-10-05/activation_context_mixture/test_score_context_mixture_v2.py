"""Small synthetic contract tests for the independent E484 grader."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_optimizer"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_placement"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_head_mixture"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-05/activation_context_mixture"))

import fit_context_mixture_v2 as actor  # noqa: E402
import score_context_mixture_v2 as scorer  # noqa: E402


class E484GraderSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_independent_forward_and_carried_state_match_all_arms(self) -> None:
        torch.manual_seed(484900)
        for arm in scorer.ARMS:
            with self.subTest(arm=arm):
                state = actor.make_initial_states()[actor.SEEDS[0]]
                candidate = actor.initialize_arm(arm, state).eval()
                reconstructed = scorer.IndependentE484Model(arm).eval()
                reconstructed.load_state_dict(candidate.state_dict(), strict=True)
                x = torch.randn(2, 19, scorer.FEATURES)
                with torch.no_grad():
                    expected, expected_state = candidate(x)
                    actual, actual_state = reconstructed(x)
                    expected_first, expected_carry = candidate(x[:, :7])
                    expected_second, expected_carry = candidate(x[:, 7:13], expected_carry)
                    expected_third, expected_carry = candidate(x[:, 13:], expected_carry)
                    actual_first, actual_carry = reconstructed(x[:, :7])
                    actual_second, actual_carry = reconstructed(x[:, 7:13], actual_carry)
                    actual_third, actual_carry = reconstructed(x[:, 13:], actual_carry)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)
                torch.testing.assert_close(actual_carry, expected_state, atol=0, rtol=0)
                torch.testing.assert_close(actual_first, expected_first, atol=0, rtol=0)
                torch.testing.assert_close(actual_second, expected_second, atol=0, rtol=0)
                torch.testing.assert_close(actual_third, expected_third, atol=0, rtol=0)

    def test_parameter_keys_counts_and_identity_buffers(self) -> None:
        for arm in scorer.ARMS:
            model = scorer.IndependentE484Model(arm)
            state = model.state_dict()
            self.assertEqual(set(state), scorer.expected_state_keys(arm))
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters()),
                scorer.EXPECTED_PARAMETERS[arm],
            )
            scorer.validate_identity_buffers(state)
        bad = dict(scorer.IndependentE484Model("gelu").state_dict())
        bad["mean"] = torch.ones_like(bad["mean"])
        with self.assertRaises(ValueError):
            scorer.validate_identity_buffers(bad)

    def test_bootstrap_is_numeric_sorted_and_shared(self) -> None:
        components = np.asarray([19, 3, 11, 2, 23, 5, 17, 7, 13, 1, 21, 9, 15, 4, 6, 8])
        sorted_ids, order, counts = scorer._bootstrap_counts(components)
        np.testing.assert_array_equal(sorted_ids, np.sort(components))
        np.testing.assert_array_equal(components[order], sorted_ids)
        self.assertEqual(counts.shape, (scorer.BOOTSTRAP_COUNT, 16))
        np.testing.assert_array_equal(counts.sum(axis=1), np.full(scorer.BOOTSTRAP_COUNT, 16))
        again = scorer._bootstrap_counts(components)[2]
        np.testing.assert_array_equal(counts, again)
        with self.assertRaises(ValueError):
            scorer._bootstrap_counts(np.zeros(16, dtype=np.int64))

    def test_targetwise_moment_pool_matches_official_metric(self) -> None:
        rng = np.random.default_rng(484901)
        targets = rng.normal(size=(16, 7, 2)).astype(np.float32)
        predictions = (targets + 0.3 * rng.normal(size=targets.shape)).astype(np.float32)
        mask = rng.random((16, 7)) > 0.2
        candidate_moments, _ = scorer.paired_component_moments(
            targets, predictions, predictions, mask
        )
        _, _, counts = scorer._bootstrap_counts(np.arange(16))
        for draw in counts[:12]:
            expected = np.zeros(2)
            selected_targets = []
            selected_predictions = []
            selected_mask = []
            for component, multiplicity in enumerate(draw):
                for _ in range(int(multiplicity)):
                    selected_targets.append(targets[component])
                    selected_predictions.append(predictions[component])
                    selected_mask.append(mask[component])
            metric = scorer.global_wp(
                np.stack(selected_targets),
                np.stack(selected_predictions),
                np.stack(selected_mask),
            )
            expected[:] = (metric["t0"], metric["t1"])
            np.testing.assert_allclose(
                scorer._score_component_moments_by_target(candidate_moments, draw),
                expected,
                atol=2e-7,
                rtol=2e-7,
            )

    def test_scoring_path_is_not_called_by_fit_only_mode(self) -> None:
        parser_source = (HERE / "score_context_mixture_v2.py").read_text()
        self.assertIn('modes.add_argument("--verify-fit-only"', parser_source)
        self.assertIn('modes.add_argument("--score"', parser_source)
        self.assertIn("if args.verify_fit_only:", parser_source)
        self.assertLess(
            parser_source.index("if args.verify_fit_only:"),
            parser_source.rindex("score_validated_run("),
        )

    def test_actual_attempt_contract_returns_artifact_files_not_parent_dirs(self) -> None:
        runner = ROOT / "research/campaign_2026-10-04/runner"
        cases = (
            (
                "probe",
                HERE / "e484_gpu_probe_v2.sealed.json",
                runner / "attempts/e484-context-mixture-real-gpu-probe-v2.json",
                ROOT
                / "research/campaign_2026-10-05/activation_context_mixture/gpu_probe_v2/probe.json",
            ),
            (
                "fit",
                HERE / "e484_twelve_fit_v2.sealed.json",
                runner / "attempts/e484-context-mixture-twelve-fit-v2.json",
                ROOT
                / "research/campaign_2026-10-05/activation_context_mixture/runs/twelve_v2/fit_manifest.json",
            ),
        )
        for phase, spec_path, attempt_path, expected_artifact in cases:
            with self.subTest(phase=phase):
                _record, _spec, artifact = scorer.validate_attempt(attempt_path, spec_path, phase)
                self.assertEqual(artifact.resolve(), expected_artifact.resolve())
                self.assertTrue(artifact.is_file())

    @unittest.skipUnless(
        os.environ.get("E484_RUN_ACTUAL_FIT_ONLY_TEST") == "1",
        "set E484_RUN_ACTUAL_FIT_ONLY_TEST=1 to check the real completed fit without test data",
    )
    def test_actual_complete_fit_contract_keeps_test_rows_unread(self) -> None:
        runner = ROOT / "research/campaign_2026-10-04/runner"
        fit_run = ROOT / "research/campaign_2026-10-05/activation_context_mixture/runs/twelve_v2"
        probe_run = ROOT / "research/campaign_2026-10-05/activation_context_mixture/gpu_probe_v2"
        proof, cells, report, _manifest, _fit_x, _fit_nodes = scorer.validate_full_run(
            fit_run,
            HERE / "e484_twelve_fit_v2.sealed.json",
            runner / "attempts/e484-context-mixture-twelve-fit-v2.json",
            probe_run,
            HERE / "e484_gpu_probe_v2.sealed.json",
            runner / "attempts/e484-context-mixture-real-gpu-probe-v2.json",
        )
        self.assertEqual(len(cells), 12)
        self.assertEqual(report["registered_contrast_count"], 16)
        self.assertFalse(report["test_rows_loaded"])
        self.assertEqual(
            proof["fit_test_boundary"], "validated before test loader; no test rows read"
        )


if __name__ == "__main__":
    unittest.main()
