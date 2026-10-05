"""Synthetic E483 grader checks; no parquet rows or fitted scores are read."""

from __future__ import annotations

import unittest

import numpy as np
import score_momentum as scorer
import torch

from wnn.nn import RNNModel


class MomentumGraderTests(unittest.TestCase):
    def test_candidate_model_matches_native_baseline_outputs_states_and_gradients(self) -> None:
        torch.manual_seed(8281)
        native = RNNModel(42, hidden=64, layers=2, dropout=0.0, n_out=2)
        independent = scorer.IndependentMomentumModel(sine=False)
        independent.load_state_dict(native.state_dict(), strict=True)
        state = torch.randn(2, 3, 64) * 0.1
        x_native = torch.randn(3, 23, 42, requires_grad=True)
        x_independent = x_native.detach().clone().requires_grad_(True)
        expected, expected_state = native(x_native, state.clone())
        actual, actual_state = independent(x_independent, state.clone())
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-5, atol=2e-6)
        (actual.square().mean() + actual_state.square().mean()).backward()
        (expected.square().mean() + expected_state.square().mean()).backward()
        torch.testing.assert_close(x_independent.grad, x_native.grad, rtol=5e-5, atol=5e-6)
        native_params = dict(native.named_parameters())
        independent_params = dict(independent.named_parameters())
        self.assertEqual(set(native_params), set(independent_params))
        for name in native_params:
            torch.testing.assert_close(
                independent_params[name].grad,
                native_params[name].grad,
                rtol=5e-5,
                atol=5e-6,
                msg=name,
            )

    def test_chunk_state_carry_matches_full_sequence(self) -> None:
        torch.manual_seed(99)
        model = scorer.IndependentMomentumModel(sine=True).eval()
        x = torch.randn(2, 31, 42)
        with torch.no_grad():
            full, full_state = model(x)
            first, state = model(x[:, :12])
            second, state = model(x[:, 12:], state)
        torch.testing.assert_close(torch.cat((first, second), dim=1), full, rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(state, full_state, rtol=2e-6, atol=2e-6)

    def test_raw_labels_reject_nonfinite_before_clip(self) -> None:
        for value in (np.inf, -np.inf, np.nan):
            raw = np.zeros((4, 2), dtype=np.float64)
            raw[0, 1] = value
            with self.assertRaises(ValueError):
                scorer.validate_raw_labels(raw)
        clipped_edge = scorer.validate_raw_labels(np.asarray([[3.0, -3.0]]))
        np.testing.assert_array_equal(clipped_edge, [[3.0, -3.0]])

    def test_official_wp_matches_centered_component_moments(self) -> None:
        rng = np.random.default_rng(410)
        targets = rng.normal(size=(5, 41, 2)).astype(np.float32) * 3.0
        predictions = (targets + rng.normal(size=targets.shape)).astype(np.float32)
        mask = rng.random((5, 41)) > 0.25
        metrics = scorer.global_wp(targets, predictions, mask)
        moments = scorer.component_moments(targets, predictions, mask)
        pooled = scorer.pool_component_moments(moments, np.ones(5, dtype=np.int16))
        self.assertAlmostEqual(pooled, float(metrics["wp"]), places=12)
        self.assertEqual(metrics["metric_id"], scorer.METRIC_ID)
        self.assertEqual(metrics["selected_rows"], int(mask.sum()))

    def test_sorted_bootstrap_is_deterministic_and_conserves_components(self) -> None:
        component_ids = np.arange(2000, 2016, dtype=np.int64)
        first = scorer.bootstrap_multiplicities(component_ids)
        second = scorer.bootstrap_multiplicities(component_ids)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2000, 16))
        np.testing.assert_array_equal(first.sum(axis=1), np.full(2000, 16))
        with self.assertRaises(ValueError):
            scorer.bootstrap_multiplicities(component_ids[::-1])

    def test_registered_contrasts_have_nine_correct_effects_and_did(self) -> None:
        names = [f"{shape}_beta{beta}" for shape in scorer.SHAPES for beta in scorer.MOMENTA]
        scores = {name: float(index) / 10.0 for index, name in enumerate(names)}
        draws = {name: np.full(scorer.BOOTSTRAP_COUNT, score) for name, score in scores.items()}
        result = scorer.registered_contrasts(scores, draws)
        self.assertEqual(len(result), 9)
        self.assertAlmostEqual(result["sine_minus_baseline_beta0.8"]["mean_diff"], 0.3)
        expected = (scores["sine_beta0.8"] - scores["baseline_beta0.8"]) - (
            scores["sine_beta0.9"] - scores["baseline_beta0.9"]
        )
        self.assertAlmostEqual(
            result["shape_by_momentum_interaction_beta0.8_vs0.9"]["mean_diff"], expected
        )
        self.assertTrue(
            all(item["ci95"] == [item["mean_diff"], item["mean_diff"]] for item in result.values())
        )

    def test_optimizer_group_receipt_rejects_beta_leak(self) -> None:
        model = scorer.IndependentMomentumModel(sine=False)
        names = {name for name, _parameter in model.named_parameters()}
        groups = [
            {
                "muon": True,
                "names": sorted(scorer.MUON_NAMES),
                "betas": [0.8, 0.999],
                "lr": 0.001,
                "weight_decay": 0.01,
                "blocks": 3,
                "ns_steps": 5,
                "rms": 0.2,
            },
            {
                "muon": False,
                "names": sorted(names - scorer.MUON_NAMES),
                "betas": [0.9, 0.999],
                "lr": 0.001,
                "weight_decay": 0.01,
            },
        ]
        scorer.validate_optimizer_groups(groups, names, 0.8)
        groups[1]["betas"] = [0.8, 0.999]
        with self.assertRaises(ValueError):
            scorer.validate_optimizer_groups(groups, names, 0.8)


if __name__ == "__main__":
    unittest.main()
