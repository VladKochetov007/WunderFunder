"""Synthetic-only tests for the E486 crossed inference scorer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import score_counterfactual as scorer
import torch


class CounterfactualScorerTests(unittest.TestCase):
    def test_manual_recurrence_matches_frozen_model_for_both_shapes(self) -> None:
        for sine in (False, True):
            torch.manual_seed(912 + int(sine))
            model = scorer.e483.IndependentMomentumModel(sine=sine).eval()
            x = torch.randn(3, 19, scorer.e483.FEATURES)
            state = torch.randn(2, 3, scorer.e483.HIDDEN) * 0.1
            expected, expected_state = model(x, state.clone())
            actual, actual_state, maximum = scorer.forward_with_state_extrema(
                model, x, state.clone()
            )
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(actual_state, expected_state))
            self.assertGreaterEqual(maximum, float(expected_state.detach().abs().max()))

    def test_shared_cross_contrasts_use_all_four_cell_draws(self) -> None:
        scores = {
            "baseline_baseline": 0.1,
            "baseline_sine": 0.2,
            "sine_baseline": 0.4,
            "sine_sine": 0.7,
        }
        draws = {name: np.full(2_000, value, dtype=np.float64) for name, value in scores.items()}
        result = scorer.registered_cross_contrasts(scores, draws)
        self.assertEqual(len(result), 6)
        self.assertAlmostEqual(result["B_minus_A_inference_sine_on_base_fit"]["mean_diff"], 0.1)
        self.assertAlmostEqual(result["C_minus_D_remove_sine_on_sine_fit"]["mean_diff"], -0.3)
        self.assertAlmostEqual(result["training_by_inference_interaction"]["mean_diff"], 0.2)
        self.assertTrue(
            all(
                value["ci95"] == [value["mean_diff"], value["mean_diff"]]
                for value in result.values()
            )
        )

    def test_exact_diagonal_prediction_gate_rejects_single_changed_value(self) -> None:
        reference_predictions = np.zeros((18, 7, 2), dtype=np.float32)
        names = [scorer.e483.arm_key(seed, shape, beta) for seed, shape, beta in scorer.e483.ARMS]
        reference = {
            "arm_names": names,
            "arrays": {"predictions_raw_selected_by_arm": reference_predictions},
        }
        predictions = {
            scorer.cell_key(seed, train_shape, eval_shape, beta): np.zeros((7, 2), dtype=np.float32)
            for seed in scorer.e483.SEEDS
            for train_shape in scorer.e483.SHAPES
            for eval_shape in scorer.e483.SHAPES
            for beta in scorer.e483.MOMENTA
        }
        scorer.verify_diagonal_prediction_gate(predictions, reference)
        key = scorer.cell_key(271_831, "baseline", "baseline", 0.8)
        predictions[key][2, 1] = np.nextafter(np.float32(0.0), np.float32(1.0))
        with self.assertRaisesRegex(ValueError, "diagonal raw predictions"):
            scorer.verify_diagonal_prediction_gate(predictions, reference)

    def test_real_scoring_requires_a_root_sealed_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "would_be_results"
            with (
                patch(
                    "sys.argv",
                    [
                        "score_counterfactual.py",
                        "--score-exposed-test",
                        "--output",
                        str(output),
                    ],
                ),
                self.assertRaisesRegex(ValueError, "requires root's --sealed-spec"),
            ):
                scorer.main()
            self.assertFalse(output.exists())

    def test_synthetic_crossed_matrix_has_36_cells_54_contrasts_and_exports(self) -> None:
        rng = np.random.default_rng(3486)
        rows = 13
        x = rng.normal(size=(16, rows, scorer.e483.FEATURES)).astype(np.float32)
        targets = rng.normal(size=(16, rows, 2)).astype(np.float32)
        selected = np.ones((16, rows), dtype=bool)
        component_ids = np.arange(2_000, 2_016, dtype=np.int64)
        nodes = [
            {"component": int(component), "row_group": index, "role": "test"}
            for index, component in enumerate(component_ids)
        ]
        cells: dict[tuple[int, str, float], dict] = {}
        old_predictions = []
        old_scores: dict[str, dict] = {}
        old_boot = []
        draws_by_seed: dict[int, dict[str, np.ndarray]] = {seed: {} for seed in scorer.e483.SEEDS}
        score_by_seed: dict[int, dict[str, float]] = {seed: {} for seed in scorer.e483.SEEDS}
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            with (
                patch.object(scorer.e483, "ROWS", rows),
                patch.object(scorer.e483, "CHUNK", 5),
            ):
                for seed, shape, beta in scorer.e483.ARMS:
                    torch.manual_seed(seed + round(beta * 100) + int(shape == "sine") * 10_000)
                    model = scorer.e483.IndependentMomentumModel(sine=shape == "sine")
                    checkpoint_name = f"{scorer.e483.arm_key(seed, shape, beta)}.pt"
                    checkpoint = run / checkpoint_name
                    torch.save(model.state_dict(), checkpoint)
                    cells[(seed, shape, beta)] = {
                        "checkpoint": checkpoint_name,
                        "checkpoint_sha256": scorer.sha256_file(checkpoint),
                    }
                    prediction = scorer.e483.predict_full(model, x, torch.device("cpu"))
                    old_predictions.append(prediction[selected].copy())
                    metric = scorer.e483.global_wp(targets, prediction, selected)
                    old_scores[scorer.e483.arm_key(seed, shape, beta)] = {
                        "wp": float(metric["wp"]),
                        "t0": float(metric["t0"]),
                        "t1": float(metric["t1"]),
                    }
                    moments = scorer.e483.component_moments(targets, prediction, selected)
                    counts = scorer.e483.bootstrap_multiplicities(component_ids)
                    arm_draws = np.asarray(
                        [scorer.e483.pool_component_moments(moments, draw) for draw in counts],
                        dtype=np.float64,
                    )
                    draws_by_seed[seed][f"{shape}_beta{beta}"] = arm_draws
                    score_by_seed[seed][f"{shape}_beta{beta}"] = float(metric["wp"])
                    old_boot.append(arm_draws)
                contrasts = {
                    str(seed): scorer.e483.registered_contrasts(
                        score_by_seed[seed], draws_by_seed[seed]
                    )
                    for seed in scorer.e483.SEEDS
                }
                reference = {
                    "arm_names": [
                        scorer.e483.arm_key(seed, shape, beta)
                        for seed, shape, beta in scorer.e483.ARMS
                    ],
                    "arrays": {
                        "predictions_raw_selected_by_arm": np.stack(old_predictions),
                        "arm_bootstrap_wp": np.stack(old_boot),
                    },
                    "scores": {
                        "arm_scores": old_scores,
                        "contrasts_by_seed": contrasts,
                    },
                }
                result, exports = scorer.run_crossed_matrix(
                    cells,
                    nodes,
                    x,
                    targets,
                    selected,
                    run,
                    reference,
                    torch.device("cpu"),
                )
            self.assertEqual(result["cell_count"], 36)
            self.assertEqual(result["contrast_count"], 54)
            self.assertEqual(len(exports["predictions_selected"]), 36)
            self.assertEqual(len(exports["contrast_draws"]), 54)
            self.assertTrue(
                all(
                    np.isfinite(cell["hidden_state_abs_max_all_layers_all_chunks"])
                    for cell in result["cells"].values()
                )
            )
            output = run / "synthetic_export"
            scorer.write_outputs(
                output,
                result,
                exports,
                targets,
                selected,
                selected,
                selected,
                nodes,
                {"test_rows_scored": "synthetic only"},
            )
            with np.load(output / "selected_exports.npz", allow_pickle=False) as artifact:
                self.assertEqual(artifact["cell_names"].shape, (36,))
                self.assertEqual(
                    artifact["predictions_raw_selected_by_cell"].shape, (36, 16 * rows, 2)
                )
                self.assertEqual(artifact["contrast_names"].shape, (54,))
                self.assertEqual(artifact["contrast_bootstrap_diffs"].shape, (54, 2_000))
            with self.assertRaises(FileExistsError):
                scorer.write_outputs(
                    output,
                    result,
                    exports,
                    targets,
                    selected,
                    selected,
                    selected,
                    nodes,
                    {},
                )


if __name__ == "__main__":
    unittest.main()
