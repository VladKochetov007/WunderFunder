"""Small synthetic checks for the fixed R4 feature and metric invariants."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CAMPAIGN_DIR = Path(__file__).resolve().parent.parent
CANONICAL_SRC = CAMPAIGN_DIR.parents[1] / "src"
sys.path.insert(0, str(CANONICAL_SRC))
sys.path.insert(0, str(CAMPAIGN_DIR))

_probe_helpers = importlib.import_module("idea_screens_r3.probe_helpers")
_ridge_math = importlib.import_module("idea_screens_r2.ridge_math")
_analysis_fit = importlib.import_module("idea_screens_r4.analysis_fit")
_analysis_report = importlib.import_module("idea_screens_r4.analysis_report")
_analysis_score = importlib.import_module("idea_screens_r4.analysis_score")
_random_features = importlib.import_module("idea_screens_r4.random_features")
_gmetric = importlib.import_module("wnn.gmetric")
metric_without = _probe_helpers.metric_without
safe_result = _probe_helpers.safe_result
RidgeModel = _ridge_math.RidgeModel
CASE_BASE = _analysis_fit.CASE_BASE
CASE_NAMES = _analysis_fit.CASE_NAMES
build_result = _analysis_report.build_result
digest = _analysis_report.digest
accumulate_component = _analysis_score.accumulate_component
augment_pair = _random_features.augment_pair
build_bases = _random_features.build_bases
draw_projection = _random_features.draw_projection
fit_base_normalizer = _random_features.fit_base_normalizer
GlobalAccumulator = _gmetric.GlobalAccumulator
global_wp = _gmetric.global_wp


class R4ProbeTest(unittest.TestCase):
    """Exercise fold-only transforms, paired projections and canonical pooling."""

    def test_base_normalizer_uses_training_rows_and_handles_constant_columns(self) -> None:
        training = np.array([[0.0, 4.0, 9.0], [2.0, 4.0, 11.0]])
        held_extreme = np.array([[1_000.0, -1_000.0, 1_000.0]])
        normalizer = fit_base_normalizer(training)
        self.assertTrue(np.array_equal(normalizer.mean, [1.0, 4.0, 10.0]))
        self.assertTrue(np.array_equal(normalizer.scale, [1.0, 1.0, 1.0]))
        self.assertFalse(
            np.array_equal(normalizer.mean, np.concatenate((training, held_extreme)).mean(axis=0))
        )

    def test_identity_and_tanh_share_fixed_projection_but_differ_on_hidden_block(self) -> None:
        rows = np.array([[-1.0, 0.0], [0.5, 1.0], [2.0, -0.5]])
        normalizer = fit_base_normalizer(rows[:2])
        draw = draw_projection(2, 19_031)
        repeated = draw_projection(2, 19_031)
        self.assertTrue(np.array_equal(draw.weights, repeated.weights))
        self.assertTrue(np.array_equal(draw.bias, repeated.bias))
        identity, nonlinear = augment_pair(rows, normalizer, draw)
        hidden = ((rows - normalizer.mean) / normalizer.scale) @ draw.weights + draw.bias
        self.assertTrue(np.array_equal(identity[:, :2], rows))
        self.assertTrue(np.allclose(identity[:, 2:], hidden))
        self.assertTrue(np.allclose(nonlinear[:, 2:], np.tanh(hidden)))
        self.assertFalse(np.allclose(identity[:, 2:], nonlinear[:, 2:]))

    def test_signed_history_is_causal_to_future_suffix(self) -> None:
        generator = np.random.default_rng(41)
        raw = generator.normal(size=(16, 112)).astype(np.float32)
        changed = raw.copy()
        changed[9:] += 100.0
        original_base = build_bases(raw)["raw_signed224"]
        changed_base = build_bases(changed)["raw_signed224"]
        self.assertTrue(np.array_equal(original_base[:9], changed_base[:9]))

    def test_component_pooling_matches_official_global_metric_and_drop_one(self) -> None:
        targets = np.array(
            [
                [[-1.2, 0.2], [-0.1, 1.3], [0.4, -0.5], [1.6, -1.4]],
                [[0.9, 0.4], [-1.1, 0.7], [0.2, 1.4], [-0.6, -0.8]],
            ]
        )
        predictions = targets * 0.73 + np.array([0.07, -0.12])
        mask = np.ones((2, 4), dtype=bool)
        total = GlobalAccumulator()
        by_component = {10: GlobalAccumulator(), 11: GlobalAccumulator()}
        for index, component in enumerate((10, 11)):
            accumulate_component(
                total,
                by_component,
                component,
                targets[index : index + 1],
                predictions[index : index + 1],
                mask[index : index + 1],
            )
        pooled = safe_result(total)
        expected = global_wp(targets, predictions, mask)
        for key in ("t0", "t1", "wp"):
            self.assertAlmostEqual(pooled[key], expected[key])
        self.assertEqual(pooled["selected_rows"], expected["selected_rows"])
        without_first = metric_without(by_component, 10)
        expected_without_first = global_wp(targets[1:], predictions[1:], mask[1:])
        for key in ("t0", "t1", "wp"):
            self.assertAlmostEqual(without_first[key], expected_without_first[key])
        self.assertEqual(without_first["selected_rows"], expected_without_first["selected_rows"])

    def test_synthetic_report_builds_all_pairs_reference_and_receipts(self) -> None:
        sample_sha = "b89455ba449f31bb9ebe815976d4da23bb1a8e499e421b6bb5f8d44566f76e64"
        reference_sha = "a7e19f6358fbd1f61496efd7c1712b80cb7d582f22445a1e3c12988851b323dc"
        source_sha = "1" * 64
        components = list(range(32))
        outer = {}
        for case_index, case in enumerate(CASE_NAMES):
            outer[case] = {}
            for label_index, label in enumerate(("selected", "alpha1_baseline")):
                score = 0.1 + case_index * 0.01 + label_index * 0.001
                metric = {"t0": score, "t1": score + 0.02, "wp": score + 0.01}
                drops = [
                    {
                        "omitted_component": component,
                        "official_metric": {
                            "t0": score - component * 1e-5,
                            "t1": score + 0.02 - component * 1e-5,
                            "wp": score + 0.01 - component * 1e-5,
                        },
                        "scored_rows": 59_436 - 1,
                        "delta_wp_from_full_fixed_predictions": -component * 1e-5,
                    }
                    for component in components
                ]
                outer[case][label] = {
                    "official_metric": metric,
                    "component_moments": [],
                    "drop_one_component": drops,
                }
        reference = {
            "schema": "bounded_r3_component_factorial_nested_ridge_v1",
            "sample_manifest_sha256": sample_sha,
            "cases": {
                case: {
                    "outer": {
                        label: {"official_metric": {"t0": 0.1, "t1": 0.2, "wp": 0.15}}
                        for label in ("selected", "alpha1_baseline")
                    }
                }
                for case in ("full64_raw112", "full64_raw_signed_ema_hl32")
            },
        }
        selected = {case: 100 for case in CASE_NAMES}
        models, draws, normalizers, projection_stats, archives = {}, {}, {}, {}, {}
        for base, dimension, seed in (("raw112", 112, 19_031), ("raw_signed224", 224, 19_032)):
            draws[base] = draw_projection(dimension, seed)
            normalizers[base] = fit_base_normalizer(np.ones((2, dimension)))
            projection_stats[base] = {"input_dimension": dimension, "width": 64, "seed": seed}
        for case in CASE_NAMES:
            dimension = (112 if CASE_BASE[case] == "raw112" else 224) + 64
            model = RidgeModel(
                np.zeros(dimension),
                np.ones(dimension),
                np.zeros((dimension, 2)),
                np.zeros(2),
                (0.0, 0.0),
            )
            models[case] = {"selected": model, "alpha1_baseline": model}
            archives[case] = {"name": f"{case}.npz", "bytes": 1, "sha256": "2" * 64}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = build_result(
                output,
                sample_sha,
                "3" * 64,
                reference_sha,
                source_sha,
                {"fit_full64_component_order": list(range(64))},
                {"fit_sampled_rows": 31_808, "fit_scored_rows": 1_273_664},
                {case: {} for case in CASE_NAMES},
                selected,
                {
                    "budget_folds": [list(range(22)), list(range(22, 43)), list(range(43, 64))],
                    "inner_alpha_models": 36,
                    "inner_two_target_solves": 72,
                },
                models,
                {case: 0.01 for case in CASE_NAMES},
                archives,
                draws,
                normalizers,
                projection_stats,
                outer,
                {"outer_scored_rows": 59_436, "outer_sequences": 32, "outer_components": 32},
                2.0,
                1.0,
                1024,
                reference,
            )
            saved = json.loads((output / "RESULT.json").read_bytes())
            done = json.loads((output / "DONE.json").read_bytes())
            self.assertEqual(result, saved)
            self.assertEqual(done["result_sha256"], digest(output / "RESULT.json"))
            self.assertEqual(
                len(
                    saved["primary_tanh_minus_identity_contrasts"]["raw112"]["selected"][
                        "drop_one_fixed_prediction_tanh_minus_identity"
                    ]
                ),
                32,
            )
            self.assertAlmostEqual(
                saved["secondary_r3_linear_reference_contrasts"]["bases"]["raw_signed224"][
                    "raw_signed224_tanh64"
                ]["selected"]["r4_minus_r3_linear_t0_t1_wp"]["wp"],
                -0.01,
            )


if __name__ == "__main__":
    unittest.main()
