"""Synthetic algebra, reset-local feature, pooling and report checks for R5."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

CAMPAIGN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAMPAIGN_DIR.parents[1] / "src"))
sys.path.insert(0, str(CAMPAIGN_DIR))

_fit = importlib.import_module("idea_screens_r5.analysis_fit")
_report = importlib.import_module("idea_screens_r5.analysis_report")
_score = importlib.import_module("idea_screens_r5.analysis_score")
_run_probe = importlib.import_module("idea_screens_r5.run_probe")
_gmetric = importlib.import_module("wnn.gmetric")
build_sequence_features = _fit.build_sequence_features
decode_moments = _fit.decode_moments
fit_moment_model = _fit.fit_moment_model
moment_targets = _fit.moment_targets
predict_moments = _fit.predict_moments
scored_stride_indices = _fit.scored_stride_indices
build_result = _report.build_result
save_model = _report.save_model
_score_role = _score._score_role
GlobalAccumulator = _gmetric.GlobalAccumulator
global_wp = _gmetric.global_wp


def _fit_synthetic_model() -> tuple[object, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(5)
    raw = generator.normal(size=(160, 112)).astype(np.float32)
    features = build_sequence_features(raw)
    labels = generator.normal(scale=0.8, size=(160, 2)).astype(np.float32)
    labels[labels == 0.0] = 0.01
    return fit_moment_model(features, labels), raw, labels


def _synthetic_sequence(component: int, role: str, raw: np.ndarray, labels: np.ndarray) -> dict:
    mask = np.zeros(len(raw), dtype=bool)
    mask[100:] = True
    if component % 11 == 0:
        mask[:] = False
    return {
        "role": role,
        "source": "train" if role == "fit" else "valid",
        "group": component + 10_000,
        "component": component,
        "x": raw,
        "y": labels,
        "scored": mask,
    }


class R5ProbeTest(unittest.TestCase):
    """Check the fixed moment algebra and the complete synthetic output path."""

    def test_constant_feature_decoder_is_abs_y_weighted_mean(self) -> None:
        targets = np.array([[-2.0, -1.5], [-0.5, 0.25], [0.75, 0.5], [1.25, 1.0]])
        fit = fit_moment_model(np.ones((len(targets), 3)), targets)
        moments = predict_moments(np.ones((2, 3)), fit)
        decoded = decode_moments(moments, fit.mu_b)
        clipped = np.clip(targets, -2.0, 2.0)
        expected = (np.abs(clipped) * clipped).sum(axis=0) / np.abs(clipped).sum(axis=0)
        self.assertTrue(np.allclose(decoded["conditional_ratio"], expected[None, :]))

    def test_moment_difference_is_signed_absolute_square_and_zero_mu_fails(self) -> None:
        targets = np.array([[-2.0, 0.25], [-0.5, -1.0], [0.75, 2.5]])
        clipped, columns = moment_targets(targets)
        moments = columns.reshape(len(targets), 2, 3)
        expected = np.abs(clipped) * clipped
        self.assertTrue(np.allclose(moments[:, :, 0] - moments[:, :, 1], expected))
        with self.assertRaisesRegex(ValueError, "positive fit mean"):
            fit_moment_model(np.ones((3, 2)), np.array([[0.0, 1.0], [0.0, -1.0], [0.0, 0.5]]))

    def test_decoder_pair_shares_numerator_and_applies_finite_floor(self) -> None:
        moments = np.array([[[0.2, 0.1, -1.0], [0.6, 0.2, 0.1]]])
        decoded = decode_moments(moments, np.array([0.4, 0.5]))
        self.assertTrue(np.allclose(decoded["D_hat"], [[0.1, 0.4]]))
        self.assertTrue(np.allclose(decoded["ratio_denominator"], [[0.02, 0.1]]))
        self.assertTrue(np.isfinite(decoded["conditional_ratio"]).all())
        self.assertTrue(
            np.allclose(
                decoded["conditional_ratio"], decoded["D_hat"] / decoded["ratio_denominator"]
            )
        )
        self.assertTrue(np.allclose(decoded["constant_denominator"], decoded["D_hat"] / [0.4, 0.5]))

    def test_feature_history_is_causal_reset_local_and_stride_is_post_history(self) -> None:
        generator = np.random.default_rng(12)
        raw = generator.normal(size=(128, 112)).astype(np.float32)
        changed_suffix = raw.copy()
        changed_suffix[80:] += 50.0
        first = build_sequence_features(raw)
        changed = build_sequence_features(changed_suffix)
        self.assertTrue(np.array_equal(first[:80], changed[:80]))
        left = np.ones((128, 112), dtype=np.float32)
        right = np.full((128, 112), 2.0, dtype=np.float32)
        right_reset = build_sequence_features(right)
        concatenated = build_sequence_features(np.concatenate((left, right)))
        self.assertTrue(np.all(right_reset[0, 112:224] == 0.0))
        self.assertTrue(np.any(concatenated[128, 112:224] != 0.0))
        scored = np.zeros(200, dtype=bool)
        scored[100], scored[120], scored[160] = True, True, True
        self.assertTrue(np.array_equal(scored_stride_indices(scored), [120, 160]))

    def test_sample_membership_metadata_gate_without_opening_arrays(self) -> None:
        fit_order = list(range(64))
        eval_order = list(range(64, 96))
        file_record = {"name": "unused.npy", "sha256": "a" * 64, "bytes": 12}
        records = []
        for role, source, components in (
            ("fit", "train", fit_order),
            ("eval", "valid", eval_order),
        ):
            for component in components:
                records.append(
                    {
                        "role": role,
                        "source": source,
                        "group": component + (10_000 if role == "fit" else 20_000),
                        "component": component,
                        "files": {name: dict(file_record) for name in ("x", "y", "scored")},
                    }
                )
        manifest = {
            "schema": "idea_screens_r3.sample.v2",
            "split_sha256": "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7",
            "selection": {
                "fit_full64_component_order": fit_order,
                "fit_low32_components": fit_order[:32],
                "eval32_component_order": eval_order,
            },
            "records": records,
        }
        self.assertEqual(_run_probe._validate_sample_manifest(manifest), (fit_order, eval_order))
        broken = json.loads(json.dumps(manifest))
        broken["records"][-1]["component"] = fit_order[0]
        with self.assertRaisesRegex(ValueError, "membership"):
            _run_probe._validate_sample_manifest(broken)

    def test_canonical_pooling_drop_one_and_zero_scored_group(self) -> None:
        fit, raw, labels = _fit_synthetic_model()
        sequences = [
            _synthetic_sequence(10, "eval", raw, labels),
            _synthetic_sequence(11, "eval", raw * 0.5, labels[::-1]),
            _synthetic_sequence(22, "eval", raw * 1.2, labels),
        ]
        result, _, _ = _score_role(sequences, fit, None)
        scored = result["by_decoder"]["conditional_ratio"]
        self.assertEqual(scored["component_moments"][-1]["scored_rows"], 0)
        self.assertIsNone(scored["component_moments"][-1]["official_metric"])
        self.assertEqual(len(scored["drop_one_component"]), 3)
        self.assertIsNotNone(scored["drop_one_component"][-1]["official_metric"])
        concatenated_y, concatenated_p, concatenated_mask = [], [], []
        for sequence in sequences:
            base = build_sequence_features(sequence["x"])
            moments = predict_moments(base, fit)
            prediction = decode_moments(moments, fit.mu_b)["conditional_ratio"]
            concatenated_y.append(sequence["y"])
            concatenated_p.append(prediction.astype(np.float32))
            concatenated_mask.append(sequence["scored"])
        expected = global_wp(
            np.concatenate(concatenated_y),
            np.concatenate(concatenated_p),
            np.concatenate(concatenated_mask),
        )
        actual = scored["official_metric"]
        for key in ("t0", "t1", "wp"):
            self.assertAlmostEqual(actual[key], expected[key], places=6)
        without = np.concatenate(
            (
                np.ones(len(raw), dtype=bool),
                np.ones(len(raw), dtype=bool),
                np.zeros(len(raw), dtype=bool),
            )
        )
        expected_drop = global_wp(
            np.concatenate(concatenated_y)[without],
            np.concatenate(concatenated_p)[without],
            np.concatenate(concatenated_mask)[without],
        )
        self.assertAlmostEqual(
            scored["drop_one_component"][-1]["official_metric"]["wp"], expected_drop["wp"], places=6
        )

    def test_end_to_end_synthetic_diagnostics_npz_and_report(self) -> None:
        fit, raw, labels = _fit_synthetic_model()
        fit_sequences = [_synthetic_sequence(component, "fit", raw, labels) for component in (1, 2)]
        eval_sequences = [
            _synthetic_sequence(component, "eval", raw, labels) for component in range(100, 132)
        ]
        fit_scores, fit_diagnostics, fit_rows = _score_role(fit_sequences, fit, None)
        eval_scores, eval_diagnostics, eval_rows = _score_role(eval_sequences, fit, None)
        scoring = {
            "fit": fit_scores,
            "eval": eval_scores,
            "mask_rows_by_component": {"fit": fit_rows, "eval": eval_rows},
        }
        diagnostics = {"fit": fit_diagnostics, "eval": eval_diagnostics}
        reference_endpoint = eval_scores["by_decoder"]["constant_denominator"]
        reference = {
            "sample_manifest_sha256": "b" * 64,
            "cases": {
                "full64_raw_signed_ema_hl32": {
                    "selected_alpha": 100,
                    "outer": {
                        "selected": {
                            "official_metric": reference_endpoint["official_metric"],
                            "component_moments": [
                                {
                                    "component": item["component"],
                                    "scored_rows": item["scored_rows"],
                                    "official_metric": item["official_metric"],
                                    "additive_raw_sums_W_WY_WP_WYY_WPP_WYP": item[
                                        "additive_raw_sums_W_WY_WP_WYY_WPP_WYP"
                                    ],
                                }
                                for item in reference_endpoint["component_moments"]
                            ],
                        }
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model_archive = save_model(output / "moment_model.npz", fit)
            with np.load(output / "moment_model.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["moment_coefficients"].shape, (224, 6))
                self.assertTrue(np.isfinite(archive["epsilon"]).all())
            fit_counts = {"fit_components": 2, "fit_sampled_rows": 12, "fit_scored_rows": 120}
            result = build_result(
                output,
                fit,
                fit_counts,
                scoring,
                diagnostics,
                reference,
                "a" * 64,
                "b" * 64,
                "c" * 64,
                model_archive,
                {"total_wall_s": 0.1, "total_process_cpu_s": 0.05},
            )
            loaded = json.loads((output / "RESULT.json").read_text())
            self.assertEqual(loaded["schema"], result["schema"])
            self.assertEqual(loaded["fit"]["fit_calls"], 1)
            self.assertEqual(loaded["fit"]["target_columns"], 6)
            self.assertEqual(loaded["fit"]["decoder_count"], 2)
            self.assertNotIn("normalizer_mean", loaded["fit"])
            self.assertNotIn("moment_coefficients", loaded["fit"])
            self.assertEqual(
                loaded["fit"]["model_parameter_metadata"]["moment_coefficients"]["shape"],
                [224, 6],
            )
            self.assertEqual(
                loaded["diagnostics"]["eval"]["pooled"]["t0"]["B_hat_over_muB"]["count"], 30 * 60
            )
            self.assertTrue((output / "DONE.json").is_file())
            self.assertEqual(
                _report.digest(output / "RESULT.json"),
                json.loads((output / "DONE.json").read_text())["result_sha256"],
            )
            done = json.loads((output / "DONE.json").read_text())
            self.assertEqual(done["result_bytes"], (output / "RESULT.json").stat().st_size)
            self.assertEqual(done["model_archive"], model_archive)


if __name__ == "__main__":
    unittest.main()
