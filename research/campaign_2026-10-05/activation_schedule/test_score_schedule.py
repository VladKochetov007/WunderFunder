"""Bounded synthetic checks for the independent E476 scorer."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import score_schedule as scorer


class ScoreScheduleTests(unittest.TestCase):
    def test_sealed_core_and_declared_runtime_artifacts(self) -> None:
        spec = json.loads(scorer.SEALED_FIT_SPEC.read_text())
        checked = scorer.validate_sealed_spec_core(spec)
        self.assertEqual(tuple(checked["artifacts"]), scorer.EXPECTED_SEALED_ARTIFACTS)
        self.assertEqual(set(spec), set(checked))

        invalid = copy.deepcopy(spec)
        invalid["runtime_sha256"] = {"runner.py": "0" * 64}
        with self.assertRaises(ValueError):
            scorer.validate_sealed_spec_core(invalid)

        run = scorer.DEFAULT_RUN
        manifest_path, trace_path = [scorer.ROOT / path for path in spec["artifacts"]]
        self.assertEqual(manifest_path, run / "fit_manifest.json")
        self.assertEqual(trace_path, run / "lr_traces.json")
        report = json.loads(manifest_path.read_text())
        self.assertEqual(set(report["source_sha256"]), scorer.EXPECTED_MANIFEST_SOURCES)
        sealed_content = {entry["path"]: entry["sha256"] for entry in spec["code"] + spec["inputs"]}
        self.assertEqual(
            report["source_sha256"],
            {key: sealed_content[key] for key in report["source_sha256"]},
        )
        traces = scorer.validate_lr_traces(run, report)
        self.assertEqual(set(traces), set(scorer.SCHEDULES))
        self.assertTrue(
            np.isclose(traces["warmcos_47_427"].sum(), traces["constant_mean_warmcos"].sum())
        )
        self.assertTrue(trace_path.is_file())

    def test_synthetic_runner_attempt_receipt_is_fail_closed(self) -> None:
        spec = json.loads(scorer.SEALED_FIT_SPEC.read_text())
        source_record = json.loads(scorer.DEFAULT_ATTEMPT_RECORD.read_text())
        with tempfile.TemporaryDirectory(prefix="e476-attempt-test-") as directory:
            attempt_path = Path(directory) / "attempt.json"

            def validate_fake(record: dict) -> dict:
                attempt_path.write_text(json.dumps(record))
                return scorer.validate_attempt_record(attempt_path, spec, scorer.DEFAULT_RUN)

            valid = copy.deepcopy(source_record)
            valid_receipt = validate_fake(valid)
            self.assertEqual(valid_receipt["attempt_id"], "e476-activation-twelve-fit-v1")
            self.assertTrue(valid_receipt["worker_oom_counters_unchanged"])
            self.assertTrue(valid_receipt["parent_oom_counters_unchanged"])
            self.assertGreater(valid["initial_parent_memory_events"]["oom"], 0)

            invalid_records = []
            changed_spec = copy.deepcopy(valid)
            changed_spec["spec"]["version"] = 2
            invalid_records.append(changed_spec)
            changed_runtime = copy.deepcopy(valid)
            changed_runtime["runtime_sha256"].pop(next(iter(changed_runtime["runtime_sha256"])))
            invalid_records.append(changed_runtime)
            failed = copy.deepcopy(valid)
            failed.update(status="failed", exit_code=1)
            invalid_records.append(failed)
            signaled = copy.deepcopy(valid)
            signaled["signals"] = [15]
            invalid_records.append(signaled)
            unverified = copy.deepcopy(valid)
            unverified["teardown_verified"] = False
            invalid_records.append(unverified)
            worker_oom = copy.deepcopy(valid)
            worker_oom["final_memory_events"]["oom_kill"] += 1
            invalid_records.append(worker_oom)
            parent_oom = copy.deepcopy(valid)
            parent_oom["final_parent_memory_events"]["oom_group_kill"] += 1
            invalid_records.append(parent_oom)
            wrong_artifacts = copy.deepcopy(valid)
            wrong_artifacts["artifacts"].pop()
            invalid_records.append(wrong_artifacts)
            for record in invalid_records:
                with self.subTest(record=record), self.assertRaises(ValueError):
                    validate_fake(record)

    def test_twenty_pair_effects_and_joint_interactions(self) -> None:
        rng = np.random.default_rng(20261005)
        n_components, n_rows = 16, 48
        targets = rng.normal(0.0, 0.65, size=(n_components, n_rows, 2)).astype(np.float32)
        mask = rng.random((n_components, n_rows)) > 0.17
        mask[:, 0] = True
        components = np.asarray([f"component-{index:02d}" for index in range(n_components)])

        predictions: dict[str, np.ndarray] = {}
        official_scores: dict[str, float] = {}
        moments: dict[str, np.ndarray] = {}
        for arm_index, (activation, optimizer, schedule) in enumerate(scorer.ARMS):
            name = "__".join((activation, optimizer, schedule))
            activation_shift = 0.08 if activation == "sine_embed" else 0.0
            optimizer_shift = 0.04 if optimizer == "muon_adamw" else 0.0
            schedule_shift = {
                "constant_peak": 0.0,
                "warmcos_47_427": 0.07,
                "constant_mean_warmcos": -0.03,
            }[schedule]
            scale = 0.72 + activation_shift + optimizer_shift + schedule_shift
            noise_scale = 0.31 + 0.002 * arm_index
            prediction = scale * targets + noise_scale * rng.normal(size=targets.shape).astype(
                np.float32
            )
            predictions[name] = prediction
            official_scores[name] = float(scorer.global_wp(targets, prediction, mask)["wp"])
            moments[name], _ = scorer.paired_component_moments(
                targets, prediction, prediction, mask
            )
            pooled = scorer.pool_component_moments(
                moments[name], np.ones(n_components, dtype=np.int64)
            )
            self.assertAlmostEqual(pooled, official_scores[name], places=7)

        self.assertEqual(len(scorer.PAIR_CONTRASTS), 20)
        scorer.BOOTSTRAP_COUNT = 23
        for candidate, reference, _family in scorer.PAIR_CONTRASTS:
            forward = scorer.paired_component_bootstrap(
                moments[candidate],
                moments[reference],
                components,
                n_boot=scorer.BOOTSTRAP_COUNT,
                seed=scorer.BOOTSTRAP_SEED,
                centered=True,
            )
            reverse = scorer.paired_component_bootstrap(
                moments[reference],
                moments[candidate],
                components,
                n_boot=scorer.BOOTSTRAP_COUNT,
                seed=scorer.BOOTSTRAP_SEED,
                centered=True,
            )
            self.assertAlmostEqual(forward["candidate_wp"], official_scores[candidate], places=7)
            self.assertAlmostEqual(forward["reference_wp"], official_scores[reference], places=7)
            self.assertAlmostEqual(forward["mean_diff"], -reverse["mean_diff"], places=12)

        for optimizer in scorer.OPTIMIZERS:
            keys = (
                f"sine_embed__{optimizer}__warmcos_47_427",
                f"base__{optimizer}__warmcos_47_427",
                f"sine_embed__{optimizer}__constant_mean_warmcos",
                f"base__{optimizer}__constant_mean_warmcos",
            )
            effect = scorer.interaction_bootstrap(moments, components, keys)
            point = (official_scores[keys[0]] - official_scores[keys[1]]) - (
                official_scores[keys[2]] - official_scores[keys[3]]
            )
            self.assertAlmostEqual(effect["mean_diff"], point, places=7)

            # Independently recompute the same joint component-count draws through
            # official global_wp on explicit repeated rows.
            order = np.argsort(components, kind="stable")
            local_rng = np.random.default_rng(scorer.BOOTSTRAP_SEED)
            explicit = np.empty(scorer.BOOTSTRAP_COUNT, dtype=np.float64)
            for draw_index in range(scorer.BOOTSTRAP_COUNT):
                selected = local_rng.integers(0, n_components, n_components)
                counts = np.bincount(selected, minlength=n_components)
                arm_wps = []
                for key in keys:
                    row_ids = np.repeat(np.arange(n_components), counts)
                    sorted_targets = targets[order]
                    sorted_predictions = predictions[key][order]
                    y = sorted_targets[row_ids]
                    p = sorted_predictions[row_ids]
                    selected_mask = mask[order][row_ids]
                    arm_wps.append(float(scorer.global_wp(y, p, selected_mask)["wp"]))
                explicit[draw_index] = (arm_wps[0] - arm_wps[1]) - (arm_wps[2] - arm_wps[3])
            np.testing.assert_allclose(
                effect["bootstrap_sd"], explicit.std(ddof=1), atol=1e-10, rtol=0
            )
            np.testing.assert_allclose(
                effect["ci95"], np.percentile(explicit, [2.5, 97.5]), atol=1e-8, rtol=0
            )

        same_name = "base__adamw__constant_peak"
        zero_pair = scorer.paired_component_bootstrap(
            moments[same_name],
            moments[same_name],
            components,
            n_boot=scorer.BOOTSTRAP_COUNT,
            seed=scorer.BOOTSTRAP_SEED,
            centered=True,
        )
        self.assertEqual(zero_pair["mean_diff"], 0.0)
        np.testing.assert_array_equal(zero_pair["ci95"], [0.0, 0.0])
        zero_interaction = scorer.interaction_bootstrap(
            moments, components, (same_name, same_name, same_name, same_name)
        )
        self.assertEqual(zero_interaction["mean_diff"], 0.0)
        np.testing.assert_array_equal(zero_interaction["ci95"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
