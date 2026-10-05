"""Synthetic tests for independent E488 raw-export reproduction."""

from __future__ import annotations

import copy
import unittest

import numpy as np
import reproduce_duration_exports as repro


class ArchiveFixture:
    """Minimal in-memory NPZ view for small synthetic score matrices."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays
        self.files = list(arrays)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]


def _direct_target_score(y: np.ndarray, p: np.ndarray) -> float:
    """Compute one clipped weighted Pearson directly for a reference oracle."""
    y = np.clip(np.asarray(y, dtype=np.float32), -2, 2).astype(np.float64)
    p = np.clip(np.asarray(p, dtype=np.float32), -2, 2).astype(np.float64)
    weight = np.abs(y)
    total = float(weight.sum())
    if total < 1e-8:
        return 0.0
    mean_y = float((weight * y).sum() / total)
    mean_p = float((weight * p).sum() / total)
    centered_y, centered_p = y - mean_y, p - mean_p
    var_y = float((weight * centered_y**2).sum() / total)
    var_p = float((weight * centered_p**2).sum() / total)
    if np.sqrt(var_y) <= 1e-8 or np.sqrt(var_p) <= 1e-8:
        return 0.0
    covariance = float((weight * centered_y * centered_p).sum() / total)
    return float(np.clip(covariance / np.sqrt(var_y * var_p), -1, 1))


def _small_fixture() -> tuple[dict, ArchiveFixture]:
    """Build all 24 snapshots and 27 exact registered summaries on tiny arrays."""
    rng = np.random.default_rng(8127)
    rows = 9
    targets = rng.normal(0, 0.7, size=(repro.COMPONENTS, rows, 2)).astype(np.float32)
    targets[0, 0, 0] = 3.5
    mask = np.ones((repro.COMPONENTS, rows), dtype=bool)
    mask[:, 2] = False
    counts = mask.sum(axis=1).astype(np.int64)
    component_ids = 500 + (np.arange(repro.COMPONENTS) * 7 % repro.COMPONENTS)
    row_groups = np.arange(1300, 1300 + repro.COMPONENTS, dtype=np.int64)
    sorted_ids = np.sort(component_ids, kind="stable")
    order = np.argsort(component_ids, kind="stable")
    draw_indices, multiplicities = repro.component_draws()
    arrays = {
        "component_ids": component_ids,
        "sorted_component_ids": sorted_ids,
        "row_groups": row_groups,
        "raw_targets": targets,
        "selected_mask": mask,
        "bootstrap_component_indices": draw_indices,
        "bootstrap_multiplicities": multiplicities,
    }
    arm_scores: dict[str, dict] = {}
    point_by_arm: dict[str, np.ndarray] = {}
    draws_by_arm: dict[str, np.ndarray] = {}
    arm_keys = sorted(repro.expected_arm_keys())
    for key in arm_keys:
        seed_label, arm, epoch_label = key.split("__")
        seed = int(seed_label.removeprefix("seed"))
        epoch = int(epoch_label.removeprefix("epoch"))
        seed_i = repro.SEEDS.index(seed)
        arm_i = repro.ARMS.index(arm)
        epoch_i = repro.EPOCHS.index(epoch)
        noise = rng.normal(size=targets.shape).astype(np.float32)
        prediction = (
            (0.45 + 0.07 * arm_i + 0.03 * epoch_i) * targets
            + (0.08 + 0.015 * seed_i + 0.01 * arm_i) * noise
            + 0.02 * arm_i
        ).astype(np.float32)
        arrays[f"prediction__{key}"] = prediction
        stats = repro.component_statistics(targets, prediction, mask, counts)[order]
        point_pair = repro.scores(stats, np.ones((1, repro.COMPONENTS), dtype=np.int64))[0]
        target_bootstrap = repro.scores(stats, multiplicities)
        point_by_arm[key] = point_pair
        draws_by_arm[key] = target_bootstrap
        arm_scores[key] = {
            "seed": seed,
            "arm": arm,
            "epoch": epoch,
            "parameters": 61_122 if arm in ("static_mix", "recurrent_mix") else 61_058,
            "updates": epoch * 474,
            "wp": float(point_pair.mean()),
            "t0": float(point_pair[0]),
            "t1": float(point_pair[1]),
            "metric_id": "global_wp_clip2_v1",
            "selected_rows": int(counts.sum()),
            "component_row_counts": counts.tolist(),
        }

    global_contrasts: dict[str, dict] = {}
    target_contrasts: dict[str, dict] = {}
    for key, definition in repro.expected_contrast_records().items():
        terms = definition["terms"]
        point_pair = sum(sign * point_by_arm[arm_key] for arm_key, sign in terms)
        sampled = sum(sign * draws_by_arm[arm_key] for arm_key, sign in terms)
        global_point = float(point_pair.mean())
        global_draws = sampled.mean(axis=1)
        global_contrasts[key] = {
            "seed": definition["seed"],
            "epoch": definition["epoch"],
            "contrast": definition["contrast"],
            "candidate": definition["candidate"],
            "reference": definition["reference"],
            **repro.summary(global_point, global_draws, definition["seed"], None),
        }
        target_contrasts[key] = {
            "seed": definition["seed"],
            "epoch": definition["epoch"],
            "contrast": definition["contrast"],
            "targets": {
                name: repro.summary(
                    float(point_pair[index]), sampled[:, index], definition["seed"], name
                )
                for index, name in enumerate(("t0", "t1"))
            },
        }
    result = {
        "kind": "independent E488 duration evaluation",
        "metric_id": "global_wp_clip2_v1",
        "trajectory_count": 12,
        "snapshot_count": 24,
        "registered_global_contrast_count": 27,
        "registered_target_contrast_count": 27,
        "bootstrap": {
            "draws": repro.BOOTSTRAP_COUNT,
            "seed": repro.BOOTSTRAP_SEED,
            "multiplicities_shared_across_all_24_snapshots_and_contrasts": True,
            "unit": "numeric-sorted fixed component ID",
        },
        "no_seed_pooling": True,
        "test_labels_used_for_fitting_or_selection": False,
        "component_ids": component_ids.tolist(),
        "sorted_component_ids": sorted_ids.tolist(),
        "row_groups": row_groups.tolist(),
        "component_row_counts": counts.tolist(),
        "selected_rows_total": int(counts.sum()),
        "arm_scores": arm_scores,
        "registered_global_contrasts": global_contrasts,
        "registered_target_contrasts": target_contrasts,
    }
    return result, ArchiveFixture(arrays)


class ReproduceDurationExportsTests(unittest.TestCase):
    """Check shifted-sum math, exact contrast mapping, and fail-closed inventories."""

    def test_shifted_sums_match_direct_weighted_pearson_oracle(self):
        rng = np.random.default_rng(19)
        targets = rng.normal(size=(repro.COMPONENTS, 13, 2)).astype(np.float32)
        predictions = (0.35 * targets + rng.normal(0, 0.2, targets.shape)).astype(np.float32)
        targets[0, 0, 0] = 4.0
        predictions[-1, -1, 1] = -3.0
        selected = rng.uniform(size=(repro.COMPONENTS, 13)) > 0.2
        counts = selected.sum(axis=1).astype(np.int64)
        stats = repro.component_statistics(targets, predictions, selected, counts)
        result = repro.scores(stats, np.ones((1, repro.COMPONENTS), dtype=np.int64))[0]
        selected_y = targets[selected]
        selected_p = predictions[selected]
        expected = np.asarray(
            [_direct_target_score(selected_y[:, index], selected_p[:, index]) for index in range(2)]
        )
        self.assertTrue(np.allclose(result, expected, atol=1e-12, rtol=0))

    def test_registered_duration_interaction_has_fixed_four_term_signs(self):
        records = repro.expected_contrast_records()
        for seed in repro.SEEDS:
            record = records[f"seed{seed}__duration_interaction"]
            self.assertEqual(
                record["terms"],
                (
                    (f"seed{seed}__recurrent_mix__epoch3", 1),
                    (f"seed{seed}__static_mix__epoch3", -1),
                    (f"seed{seed}__recurrent_mix__epoch1", -1),
                    (f"seed{seed}__static_mix__epoch1", 1),
                ),
            )
        self.assertEqual(len(records), 27)

    def test_all_24_snapshots_27_global_and_54_target_summaries_reproduce(self):
        result, archive = _small_fixture()
        receipt = repro.verify_payload(result, archive, expected_rows=9)
        self.assertEqual(receipt["arm_snapshots"], 24)
        self.assertEqual(receipt["global_contrasts"], 27)
        self.assertEqual(receipt["target_contrasts"], 54)
        self.assertEqual(receipt["bootstrap_draws_per_summary"], 2_000)
        self.assertEqual(receipt["bootstrap_draw_vectors_recomputed"]["arm_global_scores"], 24)
        self.assertEqual(receipt["bootstrap_draw_vectors_recomputed"]["arm_target_scores"], 48)
        self.assertEqual(receipt["bootstrap_draw_vectors_recomputed"]["global_contrast_arrays"], 27)
        self.assertEqual(receipt["bootstrap_draw_vectors_recomputed"]["target_contrast_scores"], 54)
        self.assertEqual(set(receipt["arm_score_draw_sha256"]), set(repro.expected_arm_keys()))
        self.assertEqual(
            set(receipt["arm_score_draw_sha256"][next(iter(repro.expected_arm_keys()))]),
            {"wp", "t0", "t1"},
        )

    def test_shared_draws_and_summary_use_all_2000_samples(self):
        draw_indices, counts = repro.component_draws()
        self.assertEqual(draw_indices.shape, (2_000, 16))
        self.assertEqual(counts.shape, (2_000, 16))
        self.assertTrue(np.array_equal(counts.sum(axis=1), np.full(2_000, 16)))
        values = np.linspace(-1, 1, 2_000)
        observed = repro.summary(0.25, values, repro.SEEDS[0], "t1")
        self.assertAlmostEqual(observed["bootstrap_sd"], float(values.std(ddof=1)))
        self.assertTrue(np.array_equal(observed["ci95"], np.percentile(values, [2.5, 97.5])))
        self.assertEqual(observed["p_le_0"], float((values <= 0).mean()))

    def test_rejects_dropped_contrast_and_modified_duration_difference(self):
        result, archive = _small_fixture()
        altered = copy.deepcopy(result)
        altered["registered_global_contrasts"].pop(f"seed{repro.SEEDS[0]}__duration_interaction")
        with self.assertRaisesRegex(ValueError, "global contrast inventory"):
            repro.verify_payload(altered, archive, expected_rows=9)
        altered, archive = _small_fixture()
        interaction = altered["registered_global_contrasts"][
            f"seed{repro.SEEDS[0]}__duration_interaction"
        ]
        interaction["point_diff"] += 0.1
        with self.assertRaisesRegex(ValueError, "global point"):
            repro.verify_payload(altered, archive, expected_rows=9)

    def test_rejects_nonfinite_export_values(self):
        result, archive = _small_fixture()
        arrays = {key: value.copy() for key, value in archive.arrays.items()}
        prediction_key = "prediction__seed271839__gelu__epoch1"
        arrays[prediction_key][0, 0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "raw export is nonfinite"):
            repro.verify_payload(result, ArchiveFixture(arrays), expected_rows=9)

    def test_rejects_bootstrap_metadata_or_duplicate_row_groups(self):
        result, archive = _small_fixture()
        altered = copy.deepcopy(result)
        altered["bootstrap"]["unit"] = "rows"
        with self.assertRaisesRegex(ValueError, "resampling unit"):
            repro.verify_payload(altered, archive, expected_rows=9)
        arrays = {key: value.copy() for key, value in archive.arrays.items()}
        arrays["row_groups"][-1] = arrays["row_groups"][0]
        with self.assertRaisesRegex(ValueError, "row groups are not unique"):
            repro.verify_payload(result, ArchiveFixture(arrays), expected_rows=9)


if __name__ == "__main__":
    unittest.main()
