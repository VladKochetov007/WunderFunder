"""Small synthetic checks for immutable grader gates and canonical scoring."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from online_relationship_grade_contract import row_refs_sha256, validate_recipe
from online_relationship_grade_core import (
    npy_coverage,
    official_scores,
    paired_component_contrast,
    paired_endpoint_moments,
    prediction_memmap,
    sha256_file,
)
from online_relationship_grade_parent import verify_parent_prediction_metadata


class GraderContractTests(unittest.TestCase):
    def test_runner_execution_status_is_accepted_without_target_access(self) -> None:
        root = Path(__file__).resolve().parents[3]
        template_path = (
            root / "research/experiment_drafts/online-relationships-20261022/recipe.json"
        )
        recipe = json.loads(template_path.read_text(encoding="utf-8"))
        recipe["status"] = "admitted_for_execution"
        recipe["admitted_new_fits"] = 12
        recipe["target_cache"]["path"] = str(root / "missing-target-cache-not-opened.npz")
        with tempfile.TemporaryDirectory() as directory:
            recipe_path = Path(directory) / "execution-recipe.json"
            recipe_path.write_text(json.dumps(recipe), encoding="utf-8")
            loaded, _ = validate_recipe(recipe_path, root, sha256_file(recipe_path))
        self.assertEqual(loaded["status"], "admitted_for_execution")

    def test_npy_header_must_cover_exact_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.npy"
            np.save(path, np.zeros((2, 20000, 2), dtype="<f4"), allow_pickle=False)
            coverage = npy_coverage(path)
            self.assertEqual(coverage["shape"], [2, 20000, 2])
            prediction, _ = prediction_memmap(path, (2, 20000, 2), sha256_file(path))
            self.assertEqual(prediction.dtype.str, "<f4")
            with path.open("ab") as stream:
                stream.write(b"x")
            with self.assertRaisesRegex(ValueError, "file length"):
                npy_coverage(path)

    def test_source_qualified_membership_digest_is_ordered(self) -> None:
        first = [
            {"split": "train", "row_group": 3},
            {"split": "valid", "row_group": 7},
        ]
        self.assertEqual(row_refs_sha256(first), row_refs_sha256(first.copy()))
        self.assertNotEqual(row_refs_sha256(first), row_refs_sha256(list(reversed(first))))
        changed = [dict(first[0]), {"split": "train", "row_group": 7}]
        self.assertNotEqual(row_refs_sha256(first), row_refs_sha256(changed))

    def test_deferred_recipe_is_rejected_before_artifact_or_target_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = root / "recipe.json"
            recipe_path.write_text(
                json.dumps(
                    {
                        "schema": "wnn.online-relationships.v1",
                        "experiment_id": "E-ONLINE-RELATIONSHIPS-20261022-R1",
                        "status": "deferred_entire_panel_after_feature_cost_profile",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not root-admitted"):
                validate_recipe(recipe_path, root, sha256_file(recipe_path))

    def test_parent_sidecar_matches_delivery_schema_and_parent_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(__file__).resolve().parents[3]
            scratch = Path(directory)
            recipe_path = (
                root / "research/experiment_drafts/online-relationships-20261022/recipe.json"
            )
            eval_meta_path = scratch / "evaluation_sequences.json"
            eval_meta_path.write_text("{}\n", encoding="utf-8")
            prediction_path = scratch / "evaluation_predictions.npy"
            prediction_meta_path = scratch / "evaluation_predictions.json"
            recipe = {"recipe_id": "test-recipe"}
            parent = {"run": "frozen-parent", "epoch": 5}
            fold = {
                "fold": 0,
                "seed": 1001,
                "parent": parent,
                "split_manifest_sha256": "split-sha",
            }
            refs = [{"split": "valid", "row_group": 7}]
            eval_meta = {
                "parent": {"endpoint_precision": "strict_fp32", "endpoint_backend": "cpu"},
                "parent_prediction_file": prediction_path.name,
                "parent_prediction_sha256": "prediction-sha",
            }
            metadata = {
                "schema": "wnn.online-relationships.parent-predictions.v1",
                "scope": "feature_only_eval",
                "recipe_id": recipe["recipe_id"],
                "recipe": str(recipe_path.resolve()),
                "recipe_sha256": "recipe-sha",
                "fold": 0,
                "seed": 1001,
                "parent": parent,
                "split_manifest_sha256": fold["split_manifest_sha256"],
                "evaluation_cache_sidecar": str(eval_meta_path),
                "evaluation_cache_sidecar_sha256": sha256_file(eval_meta_path),
                "row_refs": refs,
                "membership_sha256": row_refs_sha256(refs),
                "prediction_file": prediction_path.name,
                "prediction_path": str(prediction_path),
                "prediction_sha256": "prediction-sha",
                "shape": [1, 20000, 2],
                "dtype": "<f4",
                "endpoint": {"precision": "strict_fp32", "backend": "cpu"},
                "sampled_profile": False,
            }
            observed = verify_parent_prediction_metadata(
                root,
                recipe,
                "recipe-sha",
                fold,
                recipe_path,
                refs,
                eval_meta_path,
                eval_meta,
                prediction_path,
                prediction_meta_path,
                metadata,
            )
            self.assertEqual(observed, "prediction-sha")
            metadata["parent"] = eval_meta["parent"]
            with self.assertRaisesRegex(ValueError, "metadata is inconsistent"):
                verify_parent_prediction_metadata(
                    root,
                    recipe,
                    "recipe-sha",
                    fold,
                    recipe_path,
                    refs,
                    eval_meta_path,
                    eval_meta,
                    prediction_path,
                    prediction_meta_path,
                    metadata,
                )

    def test_official_scores_and_component_contrast_use_fixed_bins(self) -> None:
        rows = np.arange(20000, dtype=np.float32)
        target = np.stack((np.sin(rows / 11), np.cos(rows / 17)), axis=-1)[None]
        mask = np.ones((1, 20000), dtype=bool)
        candidate = (target + np.float32(0.03)).astype(np.float32)
        reference = (target - np.float32(0.04)).astype(np.float32)
        candidate_scores = official_scores(target, mask, candidate)
        reference_scores = official_scores(target, mask, reference)
        self.assertEqual(
            set(candidate_scores),
            {"overall", "rows_99_500", "rows_500_2000", "rows_2000_10000", "rows_10000_20000"},
        )
        selected = mask & (rows[None, :] >= 99)
        candidate_moments = paired_endpoint_moments(target, candidate, selected)
        reference_moments = paired_endpoint_moments(target, reference, selected)
        result = paired_component_contrast(
            target,
            mask,
            candidate,
            reference,
            np.asarray([42]),
            candidate_scores,
            reference_scores,
            candidate_moments,
            reference_moments,
        )
        self.assertEqual(
            result["overall_official_wp_paired_component_bootstrap"]["n_components"], 1
        )
        self.assertEqual(set(result["point_deltas"]), set(candidate_scores))
        self.assertEqual(set(result["point_deltas"]["overall"]), {"wp", "t0", "t1"})


if __name__ == "__main__":
    unittest.main()
