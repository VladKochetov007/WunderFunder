"""Small mapping tests and actual-producer sidecar compatibility checks."""

from __future__ import annotations

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_clean_t3_f0_homogeneous_cache_r1 import (  # noqa: E402
    homogeneous_float32_chunk,
    inspect_inputs,
    load_contract,
    load_group_target_order,
    load_split_membership,
    materialize_cache,
    output_groups_from_target_orders,
    project_full_valid_rows,
    row_positions,
    selected_target_order,
)

CONTRACT_PATH = Path(__file__).with_name("CACHE_ADAPTER_CONTRACT_R1.json")
ACTUAL_ROOT = os.environ.get("WNN_T3_ARTIFACT_ROOT")


class RowGroupMappingTests(unittest.TestCase):
    """Test source-local identity joins and strict FIT_VALID projection."""

    def test_reorders_by_group_id(self) -> None:
        values = np.asarray([30, 10, 20], dtype=np.int64)
        indices = row_positions([3, 1, 2], [1, 2, 3], [1, 2, 3], "train.parquet")
        np.testing.assert_array_equal(values[indices], [10, 20, 30])

    def test_equal_numeric_ids_remain_source_local(self) -> None:
        train_map = row_positions([0, 1], [1, 0], [0, 1], "train.parquet")
        valid_map = row_positions([1, 0], [0, 1], [0, 1], "valid.parquet")
        np.testing.assert_array_equal(np.asarray([100, 200])[train_map], [200, 100])
        np.testing.assert_array_equal(np.asarray([7, 9])[valid_map], [9, 7])

    def test_projects_fit_valid_and_excludes_evaluation(self) -> None:
        indices = project_full_valid_rows(
            full_valid_order=[3, 0, 2, 1],
            fit_valid_groups=[1, 3],
            evaluation_valid_groups=[0, 2],
            output_order=[1, 3],
        )
        np.testing.assert_array_equal(indices, [3, 0])

    def test_projection_rejects_evaluation_row(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly FIT_VALID"):
            project_full_valid_rows([0, 1, 2], [1, 2], [0], [0, 1])

    def test_membership_rejects_duplicate_and_missing_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            row_positions([1, 1], [1], [1], "train.parquet")
        with self.assertRaisesRegex(ValueError, "differs"):
            row_positions([1, 2], [1], [1, 3], "train.parquet")

    def test_target_is_float32_mean_then_float32_scale_without_clipping(self) -> None:
        shape = (1, 20_000, 2)
        first = np.full(shape, 1.0, dtype=np.float32)
        second = np.full(shape, 2.0, dtype=np.float32)
        third = np.full(shape, 4.0, dtype=np.float32)
        actual = homogeneous_float32_chunk(first, second, third)
        expected = np.float32(
            np.float32(np.float32(np.float32(1.0) + np.float32(2.0) + np.float32(4.0))
                       / np.float32(3.0))
            * np.float32(0.8125)
        )
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(np.all(actual == expected))
        self.assertGreater(float(actual.max()), 1.0)

    def test_nonfinite_teacher_chunk_fails_closed(self) -> None:
        shape = (1, 20_000, 2)
        first = np.zeros(shape, dtype=np.float32)
        second = np.zeros(shape, dtype=np.float32)
        third = np.zeros(shape, dtype=np.float32)
        first[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            homogeneous_float32_chunk(first, second, third)

    def test_cache_build_requires_exact_root_release_receipt(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "explicit root release"):
            materialize_cache(
                Path.cwd(),
                CONTRACT_PATH,
                Path.cwd() / "cache" / "clean_t3_f0_h390_t3r5_a9_r1",
            )
        with tempfile.TemporaryDirectory() as temporary:
            release_path = Path(temporary) / "root_release.json"
            release_path.write_text(
                json.dumps(
                    {
                        "schema": "wnn.clean-t3-f0-cache-adapter-release.r1",
                        "cache_materialization_authorized": True,
                        "cache_adapter_contract_sha256": "0" * 64,
                        "output_cache_root": "cache/clean_t3_f0_h390_t3r5_a9_r1",
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "does not authorize"):
                materialize_cache(
                    Path.cwd(),
                    CONTRACT_PATH,
                    Path.cwd() / "cache" / "clean_t3_f0_h390_t3r5_a9_r1",
                    root_release_path=release_path,
                )


@unittest.skipUnless(ACTUAL_ROOT, "set WNN_T3_ARTIFACT_ROOT to run producer-artifact checks")
class ActualProducerTests(unittest.TestCase):
    """Exercise the adapter against the three existing full-valid producer artifacts."""

    def test_sidecars_provenance_and_source_target_indices(self) -> None:
        root = Path(str(ACTUAL_ROOT)).resolve(strict=True)
        contract = load_contract(CONTRACT_PATH)
        inspected = inspect_inputs(root, CONTRACT_PATH, require_train=False, hash_arrays=False)
        self.assertEqual(
            inspected["status"], "PARTIAL_INPUTS_VALIDATED_AWAITING_TRAIN_EXPORTS"
        )
        self.assertEqual(
            inspected["fit_train_predictions_ready"],
            ["clean_t3_f0_muonall_h384_e3_s3_r5"],
        )
        self.assertEqual(
            inspected["fit_train_predictions_pending"],
            [
                "clean_t3_f0_muonall_h384_e3_s4_r5",
                "clean_t3_f0_muonall_h384_e3_s5_r5",
            ],
        )
        records = {
            item["provenance"]["run"]: item for item in inspected["teacher_records"]
        }
        self.assertEqual(len(records), 3)
        for teacher in contract["teachers"]:
            record = records[teacher["run"]]
            valid = record["predictions"]["valid"]
            self.assertTrue(valid["ready"])
            self.assertEqual(valid["scope"], "full-valid-feature-only")
            self.assertEqual(valid["row_group_count"], 1873)
            self.assertEqual(valid["array"]["shape"], [1873, 20_000, 2])
            self.assertEqual(valid["array"]["dtype"], "float32")
            self.assertEqual(
                valid["ordered_row_groups_sha256"],
                "008f12e284e0f72ff423fee68f34c37054759bc4a345e9581620c1fe9b2acdf7",
            )

        fit, evaluation, _ = load_split_membership(root, contract)
        train_target_order = load_group_target_order(root, "train")
        valid_target_order = load_group_target_order(root, "valid")
        target_groups = output_groups_from_target_orders(
            {"train": train_target_order, "valid": valid_target_order}, fit, evaluation
        )
        self.assertEqual(len(target_groups["train"]), 10_535)
        self.assertEqual(len(target_groups["valid"]), 1_405)
        self.assertFalse(set(target_groups["valid"]) & set(evaluation))


if __name__ == "__main__":
    unittest.main()
