"""Focused receipt-chain checks for the cache-validated R2 preparation."""

from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

PACKET_DIR = Path(__file__).resolve().parent
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_pair_preparation_r2", PACKET_DIR / "validate_pair_preparation_r2.py"
)
if VALIDATOR_SPEC is None or VALIDATOR_SPEC.loader is None:
    raise RuntimeError("cannot load the R2 preparation validator")
validator = importlib.util.module_from_spec(VALIDATOR_SPEC)
VALIDATOR_SPEC.loader.exec_module(validator)


class PairPreparationR2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path("/home/vlad/development/wunder/wnn_connectome_starterpack").resolve()
        self.plan = json.loads((PACKET_DIR / "PAIR_PREPARATION_R2.json").read_text())
        self.input_receipt = validator.load_bound_json(
            self.plan, "actual_teacher_input_validation_receipt_r2", self.project_root
        )
        self.cache_receipt = validator.load_bound_json(
            self.plan, "cache_materialization_validation_receipt", self.project_root
        )
        self.root_release = validator.load_bound_json(
            self.plan, "root_cache_materialization_release", self.project_root
        )

    def test_real_bound_cache_receipts_pass_without_array_rehash(self) -> None:
        result = validator.inspect_preparation(self.project_root)
        self.assertEqual(result["status"], "CACHE_VALIDATED_PENDING_PROFILE_DELIVERY_AND_ROOT_RELEASE")
        self.assertEqual([entry["seed"] for entry in result["teacher_ancestry"]], [3, 4, 5])
        self.assertEqual(result["source_qualified_membership"]["fit_valid_evaluation_overlap"], 0)
        self.assertFalse(result["cache_arrays_rehashed_by_preflight"])
        self.assertFalse(result["profile_authorized"])
        self.assertFalse(result["student_fit_authorized"])

    def test_overlap_mutation_is_rejected(self) -> None:
        cache = copy.deepcopy(self.cache_receipt)
        cache["source_qualified_membership"]["fit_valid_evaluation_overlap"] = 1
        with self.assertRaises(validator.ValidationError):
            validator.verify_receipt_documents(
                self.plan, self.input_receipt, cache, self.root_release, self.project_root
            )

    def test_output_hash_mutation_is_rejected(self) -> None:
        cache = copy.deepcopy(self.cache_receipt)
        cache["output_files"]["main_homogeneous_train"]["sha256"] = "0" * 64
        with self.assertRaises(validator.ValidationError):
            validator.verify_receipt_documents(
                self.plan, self.input_receipt, cache, self.root_release, self.project_root
            )

    def test_student_profile_and_fit_authorization_remain_false(self) -> None:
        self.assertFalse(self.plan["authorization"]["gpu_profile"])
        self.assertFalse(self.plan["authorization"]["student_fit"])
        self.assertFalse(self.plan["authorization"]["provider_upload_or_push"])
        with self.assertRaises(validator.ValidationError):
            validator.verify_receipt_documents(
                self.plan,
                self.input_receipt,
                self.cache_receipt,
                {**self.root_release, "cache_materialization_authorized": False},
                self.project_root,
            )


if __name__ == "__main__":
    unittest.main()
