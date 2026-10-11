"""Meaningful integrity and membership checks against actual H391 producers."""

from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path

PACKET = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_h391_packet", PACKET / "verify_h391_packet.py"
)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class H391PacketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.receipts = PACKET / "actual_producer_receipts_r1"
        self.split = json.loads((self.receipts / "fold1.json").read_text(encoding="utf-8"))
        self.cache = json.loads(
            (self.receipts / "dc_98411230_metadata.json").read_text(encoding="utf-8")
        )
        self.teachers = {}
        for run, record in VERIFY.EXPECTED_RUNS.items():
            self.teachers[run] = json.loads(
                (self.receipts / record["provenance"]).read_text(encoding="utf-8")
            )

    def test_frozen_packet_matches_actual_worker_receipts(self) -> None:
        result = VERIFY.verify_packet(PACKET)
        self.assertEqual(result["source_files"], 58)
        self.assertEqual(result["teacher_count"], 3)
        self.assertEqual(result["fit_groups"], 11_949)

    def test_fit_membership_is_exact_and_evaluation_rows_are_excluded(self) -> None:
        fit_train = set(self.split["fit"]["train"])
        fit_valid = set(self.split["fit"]["valid"])
        evaluation = set(self.split["evaluation"]["valid"])
        self.assertEqual((len(fit_train), len(fit_valid), len(evaluation)), (10_533, 1_416, 457))
        self.assertFalse(fit_valid & evaluation)
        self.assertEqual(set(self.cache["fit_groups"]["train"]), fit_train)
        self.assertEqual(set(self.cache["fit_groups"]["valid"]), fit_valid)
        self.assertEqual(set(self.cache["evaluation_valid_groups"]), evaluation)
        for teacher in self.teachers.values():
            self.assertEqual(set(teacher["fit_row_groups"]["train"]), fit_train)
            self.assertEqual(set(teacher["fit_row_groups"]["valid"]), fit_valid)
            self.assertEqual(set(teacher["evaluation_row_groups"]), evaluation)

    def test_cache_rejects_changed_target_family_and_membership(self) -> None:
        changed = copy.deepcopy(self.cache)
        changed["homogeneous_family"] = "raw_wmse"
        with self.assertRaisesRegex(ValueError, "homogeneous target"):
            VERIFY._validate_cache_contract(changed, self.split, self.teachers)
        changed = copy.deepcopy(self.cache)
        changed["fit_groups"]["valid"].pop()
        with self.assertRaisesRegex(ValueError, "valid membership"):
            VERIFY._validate_cache_contract(changed, self.split, self.teachers)

    def test_parent_is_the_seed4_teacher_checkpoint_and_student_since_is_zero(self) -> None:
        recipe = json.loads((PACKET / "RECIPE_R1.json").read_text(encoding="utf-8"))
        parent = recipe["parent"]
        teacher = self.teachers["vt_f1_muonall_h384_e3_s4"]
        self.assertEqual(parent["checkpoint_sha256"], teacher["checkpoint_sha256"]["model_ep2.pt"])
        self.assertEqual(
            parent["provenance_sha256"],
            VERIFY.EXPECTED_RUNS["vt_f1_muonall_h384_e3_s4"]["provenance_sha256"],
        )
        self.assertEqual(recipe["recipe"]["since_weight"], 0.0)
        self.assertEqual(teacher["args"]["since_weight"], 32.0)

    def test_target_coefficients_match_the_three_actual_strong_checkpoints(self) -> None:
        VERIFY._validate_cache_contract(self.cache, self.split, self.teachers)
        sources = {entry["run"]: entry for entry in self.cache["sources"]}
        for run, expected in VERIFY.EXPECTED_RUNS.items():
            self.assertEqual(sources[run]["checkpoint_sha256"], expected["checkpoint_sha256"])
            self.assertEqual(sources[run]["family"], "strong")
            self.assertEqual(sources[run]["coefficient"], 0.1)
        self.assertEqual(self.cache["homogeneous_family"], "strong")
        self.assertEqual(self.cache["coefficient_sum"], 0.8125)


if __name__ == "__main__":
    unittest.main()
