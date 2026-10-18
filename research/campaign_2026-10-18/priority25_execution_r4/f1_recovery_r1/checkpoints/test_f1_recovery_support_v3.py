from __future__ import annotations

import copy
import unittest

from f1_checkpoint_common import (
    ORIGINAL_PLAN_PATH,
    RECOVERY_ATTEMPT_ID,
    RECOVERY_PLAN_PATH,
    RECOVERY_PLAN_SHA256,
    REJECTED_ATTEMPT_ID,
    REJECTED_ATTEMPT_PATH,
    REJECTED_ATTEMPT_SHA256,
    V2_STAGE1_VERIFICATION_DEPENDENCIES,
    V2_STAGE1_VERIFIER_SHA256,
    arm_spec,
    read_json,
    validate_recovery_binding,
    validate_recovery_payload,
    validate_v2_stage1_helper_identity,
)


class RecoveryPlanTests(unittest.TestCase):
    """Test the single allowed retry and the V2 parent-helper binding."""

    def setUp(self) -> None:
        self.original = read_json(ORIGINAL_PLAN_PATH)
        self.recovery = read_json(RECOVERY_PLAN_PATH)
        self.rejected = read_json(REJECTED_ATTEMPT_PATH)

    def test_exact_retry_payload_is_pinned_and_narrow(self) -> None:
        binding = validate_recovery_binding()
        self.assertEqual(binding["recovery_plan_sha256"], RECOVERY_PLAN_SHA256)
        self.assertEqual(binding["rejected_attempt_sha256"], REJECTED_ATTEMPT_SHA256)
        self.assertEqual(binding["rejected_attempt_id"], REJECTED_ATTEMPT_ID)
        self.assertEqual(binding["rejected_attempt_status"], "not_started")
        self.assertEqual(binding["recovery_attempt_id"], RECOVERY_ATTEMPT_ID)
        self.assertEqual(
            validate_recovery_payload(self.original, self.recovery, self.rejected), binding
        )

    def test_terminal_spec_selects_only_wide_v2_retry(self) -> None:
        wide = arm_spec(self.original, "wide", "terminal")
        deep = arm_spec(self.original, "deep", "terminal")
        self.assertEqual(wide["attempt"], RECOVERY_ATTEMPT_ID)
        self.assertEqual(deep["attempt"], "computer-vision-homework-r01-deep-f1s1001-ep25-v1")
        self.assertEqual(
            wide["command"],
            next(step for step in self.original["steps"] if step["step_id"] == "continuation_wide")[
                "expected_command"
            ],
        )
        self.assertEqual(
            next(step for step in self.original["steps"] if step["step_id"] == "continuation_wide")[
                "attempt"
            ],
            REJECTED_ATTEMPT_ID,
        )

    def test_rejects_other_attempt_id(self) -> None:
        mutated = copy.deepcopy(self.recovery)
        step = next(row for row in mutated["steps"] if row["step_id"] == "continuation_wide")
        step["attempt"] = "computer-vision-homework-r01-wide-f1s1001-ep25-v3"
        with self.assertRaisesRegex(ValueError, "recovery wide attempt"):
            validate_recovery_payload(self.original, mutated, self.rejected)

    def test_rejects_changed_trainer_argv(self) -> None:
        mutated = copy.deepcopy(self.recovery)
        step = next(row for row in mutated["steps"] if row["step_id"] == "continuation_wide")
        step["expected_command"].append("--changed")
        with self.assertRaisesRegex(ValueError, "trainer argv"):
            validate_recovery_payload(self.original, mutated, self.rejected)

    def test_rejects_any_other_launch_change(self) -> None:
        mutated = copy.deepcopy(self.recovery)
        step = next(row for row in mutated["steps"] if row["step_id"] == "continuation_wide")
        step["launch_command"].append("--extra-option")
        with self.assertRaisesRegex(ValueError, "allowed-diff set"):
            validate_recovery_payload(self.original, mutated, self.rejected)

    def test_rejects_a_rejection_that_started_or_changed_contract(self) -> None:
        started = copy.deepcopy(self.rejected)
        started["status"] = "running"
        with self.assertRaisesRegex(ValueError, "not a no-start CPU admission failure"):
            validate_recovery_payload(self.original, self.recovery, started)
        wrong_contract = copy.deepcopy(self.rejected)
        wrong_contract["contract_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "base contract identity"):
            validate_recovery_payload(self.original, self.recovery, wrong_contract)

    def test_stage1_accepts_exact_v2_helper_set_only(self) -> None:
        receipt = {
            "verification_helper_sha256": V2_STAGE1_VERIFIER_SHA256,
            "verification_dependencies": dict(V2_STAGE1_VERIFICATION_DEPENDENCIES),
        }
        validate_v2_stage1_helper_identity(receipt)
        receipt["verification_helper_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "V2 stage1 verification helper SHA"):
            validate_v2_stage1_helper_identity(receipt)
        receipt["verification_helper_sha256"] = V2_STAGE1_VERIFIER_SHA256
        receipt["verification_dependencies"]["f1_checkpoint_common.py"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "V2 stage1 verification dependency hashes"):
            validate_v2_stage1_helper_identity(receipt)


if __name__ == "__main__":
    unittest.main()
