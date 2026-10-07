from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_bandit_pair_contrast as grader  # noqa: E402


class FixedPairContrastTests(unittest.TestCase):
    """Guard the frozen C1v2/C0 comparison contract and fit-time input binding."""

    def test_frozen_candidate_reference_and_teacher_receipts_validate(self) -> None:
        frozen = grader._validate_frozen_inputs(
            ROOT,
            ROOT / grader.SPEC_PATH,
        )
        self.assertEqual(frozen["spec"]["candidate"]["prefix"], "bd26_c1v2")
        self.assertEqual(frozen["spec"]["reference"]["prefix"], "bd26_c0")
        self.assertEqual(set(frozen["candidate_teacher_prediction_hashes"]), {
            "pred_c20261003_combo_s3_f0_train.npy",
            "pred_c20261003_combo_s3_f0_valid.npy",
            "pred_c20261003_combo_s4_f0_train.npy",
            "pred_c20261003_combo_s4_f0_valid.npy",
        })

    def test_frozen_command_hash_rejects_a_recipe_change(self) -> None:
        frozen = grader._validate_frozen_inputs(ROOT, ROOT / grader.SPEC_PATH)
        command = copy.deepcopy(frozen["candidate_commands"][3])
        command[command.index("--distill-alpha") + 1] = "0.6"
        observed = grader._hash_command(command, "compact-json")
        expected = frozen["spec"]["candidate"]["command_sha256"][0]
        self.assertNotEqual(observed, expected)

    def test_fit_state_must_bind_audited_teacher_bytes(self) -> None:
        sources = {"trainer": "trainer-sha"}
        teacher_hashes = {"pred_teacher_train.npy": "train-sha", "pred_teacher_valid.npy": "valid-sha"}
        args = {
            "epochs": 2,
            "name": "student",
            "resume_state": None,
            "save_training_state": True,
            "stop_after_epoch": None,
            "split_manifest": "fold0.json",
            "fit_groups_manifest": None,
            "distill": "teacher*0.8125",
        }
        state = {
            "format": "wnn.epoch-training-state",
            "version": 1,
            "boundary": "epoch",
            "next_epoch": 2,
            "contract": {
                "epochs": 2,
                "sources": sources,
                "options": {"epochs": 2, "distill": "teacher*0.8125"},
                "teacher_predictions": teacher_hashes,
                "ancestors": {"initial_checkpoint_sha256": "parent-sha"},
            },
        }
        sources_sha = grader._canonical_sha256(sources)
        valid = grader._training_state_receipt(
            state,
            args,
            teacher_hashes,
            sources_sha,
            "parent-sha",
            "student",
        )
        self.assertEqual(valid["next_epoch"], 2)

        drifted = copy.deepcopy(state)
        drifted["contract"]["teacher_predictions"]["pred_teacher_train.npy"] = "changed"
        with self.assertRaisesRegex(ValueError, "teacher prediction hashes"):
            grader._training_state_receipt(
                drifted,
                args,
                teacher_hashes,
                sources_sha,
                "parent-sha",
                "student",
            )


if __name__ == "__main__":
    unittest.main()
