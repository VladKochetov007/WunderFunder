"""Focused local checks for the fixed NLL terminal checker contract."""

from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PREPARATION = Path(__file__).resolve().parents[1]
CAMPAIGN = PREPARATION.parent
RECORDS = CAMPAIGN / "records"
CHECKER_NAME = "child_nll_terminal_check_v1"
HELPERS = {
    "child_ema_terminal_check_v3.py": PREPARATION
    / "child_ema_terminal_verifier_v3/child_ema_terminal_check_v3.py",
    "child_control_terminal_check_v3.py": PREPARATION
    / "child_control_terminal_verifier_v3/child_control_terminal_check_v3.py",
    "parent_fit_terminal_check_v3.py": PREPARATION / "parent_fit_terminal_check_v3.py",
    "propensity_terminal_support.py": PREPARATION / "propensity_terminal_support.py",
    "teacher_s5_terminal_check_v1.py": PREPARATION / "teacher_s5_terminal_check_v1.py",
}


class NllTerminalCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.private_dir = tempfile.TemporaryDirectory(prefix="nll-terminal-closure-")
        private = Path(cls.private_dir.name)
        checker_path = Path(__file__).with_name(f"{CHECKER_NAME}.py")
        shutil.copyfile(checker_path, private / checker_path.name)
        for name, source in HELPERS.items():
            shutil.copyfile(source, private / name)
        cls.private_checker_sha256 = hashlib.sha256(
            (private / checker_path.name).read_bytes()
        ).hexdigest()
        sys.path.insert(0, str(private))
        cls.checker = importlib.import_module(CHECKER_NAME)

    @classmethod
    def tearDownClass(cls) -> None:
        for name in (
            CHECKER_NAME,
            "child_ema_terminal_check_v3",
            "child_control_terminal_check_v3",
            "parent_fit_terminal_check_v3",
            "propensity_terminal_support",
            "teacher_s5_terminal_check_v1",
        ):
            sys.modules.pop(name, None)
        sys.path.remove(cls.private_dir.name)
        cls.private_dir.cleanup()

    def test_exact_nll_native_identity_and_strict_exit_types(self) -> None:
        evidence = {
            "schema": "fg19.child-nll-fit-native-terminal-evidence.v1",
            "run_name": self.checker.RUN,
            "unit": self.checker.UNIT,
            "invocation_id": self.checker.INVOCATION,
            "active_state": "active",
            "sub_state": "exited",
            "main_pid": 0,
            "exec_main_code": 1,
            "exec_main_status": 0,
            "raw_path": "/capture/native.raw.json",
            "raw_bytes": 1,
            "raw_sha256": "a" * 64,
            "captured_utc": "2026-10-21T01:01:49+00:00",
            "native_exit_utc": "2026-10-21T01:01:48+00:00",
        }
        self.checker._check_native_terminal(evidence)

        for key, invalid in (
            ("run_name", "fg19_child_dictionary_f0_s1001_fit_v1"),
            ("sub_state", "running"),
            ("main_pid", False),
            ("exec_main_status", "0"),
        ):
            altered = dict(evidence)
            altered[key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.checker._check_native_terminal(altered)

    def test_private_helper_imports_are_sibling_bytes_at_frozen_pins(self) -> None:
        observed = self.checker._check_helper_closure()
        self.assertEqual(set(observed), set(self.checker.HELPER_PINS))
        for name, (expected_bytes, expected_sha, module) in self.checker.HELPER_PINS.items():
            with self.subTest(helper=name):
                self.assertEqual(
                    observed[name], {"bytes": expected_bytes, "sha256": expected_sha}
                )
                self.assertEqual(
                    Path(module.__file__).resolve().parent,
                    Path(self.private_dir.name).resolve(),
                )

    def test_cli_identity_is_base_nll_with_parent_initializer_and_wmse_compatibility(self) -> None:
        args = {
            "name": self.checker.RUN,
            "seed": 1001,
            "batch_order": "seeded",
            "batch": 16,
            "chunk": 2000,
            "epochs": 6,
            "checkpoint_epoch": 5,
            "init": "fg19_parent_h256x2_f0_s1001_fit_r2_v1@ep3",
            "representation_mode": "base",
            "representation_cache": str(self.checker.DICTIONARY_PATH),
            "mean_variance_head": 1,
            "mean_variance_objective": "nll",
            "loss": "wmse",
            "distill_cache_root": "/cache/t3",
            "optim": "adamw",
            "precision": "fp32",
            "no_eval_during_fit": True,
            "save_training_state": True,
            "grad_stats": True,
            "split_manifest": "/split/fold0.json",
            "device": "cuda",
        }
        self.checker._check_recipe_args(args, Path("/split/fold0.json"), Path("/cache/t3"))

        for key, invalid in (
            ("representation_mode", "sparse"),
            ("mean_variance_objective", "wmse"),
            ("loss", "nll"),
            ("init", "fg19_child_dictionary_f0_s1001_fit_v1@ep5"),
        ):
            altered = dict(args)
            altered[key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.checker._check_recipe_args(
                    altered, Path("/split/fold0.json"), Path("/cache/t3")
                )

    def test_frozen_actual_launch_metadata_and_three_way_receipt_binding(self) -> None:
        nll = RECORDS / "child_nll_fit_v1"
        admission_path = nll / "ADMISSION.json"
        launch_path = nll / "LAUNCH.json"
        acceptance_path = nll / "LAUNCH_ACCEPTANCE.json"
        admission = json.loads(admission_path.read_text())
        launch = json.loads(launch_path.read_text())
        acceptance = json.loads(acceptance_path.read_text())
        self.checker._check_admission(admission)
        self.checker._check_launch_records(
            admission,
            hashlib.sha256(admission_path.read_bytes()).hexdigest(),
            launch,
            hashlib.sha256(launch_path.read_bytes()).hexdigest(),
            acceptance,
            hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
        )
        self.assertNotIn("body_sha256", launch)

    def test_dictionary_child_predecessor_requires_its_exact_accepted_stop(self) -> None:
        nll_admission = json.loads(
            (RECORDS / "child_nll_fit_v1/ADMISSION.json").read_text()
        )
        dictionary = RECORDS / "child_dictionary_fit_v1"
        accepted_path = dictionary / "TERMINAL_ACCEPTANCE.json"
        stop_source = dictionary / "exact_stop/EXACT_STOP.json"
        accepted = json.loads(accepted_path.read_text())
        with tempfile.TemporaryDirectory(prefix="nll-dictionary-predecessor-") as temp:
            evidence_root = Path(temp)
            acceptance_copy = evidence_root / "DICTIONARY_CHILD_TERMINAL_ACCEPTANCE.json"
            shutil.copyfile(accepted_path, acceptance_copy)
            stop_path = evidence_root / "DICTIONARY_CHILD_EXACT_STOP.json"
            shutil.copyfile(stop_source, stop_path)
            result = self.checker._check_dictionary_child_predecessor(
                evidence_root, nll_admission
            )
            self.assertEqual(result["status"], "ACCEPTED_AND_EXACT_STOPPED")
            altered = dict(accepted)
            altered["launch_acceptance_sha256"] = "0" * 64
            acceptance_copy.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                self.checker._check_dictionary_child_predecessor(evidence_root, nll_admission)


if __name__ == "__main__":
    unittest.main()
