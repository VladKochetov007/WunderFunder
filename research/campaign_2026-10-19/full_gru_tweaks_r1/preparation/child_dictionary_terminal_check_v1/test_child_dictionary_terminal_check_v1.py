"""Focused local fixtures for the dictionary terminal checker."""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

PREPARATION = Path(__file__).resolve().parents[1]
PACKET = Path(__file__).resolve().parent
HELPERS = {
    "child_ema_terminal_check_v3.py": PREPARATION
    / "child_ema_terminal_verifier_v3/child_ema_terminal_check_v3.py",
    "child_control_terminal_check_v3.py": PREPARATION
    / "child_control_terminal_verifier_v3/child_control_terminal_check_v3.py",
    "parent_fit_terminal_check_v3.py": PREPARATION / "parent_fit_terminal_check_v3.py",
    "propensity_terminal_support.py": PREPARATION / "propensity_terminal_support.py",
    "teacher_s5_terminal_check_v1.py": PREPARATION / "teacher_s5_terminal_check_v1.py",
}


class DictionaryTerminalCheckTests(unittest.TestCase):
    """Exercise only strict identity helpers with synthetic CPU metadata."""

    @classmethod
    def setUpClass(cls) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        cls.scratch = tempfile.TemporaryDirectory(prefix="fg19-dictionary-check-fixture-")
        cls.closure = Path(cls.scratch.name)
        shutil.copyfile(
            PACKET / "child_dictionary_terminal_check_v1.py",
            cls.closure / "child_dictionary_terminal_check_v1.py",
        )
        for name, source in HELPERS.items():
            shutil.copyfile(source, cls.closure / name)
        cls.payload_sentinel = (
            cls.closure / "data/artifacts/runs/fg19_child_dictionary_f0_s1001_fit_v1/model_ep5.pt"
        )
        cls.payload_sentinel.parent.mkdir(parents=True)
        cls.payload_sentinel.write_bytes(b"synthetic sentinel; never a model payload")

        def deny_sentinel_read(event: str, arguments: tuple[object, ...]) -> None:
            if (
                event == "open"
                and arguments
                and isinstance(arguments[0], (str, os.PathLike))
                and os.fspath(arguments[0]) == str(cls.payload_sentinel)
            ):
                raise PermissionError("fixture forbids opening the model-path sentinel")

        sys.addaudithook(deny_sentinel_read)
        sys.path.insert(0, str(cls.closure))
        cls.checker = importlib.import_module("child_dictionary_terminal_check_v1")

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(str(cls.closure))
        for name in (
            "child_dictionary_terminal_check_v1",
            "child_ema_terminal_check_v3",
            "child_control_terminal_check_v3",
            "parent_fit_terminal_check_v3",
            "propensity_terminal_support",
            "teacher_s5_terminal_check_v1",
        ):
            sys.modules.pop(name, None)
        cls.scratch.cleanup()

    def test_native_evidence_accepts_only_exact_exited_identity(self) -> None:
        stamp = datetime.now(UTC).isoformat()
        evidence = {
            "schema": "fg19.child-dictionary-fit-native-terminal-evidence.v1",
            "run_name": self.checker.RUN,
            "unit": self.checker.UNIT,
            "invocation_id": self.checker.INVOCATION,
            "active_state": "active",
            "sub_state": "exited",
            "main_pid": 0,
            "exec_main_code": 1,
            "exec_main_status": 0,
            "raw_path": "/tmp/native.raw.json",
            "raw_bytes": 1,
            "raw_sha256": "a" * 64,
            "captured_utc": stamp,
            "native_exit_utc": stamp,
        }
        self.checker._check_native_terminal(evidence)
        for key, bad in (("main_pid", False), ("exec_main_code", "1"), ("exec_main_status", 1)):
            changed = dict(evidence)
            changed[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.checker._check_native_terminal(changed)

    def test_native_evidence_rejects_wrong_run_state_and_digest(self) -> None:
        stamp = datetime.now(UTC).isoformat()
        evidence = {
            "schema": "fg19.child-dictionary-fit-native-terminal-evidence.v1",
            "run_name": self.checker.RUN,
            "unit": self.checker.UNIT,
            "invocation_id": self.checker.INVOCATION,
            "active_state": "active",
            "sub_state": "exited",
            "main_pid": 0,
            "exec_main_code": 1,
            "exec_main_status": 0,
            "raw_path": "/tmp/native.raw.json",
            "raw_bytes": 1,
            "raw_sha256": "b" * 64,
            "captured_utc": stamp,
            "native_exit_utc": stamp,
        }
        for key, bad in (
            ("invocation_id", "wrong"),
            ("sub_state", "running"),
            ("raw_sha256", "bad"),
        ):
            changed = dict(evidence)
            changed[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.checker._check_native_terminal(changed)

    def test_source_bound_sparse_recipe_rejects_ema_mode(self) -> None:
        args = {
            "name": self.checker.RUN,
            "seed": 1001,
            "train_source": "train+valid",
            "valid_fold": 0,
            "train_seqs": 10_535,
            "valid_repeat": 1,
            "valid_train_fraction": 1.0,
            "batch_order": "seeded",
            "batch": 16,
            "chunk": 2_000,
            "epochs": 6,
            "checkpoint_policy": "fixed",
            "checkpoint_epoch": 5,
            "arch": "gru",
            "cell": "gru",
            "enc": "linear",
            "hidden": 256,
            "layers": 2,
            "head_layers": 1,
            "dropout": 0.0,
            "init": "fg19_parent_h256x2_f0_s1001_fit_r2_v1@ep3",
            "inherit_init_preprocessing": True,
            "representation_mode": "sparse",
            "representation_cache": str(self.checker.DICTIONARY_PATH),
            "mean_variance_head": 1,
            "mean_variance_objective": "wmse",
            "loss": "wmse",
            "loss_norm": "chunk",
            "distill_cache_root": "/fixture/fg19_t3_fit_homogeneous_cache_v1",
            "distill_cache_variant": "homogeneous",
            "distill_cache_scale": 1.0,
            "distill_alpha": 0.9,
            "distill_family_weight": 0.0,
            "aux_struct": 0.3,
            "aux_next": 0.3,
            "optim": "adamw",
            "lr": 1e-4,
            "wd": 0.01,
            "beta2": 0.999,
            "sched": "onecycle",
            "warmup_frac": 0.05,
            "clip": 1.0,
            "precision": "fp32",
            "no_eval_during_fit": True,
            "save_training_state": True,
            "grad_stats": True,
            "split_manifest": "/fixture/splits/fold0.json",
            "device": "cuda",
        }
        self.checker._check_recipe_args(
            args,
            Path("/fixture/splits/fold0.json"),
            Path("/fixture/fg19_t3_fit_homogeneous_cache_v1"),
        )
        args["representation_mode"] = "ema"
        with self.assertRaises(ValueError):
            self.checker._check_recipe_args(
                args,
                Path("/fixture/splits/fold0.json"),
                Path("/fixture/fg19_t3_fit_homogeneous_cache_v1"),
            )

    def test_private_helper_closure_matches_all_five_source_pins(self) -> None:
        observed = self.checker._check_helper_closure()
        self.assertEqual(set(observed), set(HELPERS))
        self.assertEqual(len(observed), 5)
        self.assertEqual(
            Path(self.checker.control_v3.__file__).resolve().parent, self.closure.resolve()
        )
        self.assertEqual(
            Path(self.checker.ema_v3.__file__).resolve().parent, self.closure.resolve()
        )

    def test_import_and_helper_check_leave_model_path_sentinel_unopened(self) -> None:
        self.assertTrue(self.payload_sentinel.is_file())
        self.checker._check_helper_closure()
        self.assertEqual(self.checker._HELPER_DIR, self.closure.resolve())

    def test_sparse_auxiliary_provenance_is_exact_and_rejects_ema_mode(self) -> None:
        import torch

        representation = {
            "schema": "wnn.fitted_input_representation.v1",
            "raw_mean": torch.zeros(2),
            "raw_scale": torch.ones(2),
            "encoder_weight": torch.zeros((2, 2)),
            "decoder_weight": torch.zeros((2, 2)),
            "fit_metadata": {"fixture": True},
            "state_sha256": "a" * 64,
        }
        auxiliary = {
            "norm_ema": None,
            "dynamics": {
                "switch": None,
                "caps": {},
                "eta": 0.0,
                "clip": 1.0,
                "no_clip": False,
                "seed": 1001 + 104729,
                "is_switched": False,
                "events": [],
                "generators": {},
            },
            "pool_memory": None,
            "readout_moments": None,
            "input_representation": representation,
        }
        provenance = {
            "input_representation": {
                "mode": "sparse",
                "state_sha256": "a" * 64,
                "artifact_path": str(self.checker.DICTIONARY_PATH),
                "artifact_file_sha256": self.checker.control_v3.DICTIONARY_SHA,
                "artifact_binding": "shared_fit_cache",
                "fit_metadata": {"fixture": True},
            }
        }
        result = self.checker._check_sparse_auxiliary(
            auxiliary, provenance, {"representation": representation}
        )
        self.assertEqual(result["mode"], "sparse")
        provenance["input_representation"]["mode"] = "ema"
        with self.assertRaises(ValueError):
            self.checker._check_sparse_auxiliary(
                auxiliary, provenance, {"representation": representation}
            )

    def test_import_keeps_cuda_uninitialized(self) -> None:
        import torch

        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    unittest.main()
