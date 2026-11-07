"""Small synthetic receiving-gate tests; no processes, data, models or GPU execution."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import observed_worker
from receiving_checks import (
    ORACLE_TEST,
    SPLIT_SHA,
    check_native_profile,
    evaluations,
    oracle_ready,
    runtime_recipe,
    source_files,
    verify_source,
)
from run_cohort import worker_environment

from wnn.multistage_search import canonical_json_sha256, file_sha256


class ReceivingTests(unittest.TestCase):
    """Exercise the boundaries which can otherwise silently admit a changed comparison."""

    def test_runtime_environment_matches_initial_ROOT_profile(self) -> None:
        """Inherited cuBLAS or rank settings cannot create profile/fit identity drift."""
        with patch.dict(
            "os.environ",
            {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUBLAS_EMULATE_SINGLE_PRECISION": "1",
                "RANK": "7",
            },
        ):
            environment = worker_environment(
                {"source": "/frozen/source"}, {"gpu_uuid": "GPU-owned"}
            )
        self.assertNotIn("CUBLAS_WORKSPACE_CONFIG", environment)
        self.assertNotIn("CUBLAS_EMULATE_SINGLE_PRECISION", environment)
        self.assertNotIn("RANK", environment)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-owned")
        self.assertEqual(environment["OMP_NUM_THREADS"], "8")
        self.assertEqual(environment["NVIDIA_TF32_OVERRIDE"], "0")
        self.assertEqual(environment["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"], "0")

    def test_failed_observer_keeps_owned_leader_until_canonical_kill(self) -> None:
        """The observer is spawned before relay-only TERM ignore; KILL remains reachable."""
        calls = []
        process = Mock()
        process.wait.return_value = 2
        process.poll.return_value = 2
        with (
            patch.object(observed_worker.os, "name", "posix"),
            patch.object(observed_worker.os, "getpid", return_value=13),
            patch.object(observed_worker.os, "getpgrp", return_value=13, create=True),
            patch.object(observed_worker.sys, "argv", ["observed_worker.py", "--", "observer"]),
            patch.object(observed_worker, "current_process_owner", return_value={"pid": 13}),
            patch.object(
                observed_worker.subprocess,
                "Popen",
                side_effect=lambda _command: calls.append("spawn") or process,
            ),
            patch.object(
                observed_worker.signal,
                "signal",
                side_effect=lambda _kind, _handler: calls.append("ignore_TERM"),
            ),
            patch.object(
                observed_worker,
                "terminate_owned_process",
                side_effect=lambda _owner, **_kwargs: calls.append("canonical_KILL"),
            ),
        ):
            self.assertEqual(observed_worker.main(), 2)
        self.assertEqual(calls, ["spawn", "ignore_TERM", "canonical_KILL"])

    def test_runtime_recipe_preserves_scientific_policy(self) -> None:
        """Only batch and declared monitor cadence/batch settings may differ."""
        old = {
            "epochs": 12,
            "kd_weight": 0,
            "hint_weight": 0,
            "lr": 0.0003,
            "batch_size": 8,
            "monitoring": {
                "diagnostics_every_steps": 100,
                "training_telemetry_every_steps": 50,
                "eval_batch_size": 8,
            },
        }
        new = copy.deepcopy(old)
        new["batch_size"] = 16
        new["monitoring"].update(
            diagnostics_every_steps=20, training_telemetry_every_steps=10, eval_batch_size=32
        )
        runtime_recipe(old, new)
        new["lr"] = 0.003
        with self.assertRaises(ValueError):
            runtime_recipe(old, new)

    def test_source_inventory_rejects_changes_and_extra_importable_python(self) -> None:
        """Trusted inventory bytes and exact source files are independently checked."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {}
            for name in ("src/wnn/model.py", "scripts/train_sequence.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("pass\n")
                files[name] = file_sha256(path)
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps({"files": files}))
            loaded = source_files(inventory, file_sha256(inventory))
            self.assertTrue(verify_source(root, loaded))
            (root / "src/extra.py").write_text("pass\n")
            with self.assertRaises(ValueError):
                verify_source(root, loaded)
            with self.assertRaises(ValueError):
                source_files(inventory, "0" * 64)

    def test_profile_rejects_other_device_config_and_membership(self) -> None:
        """A finite profile from another seed, GPU or data membership cannot admit fitting."""
        data = {
            "split_sha256": SPLIT_SHA,
            "fit_members": [["train", 0]],
            "evaluation_members": [["valid", 1]],
            "normalization": {"scope": "all_fit", "sequence_count": 10325},
            "teacher": None,
        }
        panel = {
            "trainer_source_sha256": "source",
            "fit_members": data["fit_members"],
            "evaluation_members": data["evaluation_members"],
        }
        cell = {"normalized_config_sha256": "config", "gpu_uuid": "GPU-physical"}
        identity = {
            "config_sha256": "config",
            "source_sha256": "source",
            "data": data,
            "data_sha256": canonical_json_sha256(data),
            "world_size": 1,
            "runtime": {"devices": [{"uuid": "GPU-physical", "name": "NVIDIA L4"}]},
        }
        receipt = {
            "status": "profile_complete",
            "updates": 20,
            "scientific_checkpoint_written": False,
            "errors": [],
            "identity": identity,
        }
        check_native_profile(panel, cell, receipt)
        for field, value in (
            ("config_sha256", "otherseed"),
            ("source_sha256", "other-source"),
            ("world_size", 2),
        ):
            changed = copy.deepcopy(receipt)
            changed["identity"][field] = value
            with self.assertRaises(ValueError):
                check_native_profile(panel, cell, changed)
        changed = copy.deepcopy(receipt)
        changed["identity"]["runtime"]["devices"][0]["uuid"] = "GPU-foreign"
        with self.assertRaises(ValueError):
            check_native_profile(panel, cell, changed)

    def test_every_full_valid_epoch_requires_exact_mask_coverage(self) -> None:
        """A late wrong-mask evaluation fails even when the last epoch looks valid."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "monitor_evaluations.jsonl"
            events = [
                {
                    "kind": "evaluation",
                    "epoch": epoch,
                    "metrics": {
                        "valid/sequence_count": 1873,
                        "valid/selected_rows": 3528273,
                        "valid/wp": 0.6,
                    },
                }
                for epoch in range(1, 13)
            ]
            path.write_text("".join(json.dumps(event) + "\n" for event in events))
            self.assertEqual(len(evaluations(root, complete=True)), 12)
            events[3]["metrics"]["valid/selected_rows"] -= 1
            path.write_text("".join(json.dumps(event) + "\n" for event in events))
            with self.assertRaises(ValueError):
                evaluations(root)

    def test_mamba_requires_bound_raw_oracle_evidence(self) -> None:
        """A missing oracle waits; a passing other-device oracle is rejected."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = {
                "family": "mamba2",
                "gpu_uuid": "GPU-owned",
                "oracle_receipt": str(root / "oracle.json"),
            }
            panel = {"source_inventory_sha256": "source"}
            self.assertFalse(oracle_ready(panel, cell))
            log = root / "oracle.log"
            log.write_text("1 passed\n")
            receipt = {
                "status": "passed",
                "exit_code": 0,
                "test": ORACLE_TEST,
                "gpu_uuid": "GPU-owned",
                "source_inventory_sha256": "source",
                "evidence": {"path": str(log), "sha256": file_sha256(log)},
            }
            Path(cell["oracle_receipt"]).write_text(json.dumps(receipt))
            self.assertTrue(oracle_ready(panel, cell))
            receipt["gpu_uuid"] = "GPU-foreign"
            Path(cell["oracle_receipt"]).write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                oracle_ready(panel, cell)


if __name__ == "__main__":
    unittest.main()
