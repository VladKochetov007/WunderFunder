"""Small synthetic invariants for the bounded child CONTROL checker."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from datetime import UTC, datetime
from pathlib import Path

import child_control_terminal_check_v1 as checker
import numpy as np
import torch


def _terminal() -> dict[str, object]:
    """Build typed successful-exit metadata without any worker artifacts."""
    return {
        "schema": "fg19.child-control-fit-native-terminal-evidence.v1",
        "unit": checker.UNIT,
        "run_name": checker.RUN,
        "invocation_id": checker.INVOCATION,
        "active_state": "active",
        "sub_state": "exited",
        "main_pid": 0,
        "exec_main_code": 1,
        "exec_main_status": 0,
        "raw_bytes": 17,
        "raw_sha256": "a" * 64,
        "captured_utc": datetime.now(UTC).isoformat(),
        "raw_path": "/private/synthetic/native.stdout",
        "native_exit_utc": datetime.now(UTC).isoformat(),
    }


def _loader() -> dict[str, object]:
    """Build the source-bound final EpochOrderSampler state."""
    return {
        "sampler_type": "wnn.training_sampling.EpochOrderSampler",
        "batch_size": 16,
        "drop_last": True,
        "num_workers": 0,
        "dataset_length": 11_940,
        "loader_generator": None,
        "sampler_generator": None,
        "sampler_state": None,
        "sampler_epoch": 5,
    }


def _rng_state() -> dict[str, object]:
    """Build format-only RNG states with arbitrary synthetic content."""
    return {
        "python": (3, (0,) * 625, None),
        "numpy": ("MT19937", np.zeros(624, dtype=np.uint32), 624, 0, 0.0),
        "torch": torch.arange(5_056, dtype=torch.uint8),
        "cuda": [torch.arange(16, dtype=torch.uint8)],
    }


def _sample_order(epoch: int, digest: str = "b" * 64) -> dict[str, object]:
    """Build a logged global-RNG identity without pinned child RNG bytes."""
    identity = {
        "shape": [5_056],
        "dtype": "torch.uint8",
        "bytes_sha256": digest,
    }
    return {
        "sampler_class": "EpochOrderSampler",
        "batch_size": 16,
        "drop_last": True,
        "loader_generator": "global CPU RNG",
        "loader_rng": identity.copy(),
        "sampler_generator": "global CPU RNG",
        "sampler_rng": identity.copy(),
        "num_workers": 0,
        "persistent_workers": False,
        "sampler_replacement": None,
        "sampler_num_samples": None,
        "seeded_order": {
            "seed": 1001,
            "epoch": epoch,
            "antithetic": False,
            "shared_dropped_tail": True,
        },
    }


def _scheduler() -> dict[str, object]:
    """Build the source-pinned completed OneCycle state without fit updates."""
    return {
        "total_steps": checker.UPDATES,
        "_schedule_phases": [
            {
                "end_step": 2_237.0,
                "start_lr": "initial_lr",
                "end_lr": "max_lr",
                "start_momentum": "max_momentum",
                "end_momentum": "base_momentum",
            },
            {
                "end_step": checker.UPDATES - 1,
                "start_lr": "max_lr",
                "end_lr": "min_lr",
                "start_momentum": "base_momentum",
                "end_momentum": "max_momentum",
            },
        ],
        "_anneal_func_type": "cos",
        "cycle_momentum": True,
        "use_beta1": True,
        "base_lrs": [4e-6],
        "last_epoch": checker.UPDATES,
        "_step_count": checker.UPDATES + 1,
        "_is_initial": False,
        "_get_lr_called_within_step": False,
        "_last_lr": [4.0000136407623314e-8],
    }


def _terminal_scheduler_values() -> tuple[float, float]:
    """Evaluate the pinned final OneCycle cosine phase at last_epoch=total_steps."""
    phase_start = 2_237.0
    phase_end = float(checker.UPDATES - 1)
    fraction = (checker.UPDATES - phase_start) / (phase_end - phase_start)
    cosine_out = math.cos(math.pi * fraction) + 1
    return (
        4e-8 + (1e-4 - 4e-8) / 2.0 * cosine_out,
        0.95 + (0.85 - 0.95) / 2.0 * cosine_out,
    )


def _optimizer_shell(beta1: float, learning_rate: float) -> dict[str, object]:
    """Return only fields reached before optimizer moment validation."""
    return {
        "param_groups": [
            {
                "betas": (beta1, 0.999),
                "weight_decay": 0.01,
                "eps": 1e-8,
                "lr": learning_rate,
                "initial_lr": 4e-6,
                "max_lr": 1e-4,
                "min_lr": 4e-8,
                "base_momentum": 0.85,
                "max_momentum": 0.95,
                "params": [],
            }
        ],
        "state": {},
    }


class ChildControlTerminalCheckerTests(unittest.TestCase):
    """Exercise identity and shape rejection using small local values only."""

    def test_terminal_evidence_accepts_only_the_exact_successful_invocation(self) -> None:
        """The typed exit must bind the CONTROL unit and invocation exactly."""
        checker._check_terminal(_terminal())
        invalid = _terminal()
        invalid["invocation_id"] = "0" * 32
        with self.assertRaisesRegex(ValueError, "native terminal identity"):
            checker._check_terminal(invalid)

    def test_terminal_pid_rejects_bool_even_though_bool_is_an_int(self) -> None:
        """Typed native integer fields must not accept JSON booleans."""
        invalid = _terminal()
        invalid["main_pid"] = False
        with self.assertRaisesRegex(ValueError, "main_pid"):
            checker._check_terminal(invalid)

    def test_rng_signature_checks_source_shape_without_parent_content_hashes(self) -> None:
        """The child RNG digest is observed, not borrowed from the parent fit."""
        cpu = torch.arange(5_056, dtype=torch.uint8)
        observed = checker._tensor_rng_signature(cpu, "CPU Torch")
        expected_sha = hashlib.sha256(cpu.numpy().tobytes()).hexdigest()
        self.assertEqual(observed["sha256"], expected_sha)
        self.assertEqual(observed["shape"], [5_056])
        with self.assertRaisesRegex(ValueError, "cardinality"):
            checker._tensor_rng_signature(torch.zeros(5_055, dtype=torch.uint8), "CPU Torch")

    def test_saved_sampler_is_completed_epoch_five_while_checkpoint_is_next_epoch_six(self) -> None:
        """EpochOrderSampler saves the completed epoch, not the continuation cursor."""
        signature = checker._check_loader_and_rng({"loader": _loader(), "rng": _rng_state()})
        self.assertEqual(signature["torch"]["shape"], [5_056])
        self.assertEqual(signature["cuda"]["components"][0]["shape"], [16])
        self.assertEqual(checker.EPOCHS, 6)
        invalid = _loader()
        invalid["sampler_epoch"] = 6
        with self.assertRaisesRegex(ValueError, "loader"):
            checker._check_loader_and_rng({"loader": invalid, "rng": _rng_state()})

    def test_installed_onecycle_fixture_confirms_post_endpoint_cosine_formula(self) -> None:
        """A tiny CPU schedule confirms the saved extra cursor evaluates beyond phase end."""
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-4,
            total_steps=100,
            pct_start=0.05,
            final_div_factor=100,
            cycle_momentum=True,
        )
        for _ in range(100):
            optimizer.zero_grad()
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            scheduler.step()
        phase_start = 4.0
        phase_end = 99.0
        fraction = (100 - phase_start) / (phase_end - phase_start)
        cosine_out = math.cos(math.pi * fraction) + 1
        expected_lr = 4e-8 + (1e-4 - 4e-8) / 2.0 * cosine_out
        expected_beta1 = 0.95 + (0.85 - 0.95) / 2.0 * cosine_out
        self.assertEqual(scheduler.last_epoch, 100)
        self.assertEqual(scheduler.get_last_lr(), [expected_lr])
        self.assertEqual(optimizer.param_groups[0]["betas"][0], expected_beta1)
        self.assertGreater(expected_lr, 4e-8)
        self.assertLess(expected_beta1, 0.95)

    def test_terminal_scheduler_rejects_nominal_min_lr_instead_of_cosine_cursor_value(self) -> None:
        """The actual saved cursor is one past phase end, so min_lr is too small."""
        expected_lr, expected_beta1 = _terminal_scheduler_values()
        self.assertEqual(checker._check_scheduler(_scheduler()), (expected_lr, expected_beta1))
        invalid = _scheduler()
        invalid["_last_lr"] = [4e-8]
        with self.assertRaisesRegex(ValueError, "terminal learning rate"):
            checker._check_scheduler(invalid)

    def test_optimizer_rejects_unannealed_beta1_and_wrong_schedule_endpoints(self) -> None:
        """Arbitrary legal beta1 or altered group endpoint metadata must fail."""
        expected_lr, expected_beta1 = _terminal_scheduler_values()
        valid_schedule = _optimizer_shell(expected_beta1, expected_lr)
        with self.assertRaisesRegex(ValueError, "parameter identity count"):
            checker._check_optimizer(valid_schedule, expected_lr, expected_beta1)
        invalid_beta = _optimizer_shell(0.9, expected_lr)
        with self.assertRaisesRegex(ValueError, "terminal beta1"):
            checker._check_optimizer(invalid_beta, expected_lr, expected_beta1)
        invalid_endpoint = _optimizer_shell(expected_beta1, expected_lr)
        invalid_endpoint["param_groups"][0]["max_lr"] = 2e-4
        with self.assertRaisesRegex(ValueError, "group endpoints"):
            checker._check_optimizer(invalid_endpoint, expected_lr, expected_beta1)

    def test_sample_order_accepts_arbitrary_child_digest_but_requires_each_epoch(self) -> None:
        """Seeded order epochs are checked without hard-coding trajectory RNG hashes."""
        observed = checker._check_sample_order(_sample_order(5, "c" * 64), 5)
        self.assertEqual(observed["rng_sha256"], "c" * 64)
        with self.assertRaisesRegex(ValueError, "sample-order contract"):
            checker._check_sample_order(_sample_order(4), 5)

    def test_representation_copy_rejects_a_changed_tensor(self) -> None:
        """Saved checkpoint and state copies must match the accepted transform."""
        representation = {
            "schema": "wnn.fitted_input_representation.v1",
            "raw_mean": torch.zeros(2),
            "raw_scale": torch.ones(2),
            "encoder_weight": torch.zeros((2, 2)),
            "decoder_weight": torch.zeros((2, 2)),
            "fit_metadata": {"split_manifest_sha256": checker.SPLIT_SHA},
            "state_sha256": "d" * 64,
        }
        checker._check_representation_payload(representation.copy(), representation, "synthetic")
        changed = representation.copy()
        changed["raw_mean"] = torch.ones(2)
        with self.assertRaisesRegex(ValueError, "raw_mean"):
            checker._check_representation_payload(changed, representation, "synthetic")

    def test_isolated_cold_import_closure_and_lazy_helpers(self) -> None:
        """Import the complete local checker closure in a fresh isolated interpreter."""
        repository = Path(__file__).resolve().parents[4]
        preparation = Path(__file__).resolve().parent
        source_root = repository.parent / "WunderFunder-strong-gru-tweaks-20261019"
        launch_path = (
            repository
            / "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_control_fit_v1/LAUNCH.json"
        )
        script = textwrap.dedent(
            """
            import hashlib, importlib, json, os, sys
            from pathlib import Path
            import numpy as np
            import torch
            preparation, canonical_preparation, source_root, launch_path = map(Path, sys.argv[1:5])
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["WNN_PROJECT_ROOT"] = "/private/nonexistent-child-control-payload-sentinel"
            sys.path.insert(0, str(preparation))
            blocked_roots = (
                (source_root.parent / "data").resolve(),
                (source_root.parent / "research_tmp").resolve(),
                Path("/home/grinchenkoeg/computer-vision-homework-20261018/data"),
                Path("/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp"),
            )
            payload_opens = []
            def payload_audit(event, arguments):
                if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes, os.PathLike)):
                    return
                candidate = Path(arguments[0]).resolve()
                if any(candidate == root or candidate.is_relative_to(root) for root in blocked_roots):
                    payload_opens.append(str(candidate))
                    raise AssertionError("cold import attempted to open a B payload")
            sys.addaudithook(payload_audit)
            assert not any(name in sys.modules for name in (
                "child_control_terminal_check_v1", "parent_fit_terminal_check_v3",
                "propensity_terminal_support", "teacher_s5_terminal_check_v1",
            ))
            import child_control_terminal_check_v1 as child
            import parent_fit_terminal_check_v3 as parent
            import propensity_terminal_support as support
            import teacher_s5_terminal_check_v1 as teacher
            files = {
                "child_control_terminal_check_v1.py": preparation / "child_control_terminal_check_v1.py",
                "parent_fit_terminal_check_v3.py": preparation / "parent_fit_terminal_check_v3.py",
                "propensity_terminal_support.py": preparation / "propensity_terminal_support.py",
                "teacher_s5_terminal_check_v1.py": preparation / "teacher_s5_terminal_check_v1.py",
            }
            hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
            assert hashes["child_control_terminal_check_v1.py"] == "663d4aee291c86e87475e0599a3151f60f684943c59dfc406d2f35ddfba63cad"
            assert hashes["parent_fit_terminal_check_v3.py"] == "f2b819c1e105171f0093fae433aec301347645f8f64717effa16917005dbfe41"
            assert hashes["propensity_terminal_support.py"] == "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409"
            assert hashes["teacher_s5_terminal_check_v1.py"] == "cb7b482cc0821421b3c62f87fea1724b4148e8e85f468ae1d65eb6ccf5db41a5"
            raw, _, _ = support._stable_file(launch_path, child.MAX_JSON_BYTES)
            launch = json.loads(raw)
            command = launch["command"]
            positions = [i for i, item in enumerate(command) if item.endswith("/scripts/train_rnn.py")]
            assert len(positions) == 1
            command[positions[0]] = str(source_root / "scripts/train_rnn.py")
            launch["command"] = command
            args = teacher._launch_cli_args(launch, source_root)
            assert (args["name"], args["seed"], args["batch_order"], args["representation_mode"]) == (
                child.RUN, 1001, "seeded", "base"
            )
            tensor_identity = teacher._canonical_tensor_identity(source_root)
            assert tensor_identity(torch.tensor([1.0, 2.0]))["shape"] == [2]
            imported = {}
            for module_name, relative in (
                ("wnn.training.cli", "src/wnn/training/cli.py"),
                ("wnn.run_identity", "src/wnn/run_identity.py"),
                ("wnn.training.input_representation", "src/wnn/training/input_representation.py"),
                ("wnn.training.input_representation_fit", "src/wnn/training/input_representation_fit.py"),
                ("wnn.training.input_representation_dictionary", "src/wnn/training/input_representation_dictionary.py"),
            ):
                module = importlib.import_module(module_name)
                assert Path(module.__file__).resolve() == (source_root / relative).resolve()
                imported[relative] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            transitive_source_modules = {}
            for module_name, module in sys.modules.items():
                module_file = getattr(module, "__file__", None)
                if module_file is None or not (module_name == "wnn" or module_name.startswith("wnn.")):
                    continue
                resolved = Path(module_file).resolve()
                if not resolved.is_relative_to(source_root.resolve()):
                    raise AssertionError(f"wnn source import escaped the verified tree: {resolved}")
                relative = str(resolved.relative_to(source_root.resolve()))
                transitive_source_modules[relative] = hashlib.sha256(resolved.read_bytes()).hexdigest()
            private_d_origins = {}
            for module_name, filename in (
                ("child_control_terminal_check_v1", "child_control_terminal_check_v1.py"),
                ("parent_fit_terminal_check_v3", "parent_fit_terminal_check_v3.py"),
                ("propensity_terminal_support", "propensity_terminal_support.py"),
                ("teacher_s5_terminal_check_v1", "teacher_s5_terminal_check_v1.py"),
            ):
                origin = Path(sys.modules[module_name].__file__).resolve()
                assert origin == (preparation / filename).resolve()
                private_d_origins[module_name] = str(origin)
            for module_name, module in sys.modules.items():
                module_spec = getattr(module, "__spec__", None)
                module_origin = getattr(module_spec, "origin", None)
                if module_origin is None or not Path(module_origin).is_absolute():
                    continue
                origin = Path(module_origin).resolve()
                if origin.is_relative_to(canonical_preparation.resolve()):
                    assert module_name in private_d_origins, (
                        f"unexpected preparation module loaded outside the four-file copy: {module_name}"
                    )
                    assert origin == (preparation / origin.name).resolve()
            safe_globals = parent._safe_globals()
            assert "numpy.ndarray" in {f"{item.__module__}.{item.__qualname__}" for item in safe_globals}
            assert support._json_file(launch_path)[0]["unit"] == child.UNIT
            observed = child._tensor_rng_signature(torch.zeros(5_056, dtype=torch.uint8), "CPU Torch")
            assert observed["shape"] == [5_056]
            print(json.dumps({
                "private_d_sha256": hashes,
                "private_d_module_origins": private_d_origins,
                "private_d_origins_are_temporary_exact_copies": True,
                "pinned_source_modules_sha256": imported,
                "transitive_source_modules_sha256": transitive_source_modules,
                "all_transitive_wnn_origins_within_source_root": True,
                "source_imports": "cli, run_identity, fitted representation, representation fit validator, dictionary recipe",
                "lazy_helpers": ["teacher._launch_cli_args", "teacher._canonical_tensor_identity", "parent._safe_globals", "support._stable_file", "support._json_file", "child._tensor_rng_signature"],
                "torch_version_local_cold_import": torch.__version__,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "cuda_initialized": torch.cuda.is_initialized(),
                "payload_open_attempts": len(payload_opens),
            }, sort_keys=True))
            """
        )
        environment = os.environ.copy()
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "WNN_THREADS",
            "RAYON_NUM_THREADS",
        ):
            environment[name] = "1"
        private_sources = (
            "child_control_terminal_check_v1.py",
            "parent_fit_terminal_check_v3.py",
            "propensity_terminal_support.py",
            "teacher_s5_terminal_check_v1.py",
        )
        with tempfile.TemporaryDirectory(prefix="child-control-terminal-cold-") as temporary:
            isolated_preparation = Path(temporary)
            for filename in private_sources:
                shutil.copyfile(preparation / filename, isolated_preparation / filename)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    script,
                    str(isolated_preparation),
                    str(preparation),
                    str(source_root),
                    str(launch_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
                env=environment,
            )
        if completed.returncode != 0:
            self.fail(f"isolated cold import failed: {completed.stderr}")
        proof = json.loads(completed.stdout)
        if os.environ.get("CHILD_CONTROL_PRINT_COLD_PROOF") == "1":
            print(completed.stdout.strip())
        self.assertEqual(len(proof["private_d_sha256"]), 4)
        self.assertEqual(len(proof["private_d_module_origins"]), 4)
        self.assertTrue(proof["private_d_origins_are_temporary_exact_copies"])
        self.assertTrue(proof["all_transitive_wnn_origins_within_source_root"])
        self.assertEqual(len(proof["pinned_source_modules_sha256"]), 5)
        self.assertGreaterEqual(len(proof["transitive_source_modules_sha256"]), 5)
        self.assertIn("teacher._launch_cli_args", proof["lazy_helpers"])
        self.assertEqual(proof["cuda_visible_devices"], "")
        self.assertFalse(proof["cuda_initialized"])
        self.assertEqual(proof["payload_open_attempts"], 0)
        proof_path = os.environ.get("CHILD_CONTROL_COLD_PROOF_PATH")
        if proof_path is not None:
            input_paths = {
                "launch": repository
                / "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_control_fit_v1/LAUNCH.json",
                "admission": repository
                / "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_control_fit_v1/ADMISSION.json",
                "launch_acceptance": repository
                / "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_control_fit_v1/LAUNCH_ACCEPTANCE.json",
                "recipe_binding": repository
                / "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_control_fit_v1/CONTROL_RECIPE_BINDING_V1.json",
            }
            receipt = {
                "schema": "fg19.child-control-terminal-cold-import-receipt.v1",
                "status": "PASS",
                "runner": {
                    "test_file": str(Path(__file__).resolve()),
                    "test_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "checker_file": str(preparation / "child_control_terminal_check_v1.py"),
                    "checker_file_sha256": proof["private_d_sha256"][
                        "child_control_terminal_check_v1.py"
                    ],
                    "command": "from the preparation directory: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WNN_THREADS=1 RAYON_NUM_THREADS=1 ${WUNDER_MAINTENANCE_ROOT}/.venv/bin/python -m unittest test_child_control_terminal_check_v1.ChildControlTerminalCheckerTests.test_isolated_cold_import_closure_and_lazy_helpers -v",
                },
                "inputs": {
                    label: {
                        "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for label, path in input_paths.items()
                },
                "source_tree": {
                    "path": str(source_root),
                    "commit": checker.COMMIT,
                    "manifest_sha256": checker.SOURCE_MANIFEST_SHA,
                },
                "local_runtime": {
                    "python_executable": sys.executable,
                    "python_version": sys.version.split()[0],
                    "numpy_version": np.__version__,
                    "torch_version": torch.__version__,
                    "cuda_visible_devices": "",
                    "cuda_initialized": False,
                    "payload_open_attempts": 0,
                    "thread_environment": {
                        name: os.environ.get(name)
                        for name in (
                            "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "WNN_THREADS",
                            "RAYON_NUM_THREADS",
                        )
                    },
                },
                "cold_import_proof": proof,
            }
            descriptor = os.open(proof_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())


if __name__ == "__main__":
    unittest.main()
