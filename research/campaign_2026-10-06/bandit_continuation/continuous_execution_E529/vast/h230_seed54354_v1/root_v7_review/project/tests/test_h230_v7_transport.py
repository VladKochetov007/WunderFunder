from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

WORKER_ROOT = Path(__file__).resolve().parents[2]
worker_spec = importlib.util.spec_from_file_location("h230_v7_worker", WORKER_ROOT / "worker.py")
if worker_spec is None or worker_spec.loader is None:
    raise RuntimeError("cannot load the V7 worker")
worker = importlib.util.module_from_spec(worker_spec)
sys.modules[worker_spec.name] = worker
worker_spec.loader.exec_module(worker)


def _identity(path: Path) -> dict[str, object]:
    content = path.read_bytes()
    return {"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _load_cache_helper():
    path = WORKER_ROOT / "project/scripts/validate_h230_mounted_cache.py"
    spec = importlib.util.spec_from_file_location("validate_h230_mounted_cache", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the mounted-cache validation helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StagedInputTests(unittest.TestCase):
    def test_materializes_cache_and_split_but_links_raw_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "inputs"
            project = root / "project"
            records = {
                "cache/model/main.npy": b"npy bytes",
                "cache/model/metadata.json": b"metadata",
                "research/fold0.json": b"split bytes",
                "datasets/train.parquet": b"parquet bytes",
            }
            expected: dict[str, dict[str, object]] = {}
            for relative, value in records.items():
                source = input_root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_bytes(value)
                expected[relative] = {"path": relative, **_identity(source)}

            materialized = {
                "cache/model/main.npy",
                "cache/model/metadata.json",
                "research/fold0.json",
            }
            receipt = worker.link_staged_inputs(project, input_root, expected, materialized)
            by_path = {item["path"]: item for item in receipt}

            for relative in materialized:
                destination = project / relative
                self.assertFalse(destination.is_symlink())
                self.assertTrue(stat.S_ISREG(destination.lstat().st_mode))
                self.assertEqual(destination.read_bytes(), records[relative])
                self.assertEqual(by_path[relative]["mode"], "regular_copy_inside_project")
            raw = project / "datasets/train.parquet"
            self.assertTrue(raw.is_symlink())
            self.assertEqual(raw.resolve(), (input_root / "datasets/train.parquet").resolve())
            self.assertEqual(by_path["datasets/train.parquet"]["mode"], "input_root_symlink")

    def test_copy_hash_mismatch_leaves_no_final_or_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.bin"
            destination = root / "project/cache/input.bin"
            source.write_bytes(b"verified bytes")
            expected = {"size_bytes": source.stat().st_size, "sha256": "0" * 64}

            with self.assertRaisesRegex(ValueError, "frozen identity"):
                worker._copy_staged_input(source, destination, expected)

            self.assertFalse(destination.exists())
            self.assertFalse(list(destination.parent.glob("*.partial")))

    def test_input_count_is_explicit_and_exact(self) -> None:
        records = [{"path": f"in/{index}.bin"} for index in range(11)]
        self.assertEqual(
            len(worker.expected_inputs({"inputs": records, "expected_input_count": 11})), 11
        )
        with self.assertRaisesRegex(ValueError, "expected count"):
            worker.expected_inputs({"inputs": records, "expected_input_count": 10})


class MountedCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.helper = _load_cache_helper()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        self.cache = self.project / "cache/model"
        self.cache.mkdir(parents=True)
        self.split = self.project / "research/fold0.json"
        self.split.parent.mkdir(parents=True)
        self.split.write_text('{"version": 1}\n', encoding="utf-8")
        self.metadata = {
            "teacher_source_split": {"path": "research/fold0.json", "sha256": "a" * 64},
            "files": {
                "main_source_train": {"path": "main_source_train.npy", "sha256": "b" * 64},
                "main_source_valid": {"path": "main_source_valid.npy", "sha256": "c" * 64},
            },
        }
        (self.cache / "metadata.json").write_text(json.dumps(self.metadata), encoding="utf-8")
        for name in ("main_source_train.npy", "main_source_valid.npy"):
            array = np.lib.format.open_memmap(
                self.cache / name,
                mode="w+",
                dtype=np.float32,
                shape=(1, 20_000, 2),
            )
            array.fill(0)
            array.flush()
            del array

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _stub_core_validator(self):
        package = types.ModuleType("wnn")
        package.__path__ = []
        module = types.ModuleType("wnn.all_data_distillation_cache")
        module.validate_all_data_distillation_cache = lambda *args, **kwargs: {
            "metadata_sha256": "d" * 64,
            "verified_files": {"main_source_train": "b" * 64, "main_source_valid": "c" * 64},
        }
        return package, module

    def test_preflight_rejects_split_and_cache_array_symlinks(self) -> None:
        outside = self.root / "outside.json"
        outside.write_text('{"version": 1}\n', encoding="utf-8")
        self.split.unlink()
        self.split.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "teacher source split.*linked"):
            self.helper.validate_mounted_cache(self.project, "cache/model", "source", False)

        self.split.unlink()
        self.split.write_text('{"version": 1}\n', encoding="utf-8")
        array = self.cache / "main_source_train.npy"
        original = self.root / "original.npy"
        array.replace(original)
        array.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "cache array main_source_train.*linked"):
            self.helper.validate_mounted_cache(self.project, "cache/model", "source", False)

    def test_preflight_rejects_a_linked_cache_root(self) -> None:
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        linked = self.project / "cache/linked"
        linked.symlink_to(elsewhere, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "cache root.*linked"):
            self.helper.validate_mounted_cache(self.project, "cache/linked", "source", False)

    def test_preflight_calls_frozen_validator_after_regular_file_checks(self) -> None:
        package, module = self._stub_core_validator()
        with patch.dict(sys.modules, {"wnn": package, "wnn.all_data_distillation_cache": module}):
            receipt = self.helper.validate_mounted_cache(
                self.project, "cache/model", "source", False
            )

        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["verified_files"]["main_source_train"], "b" * 64)
        self.assertEqual(
            set(receipt["regular_arrays_under_cache_root"]),
            {"main_source_train", "main_source_valid"},
        )


class WorkerPhaseOrderTests(unittest.TestCase):
    def test_mounted_cache_preflight_runs_before_device_and_fit_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            packet_root = Path(temporary) / "packet"
            project = packet_root / "project"
            project.mkdir(parents=True)
            output = Path(temporary) / "output"
            output.mkdir()
            (output / "image_preflight.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "expected_image": "image",
                        "python_executable": "/venv/main/bin/python",
                        "torch_version": "2.14.0+cu130",
                        "cuda_runtime": "13.0",
                        "device_count": 1,
                        "device_capability": [8, 6],
                        "cuda_bf16_supported": True,
                    }
                ),
                encoding="utf-8",
            )
            calls: list[str] = []

            def fake_run_phase(argv, phase, *args, **kwargs):
                calls.append(f"phase:{phase}")
                if phase == "mounted_cache_preflight":
                    receipt = project / "artifacts/h230_mounted_cache_preflight.json"
                    receipt.parent.mkdir(parents=True, exist_ok=True)
                    receipt.write_text('{"status":"passed"}\n', encoding="utf-8")
                else:
                    gate = project / "artifacts/gate.json"
                    gate.parent.mkdir(parents=True, exist_ok=True)
                    gate.write_text(
                        '{"status":"fit_not_started_projected_runtime_exceeds_cap"}\n',
                        encoding="utf-8",
                    )
                return {"phase": phase, "exit_code": 0}

            def fake_verify_device(expected):
                calls.append("verify_device")
                return {"device_name": "fixture"}

            spec = {
                "schema": "h230-vast-worker-spec-v2",
                "attempt_id": "fixture-attempt",
                "expected_input_count": 1,
                "inputs": [{"path": "fixture.bin"}],
                "materialized_input_paths": [],
                "expected_image": "image",
                "expected_device": {},
                "resources": {
                    "environment": {},
                    "worker_ram_bytes": 1_000_000,
                    "worker_vram_bytes": 1_000_000,
                    "minimum_available_ram_bytes": 0,
                    "profile_fit_timeout_seconds": 1,
                },
                "recipe_id": "fixture",
                "source_manifest_sha256": "0" * 64,
                "artifact_project_paths": [],
                "profile_plan_path": "fixture.json",
                "profile_gate_path": "artifacts/gate.json",
                "mounted_cache_preflight": {
                    "cache_root": "cache/model",
                    "variant": "source",
                    "require_families": False,
                    "timeout_seconds": 1,
                    "receipt_path": "artifacts/h230_mounted_cache_preflight.json",
                },
            }

            with (
                patch.object(worker, "PACKET_ROOT", packet_root),
                patch.object(worker, "expected_inputs", return_value={"fixture.bin": {}}),
                patch.object(worker, "validate_staged_manifest", return_value=[]),
                patch.object(worker, "link_staged_inputs", return_value=[]),
                patch.object(worker, "verify_source", return_value={}),
                patch.object(worker, "prepare_environment", return_value={}),
                patch.object(worker, "verify_device", side_effect=fake_verify_device),
                patch.object(worker, "run_phase", side_effect=fake_run_phase),
                patch.object(worker, "write_artifact_tar"),
                patch.dict(
                    "os.environ",
                    {
                        "WNN_ATTEMPT_ID": "fixture-attempt",
                        "WNN_INPUT_DIR": str(Path(temporary) / "inputs"),
                        "WNN_INPUT_MANIFEST": "[]",
                    },
                ),
            ):
                result = worker.run_worker(spec, output)

        self.assertEqual(
            calls,
            ["phase:mounted_cache_preflight", "verify_device", "phase:profile_fit"],
        )
        self.assertEqual(result["status"], "deferred_insufficient_runtime")


if __name__ == "__main__":
    unittest.main()
