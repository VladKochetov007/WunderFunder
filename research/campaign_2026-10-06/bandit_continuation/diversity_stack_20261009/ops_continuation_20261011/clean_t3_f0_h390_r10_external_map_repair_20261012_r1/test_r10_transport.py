from __future__ import annotations

import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import build_r10_transport as build

MAIN_ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
PACKET_ROOT = Path(__file__).resolve().parent
R9_PACKET = MAIN_ROOT / build.R9_PACKET_RELATIVE
R5_FIXTURE = PACKET_ROOT / "fixtures/r5_bundle_package_r9"


def import_module(name: str, path: Path):
    """Import a frozen script by exact path for its offline consumer fixture."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mount_identity(external: dict[str, Any], logical: str) -> tuple[str, str]:
    mapping = external[logical]
    if isinstance(mapping, dict):
        dataset = mapping["dataset"]
        filename = mapping["filename"]
    else:
        dataset = "vladkochetov07/wunder-funder"
        filename = mapping
    slug = dataset.split("/", maxsplit=1)[-1]
    if Path(filename).name != filename:
        raise AssertionError(f"fixture mount filename is not flat: {filename}")
    return slug, filename


class R10TransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kernel_dir = PACKET_ROOT / "kernel"
        cls.wrapper = import_module("r10_wrapper_under_test", cls.kernel_dir / "kernel.py")
        cls.body_bytes, cls.bindings_bytes, cls.pending_release_bytes = (
            cls.wrapper._decode_embedded()
        )
        cls.bindings = json.loads(cls.bindings_bytes)
        cls.pending_release = json.loads(cls.pending_release_bytes)
        cls.worker_file_sha256 = build.sha256(cls.body_bytes)

    def _worker_module(self, path: Path) -> Any:
        path.write_bytes(self.body_bytes)
        return import_module("r10_worker_under_test", path)

    def _fixture_release(self) -> bytes:
        release = dict(self.pending_release)
        release["status"] = "admitted"
        release["authorized"] = True
        release["binding"] = self.bindings["root_release_binding"]
        release["logical_recipe_target_kernel_id"] = self.bindings["root_release_binding"][
            "target_kernel_id"
        ]
        release["transport_target_kernel_id"] = build.R10_ID
        return build.canonical_json(release)

    def test_manifest_allowlist_is_exact_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r10-helper-") as helper_temporary:
            worker = self._worker_module(Path(helper_temporary) / "worker.py")
            r5_manifest_bytes = (R5_FIXTURE / "bundle-manifest.json").read_bytes()
            self.assertEqual(build.sha256(r5_manifest_bytes), build.R5_MANIFEST_SHA256)
            manifest = json.loads(r5_manifest_bytes)
            declared = worker.authorized_external_declared_paths(manifest, self.bindings)
            external_paths = set(manifest["external_mounts"])
            self.assertEqual(len(external_paths), 13)
            self.assertTrue(external_paths.issubset(declared))
            archive_and_specs = {row["path"] for row in manifest["files"]} | set(
                manifest["spec_paths"]
            )
            self.assertFalse(external_paths & archive_and_specs)

            first_external = next(iter(external_paths))
            changed_maps = (
                {
                    key: value
                    for key, value in manifest["external_mounts"].items()
                    if key != first_external
                },
                dict(manifest["external_mounts"], unexpected_path="wrong.wnn"),
                {
                    **manifest["external_mounts"],
                    first_external: {"dataset": "wrong", "filename": "wrong.wnn"},
                },
            )
            for changed in changed_maps:
                mutated = dict(manifest, external_mounts=changed)
                with self.assertRaisesRegex(ValueError, "external mount map"):
                    worker.authorized_external_declared_paths(mutated, self.bindings)

            overlap_path = next(iter(archive_and_specs))
            overlap_map = dict(self.bindings["r5_external_mounts"])
            overlap_map[overlap_path] = {"dataset": "fixture", "filename": "overlap.wnn"}
            overlap_bindings = dict(self.bindings, r5_external_mounts=overlap_map)
            overlap_manifest = dict(manifest, external_mounts=overlap_map)
            with self.assertRaisesRegex(ValueError, "overlap"):
                worker.authorized_external_declared_paths(overlap_manifest, overlap_bindings)

            with tempfile.TemporaryDirectory(prefix="r10-unsafe-path-") as temporary:
                input_root = Path(temporary) / "input"
                project_root = Path(temporary) / "project"
                input_root.mkdir()
                for unsafe in ("/absolute/path", "../traversal"):
                    with self.assertRaisesRegex(ValueError, "unsafe R5 external"):
                        worker.mount_verified_inputs(
                            input_root, project_root, {unsafe: "fixture.wnn"}, {unsafe}
                        )

    def test_pending_release_refuses_before_materialization(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r10-pending-") as temporary:
            control_dir = Path(temporary) / "control"
            with self.assertRaisesRegex(PermissionError, "pending"):
                self.wrapper.run_embedded(control_dir)
            self.assertFalse(control_dir.exists())

        admitted = self._fixture_release()
        parsed = self.wrapper._validate_release(self.bindings_bytes, admitted)
        self.assertEqual(parsed["root_release_binding"], self.bindings["root_release_binding"])
        wrong_target = json.loads(admitted)
        wrong_target["transport_target_kernel_id"] = (
            "vladkochetov07/wnn-clean-t3-f0-h384-exp-20261012-r9"
        )
        with self.assertRaisesRegex(PermissionError, "R10 transport"):
            self.wrapper._validate_release(self.bindings_bytes, build.canonical_json(wrong_target))

    def test_frozen_builder_reproduces_kernel_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r10-rebuild-") as temporary:
            temp_root = Path(temporary)
            fixture_dir = temp_root / "fixtures/r5_bundle_package_r9"
            fixture_dir.mkdir(parents=True)
            for filename in ("bundle-manifest.json", "bundle.wnn"):
                (fixture_dir / filename).symlink_to(R5_FIXTURE / filename)
            rebuilt = build.build(MAIN_ROOT, temp_root)
            self.assertEqual(
                rebuilt["files"],
                json.loads((PACKET_ROOT / build.R10_RECEIPT_NAME).read_text())["files"],
            )
            for relative in ("kernel/kernel.py", "kernel/kernel-metadata.json"):
                self.assertEqual(
                    build.sha256_file(temp_root / relative),
                    build.sha256_file(PACKET_ROOT / relative),
                )

    def test_actual_r5_inputs_and_checkpoints_reach_intercepted_first_worker(self) -> None:
        body_path = PACKET_ROOT / "kernel/kernel.py"
        self.assertEqual(
            build.sha256_file(body_path),
            json.loads((PACKET_ROOT / build.R10_RECEIPT_NAME).read_text())["files"][
                "kernel/kernel.py"
            ]["sha256"],
        )
        self.assertEqual(
            build.sha256_file(R9_PACKET / "R9_CHECKPOINT_INPUT_MANIFEST_R2.json"),
            self.bindings["checkpoint_source_manifest_sha256"],
        )
        r5_manifest_bytes = (R5_FIXTURE / "bundle-manifest.json").read_bytes()
        r5_archive_bytes = (R5_FIXTURE / "bundle.wnn").read_bytes()
        self.assertEqual(build.sha256(r5_manifest_bytes), build.R5_MANIFEST_SHA256)
        self.assertEqual(build.sha256(r5_archive_bytes), build.R5_ARCHIVE_SHA256)
        manifest = json.loads(r5_manifest_bytes)
        observed_inputs: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="r10-full-startup-") as temporary:
            fixture_root = Path(temporary)
            declared = self._worker_module(fixture_root / "worker-helper.py")
            self.assertEqual(
                declared.authorized_external_declared_paths(manifest, self.bindings)
                & set(manifest["external_mounts"]),
                set(manifest["external_mounts"]),
            )
            input_root = fixture_root / "input"
            working_root = fixture_root / "working"
            control_dir = working_root / "control"
            input_root.mkdir()
            control_dir.mkdir(parents=True)
            bundle_mount = input_root / declared.BUNDLE_SLUG
            bundle_mount.mkdir()
            shutil.copyfile(
                R5_FIXTURE / "bundle-manifest.json", bundle_mount / "bundle-manifest.json"
            )
            shutil.copyfile(R5_FIXTURE / "bundle.wnn", bundle_mount / "bundle.wnn")

            checkpoint_mount = input_root / declared.R6_KERNEL_SLUG
            checkpoint_mount.mkdir()
            shutil.copyfile(
                R9_PACKET / "R9_CHECKPOINT_INPUT_MANIFEST_R2.json",
                checkpoint_mount / declared.CHECKPOINT_MANIFEST_FILENAME,
            )
            checkpoint_files = []
            for run_name, expected in self.bindings["checkpoints"].items():
                source_dir = MAIN_ROOT / "artifacts" / "runs" / run_name
                for kind, source_name, target_name, size_key, sha_key in (
                    (
                        "checkpoint",
                        "model_ep2.pt",
                        f"{run_name}__model_ep2.pt",
                        "model_ep2_bytes",
                        "model_ep2_sha256",
                    ),
                    (
                        "provenance",
                        "provenance.json",
                        f"{run_name}__provenance.json",
                        "provenance_bytes",
                        "provenance_sha256",
                    ),
                ):
                    source = source_dir / source_name
                    if not source.is_file() or source.is_symlink():
                        raise AssertionError(f"actual staged {kind} is unavailable: {source}")
                    declared.sha_member(source, expected[size_key], expected[sha_key])
                    target = checkpoint_mount / target_name
                    shutil.copyfile(source, target)
                    declared.sha_member(target, expected[size_key], expected[sha_key])
                    checkpoint_files.append(
                        {
                            "kind": kind,
                            "run_name": run_name,
                            "bytes": target.stat().st_size,
                            "sha256": build.sha256_file(target),
                        }
                    )

            for logical_name, _mapping in manifest["external_mounts"].items():
                source = MAIN_ROOT / logical_name
                if not source.is_file():
                    raise AssertionError(f"actual mounted source is unavailable: {source}")
                slug, filename = _mount_identity(manifest["external_mounts"], logical_name)
                destination = input_root / slug / filename
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() or destination.is_symlink():
                    raise AssertionError(f"duplicate fixture mount: {destination}")
                destination.symlink_to(source.resolve(strict=True))
                observed_inputs.append(
                    {
                        "logical_path": logical_name,
                        "dataset_slug": slug,
                        "filename": filename,
                        "source_path": str(source.resolve(strict=True)),
                        "source_bytes": source.stat().st_size,
                    }
                )

            bindings_path = control_dir / declared.BINDINGS_FILENAME
            release_path = control_dir / declared.ROOT_RELEASE_FILENAME
            worker_path = control_dir / "kernel.py"
            bindings_path.write_bytes(self.bindings_bytes)
            release_bytes = self._fixture_release()
            release_path.write_bytes(release_bytes)
            self.wrapper._validate_release(self.bindings_bytes, release_bytes)
            worker_path.write_bytes(self.body_bytes)
            worker = import_module("r10_full_startup_worker", worker_path)

            original_path = worker.Path
            original_import = worker.import_source_module
            intercepted: list[dict[str, Any]] = []

            def kaggle_path(value: str | Path) -> Path:
                text = os.fspath(value)
                if text == "/kaggle/input":
                    return input_root
                if text == "/kaggle/working":
                    return working_root
                prefix = "/kaggle/working/"
                if text.startswith(prefix):
                    return working_root / text[len(prefix) :]
                return original_path(value)

            class FirstWorkerBoundaryReached(RuntimeError):
                pass

            def import_with_worker_intercept(name: str, path: Path) -> Any:
                module = original_import(name, path)
                if name == "r7_r5_export_driver":

                    def stop_before_subprocess(
                        root: Path,
                        output_root: Path,
                        stage: str,
                        names: list[str],
                        deadline: float,
                        helpers: Any,
                    ) -> None:
                        mount_checks = []
                        for logical_name, _mapping in manifest["external_mounts"].items():
                            slug, filename = _mount_identity(
                                manifest["external_mounts"], logical_name
                            )
                            mounted = root.joinpath(*Path(logical_name).parts)
                            expected_target = input_root / slug / filename
                            mount_checks.append(
                                mounted.is_symlink()
                                and mounted.resolve(strict=True)
                                == expected_target.resolve(strict=True)
                            )
                        intercepted.append(
                            {
                                "root": str(root),
                                "stage": stage,
                                "names": list(names),
                                "external_mount_count": len(mount_checks),
                                "all_external_mounts_resolve_to_actual_sources": all(mount_checks),
                                "checkpoint_mount_count": len(checkpoint_files),
                                "driver_loaded_from": str(Path(path).resolve()),
                            }
                        )
                        raise FirstWorkerBoundaryReached(
                            "offline fixture stopped immediately before runner subprocess"
                        )

                    module.run_stage = stop_before_subprocess
                return module

            worker.Path = kaggle_path
            worker.import_source_module = import_with_worker_intercept
            before_env = {
                key: os.environ.get(key)
                for key in ("PYTHONDONTWRITEBYTECODE", "WNN_KERNEL_DEADLINE_MONOTONIC")
            }
            try:
                with self.assertRaises(FirstWorkerBoundaryReached):
                    worker.main()
            finally:
                worker.Path = original_path
                worker.import_source_module = original_import
                for key, value in before_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

            self.assertEqual(len(intercepted), 1)
            self.assertEqual(intercepted[0]["names"], [self.bindings["export_names"][0]])
            self.assertEqual(intercepted[0]["external_mount_count"], 13)
            self.assertTrue(intercepted[0]["all_external_mounts_resolve_to_actual_sources"])
            self.assertEqual(intercepted[0]["checkpoint_mount_count"], 6)
            pipeline_path = working_root / "results/r7_export_recovery/r7_export_pipeline.json"
            pipeline = json.loads(pipeline_path.read_text())
            self.assertEqual(pipeline["status"], "failed_or_blocked")
            self.assertEqual(pipeline["exports"][0]["error_type"], "FirstWorkerBoundaryReached")
            self.assertFalse(list(working_root.rglob("pred_*.npy")))

        receipt = {
            "schema": "wnn.r10_actual_producer_startup_fixture.r1",
            "created_utc": datetime.now(UTC).isoformat(),
            "result": "PASS_TO_INTERCEPTED_FIRST_EXPORT_WORKER",
            "scope": "The actual embedded R9 worker executed bundle extraction, R9 checkpoint validation/mounts, all 13 external input mounts, and exact R5 export-spec preflight.",
            "fixture_limitations": [
                "A synthetic admitted receipt was used only inside the temporary fixture to pass the wrapper authority check; it is not ROOT authorization.",
                "The first runner subprocess was intercepted before execution; no GPU worker, model load, fit, export, or prediction was started.",
                "Large external datasets and caches were symlinked by their actual MAIN paths; their payload bytes were not read by this fixture.",
            ],
            "r10_kernel_sha256": build.sha256_file(PACKET_ROOT / "kernel/kernel.py"),
            "r10_worker_sha256": self.worker_file_sha256,
            "r10_bindings_sha256": build.sha256(self.bindings_bytes),
            "r5_bundle_manifest_sha256": build.sha256(r5_manifest_bytes),
            "r5_bundle_archive_sha256": build.sha256(r5_archive_bytes),
            "r9_checkpoint_manifest_sha256": build.sha256_file(
                R9_PACKET / "R9_CHECKPOINT_INPUT_MANIFEST_R2.json"
            ),
            "r9_checkpoint_artifacts": checkpoint_files,
            "external_mounts": observed_inputs,
            "intercepted_first_worker": intercepted[0],
            "fit_workers_started": 0,
            "provider_actions": 0,
            "prediction_files_created": 0,
        }
        receipt_path = PACKET_ROOT / "R10_ACTUAL_PRODUCER_STARTUP_FIXTURE_R1.json"
        receipt_path.write_bytes(build.canonical_json(receipt))


if __name__ == "__main__":
    unittest.main(verbosity=2)
