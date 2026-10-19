"""CPU-only checks for watcher admission, hashes, and official grade replay."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from delivery_watch_core import (
    fresh_attempt_allowed,
    sha256_file,
    verify_watcher_files,
)
from delivery_watch_grades import _grade_pair
from delivery_watch_receipts import (
    verify_export_receipt,
    verify_grade_receipt,
    verify_gradient_history,
    verify_original_actor_binding,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class RetryPolicyTests(unittest.TestCase):
    """A phase is startable only if both receipt and attempt path are absent."""

    def test_fresh_attempt_requires_no_receipt_no_path_and_pending_state(self) -> None:
        self.assertTrue(fresh_attempt_allowed(False, False, "pending"))
        self.assertFalse(fresh_attempt_allowed(True, False, "pending"))
        self.assertFalse(fresh_attempt_allowed(False, True, "pending"))
        self.assertFalse(fresh_attempt_allowed(False, False, "validation_started"))
        self.assertFalse(fresh_attempt_allowed(False, False, "rejected"))


class WatcherIdentityTests(unittest.TestCase):
    """Require the service config to pin the watcher bytes it will execute."""

    def test_watcher_self_hash_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = (
                "r01_delivery_watch.py",
                "delivery_watch_core.py",
                "delivery_watch_state.py",
                "delivery_watch_phases.py",
                "delivery_watch_grades.py",
                "delivery_watch_receipts.py",
                "publish_recovery.py",
            )
            files = {}
            for name in names:
                script = root / name
                script.write_text("pinned\n", encoding="utf-8")
                files[name] = sha256_file(script)
            verify_watcher_files(root, files)
            (root / "r01_delivery_watch.py").write_text("changed\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_watcher_files(root, files)

    def test_unsafe_watcher_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            verify_watcher_files(
                Path(temporary),
                {
                    "r01_delivery_watch.py": "x",
                    "delivery_watch_core.py": "x",
                    "delivery_watch_state.py": "x",
                    "delivery_watch_phases.py": "x",
                    "delivery_watch_grades.py": "x",
                    "delivery_watch_receipts.py": "x",
                    "publish_recovery.py": "x",
                    "../escape.py": "x",
                },
            )


class ExistingGradeTests(unittest.TestCase):
    """Accept only a complete official endpoint/reference receipt and score."""

    def test_valid_grade_receipt_reconciles_without_running_grader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            attempt = root / "grades/seed1003/fold0"
            view = attempt / "view"
            view.mkdir(parents=True)
            grader = root / "runtime/b300-r01-v1/grader/grade_endpoint.py"
            grader.parent.mkdir(parents=True)
            grader.write_text("grader\n", encoding="utf-8")
            split = root / "splits/fold0.json"
            split.parent.mkdir(parents=True)
            split.write_text("{}\n", encoding="utf-8")
            split_sha256 = sha256_file(split)
            wide_fit = root / "receipts/wide_f0_s1003.json"
            deep_fit = root / "receipts/deep_f0_s1003.json"
            wide_export = root / "exports/wide_f0_s1003/export_receipt.json"
            deep_export = root / "exports/deep_f0_s1003/export_receipt.json"
            for path in (wide_fit, deep_fit, wide_export, deep_export):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            grade = {
                "metric_id": "global_wp_clip2_v1",
                "fold": 0,
                "split_manifest_sha256": split_sha256,
                "endpoint": {"run": "deep_f0_s1003", "epoch": 5, "score": {"wp": 0.21}},
                "reference": {"run": "wide_f0_s1003", "epoch": 5, "score": {"wp": 0.20}},
                "paired_component_bootstrap": {"resamples": 2000},
            }
            command = {
                "schema": "tabdata-r01-delivery-grade-command.v2",
                "candidate": "deep_f0_s1003",
                "reference": "wide_f0_s1003",
                "fold": 0,
                "seed": 1003,
                "epoch": 5,
                "cuda_disabled": True,
                "grader_sha256": "56ffc693da52668726bab1f7008e31753a76aed3ff5d79089167785079d9eb31",
                "split_sha256": split_sha256,
                "wide_fit_recovery_sha256": sha256_file(wide_fit),
                "deep_fit_recovery_sha256": sha256_file(deep_fit),
                "wide_export_receipt_sha256": sha256_file(wide_export),
                "deep_export_receipt_sha256": sha256_file(deep_export),
                "argv": [
                    "python",
                    str(grader),
                    "--root",
                    str(view),
                    "--run",
                    "deep_f0_s1003",
                    "--split",
                    str(split),
                    "--epoch",
                    "5",
                ],
            }
            _write(attempt / "grade.json", grade)
            _write(attempt / "grade_command.json", command)
            (attempt / "stdout.txt").write_text("", encoding="utf-8")
            (attempt / "stderr.txt").write_text("", encoding="utf-8")
            receipt = {
                "schema": "tabdata-r01-delivery-grade.v2",
                "status": "graded",
                "candidate": "deep_f0_s1003",
                "reference": "wide_f0_s1003",
                "fold": 0,
                "seed": 1003,
                "metric_id": "global_wp_clip2_v1",
                "grade_sha256": sha256_file(attempt / "grade.json"),
                "command_sha256": sha256_file(attempt / "grade_command.json"),
                "stdout_sha256": sha256_file(attempt / "stdout.txt"),
                "stderr_sha256": sha256_file(attempt / "stderr.txt"),
                "score": {"deep_wp": 0.21, "wide_wp": 0.20},
            }
            _write(attempt / "grade_receipt.json", receipt)
            result = verify_grade_receipt(root, root, "wide_f0_s1003", "deep_f0_s1003", 0, 1003)
            self.assertEqual(result["status"], "graded")
            grade["endpoint"]["epoch"] = 4
            _write(attempt / "grade.json", grade)
            with self.assertRaises(ValueError):
                verify_grade_receipt(root, root, "wide_f0_s1003", "deep_f0_s1003", 0, 1003)


class GradientHistoryTests(unittest.TestCase):
    """Require recorded gradient health to cover every completed optimizer update."""

    def test_all_six_epoch_gradient_counts_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            history_path = run_dir / "artifacts/runs/cell/history.json"
            history = [
                {
                    "epoch": index,
                    "update_diagnostics": {
                        "epoch": {"optimizer_completed_updates": 10},
                        "grad_stats": {"steps": 10, "nonfinite_steps": 0},
                    },
                }
                for index in range(6)
            ]
            _write(history_path, history)
            self.assertEqual(verify_gradient_history(run_dir, "cell"), sha256_file(history_path))
            history[4]["update_diagnostics"]["grad_stats"]["steps"] = 9
            _write(history_path, history)
            with self.assertRaises(ValueError):
                verify_gradient_history(run_dir, "cell")


class PlacementBindingTests(unittest.TestCase):
    """Legacy receipts are accepted only for original placement; moved cells need proof."""

    def test_initial_legacy_receipt_and_moved_receipt_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "runs/cell"
            run_dir.mkdir(parents=True)
            resource_path = run_dir / "train_cell_resource_receipt.json"
            resource = {"visible_devices": "GPU-1", "cpu_affinity": [8, 10]}
            _write(resource_path, resource)
            job = {"id": "cell", "run_dir": str(run_dir), "cpu_ids_initial": [8, 10]}
            phase = {"resource_receipt_sha256": sha256_file(resource_path)}
            receipt = {"original_actor_phase": phase}

            def initial(*_args: object) -> dict[str, str]:
                return {"authority": "frozen_panel_initial_placement"}

            self.assertEqual(
                verify_original_actor_binding(root, job, receipt, initial)["authority"],
                "frozen_panel_initial_placement",
            )

            def moved(*_args: object) -> dict[str, str]:
                return {"authority": "sealed_density4_launch_event", "event": "x"}

            with self.assertRaises(ValueError):
                verify_original_actor_binding(root, job, receipt, moved)
            phase["placement_authorization"] = moved(root, job, resource)
            self.assertEqual(
                verify_original_actor_binding(root, job, receipt, moved),
                phase["placement_authorization"],
            )


class ExistingExportTests(unittest.TestCase):
    """Match the producer's predictor argv and the recovery wrapper argv separately."""

    def test_actual_export_command_and_actor_wrapper_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            recovery = Path(temporary) / "recovery"
            source = project / "runtime/source-a266-lf-subset-v1"
            predictor = source / "scripts/predict_run.py"
            predictor.parent.mkdir(parents=True)
            predictor.write_text("predictor\n", encoding="utf-8")
            code = project / "runtime/b300-r01-v1/code"
            code.mkdir(parents=True)
            wrapper, actor = code / "gpu_entrypoint.py", code / "export_cell.py"
            wrapper.write_text("wrapper\n", encoding="utf-8")
            actor.write_text("actor\n", encoding="utf-8")
            grader = project / "runtime/b300-r01-v1/grader/grade_endpoint.py"
            grader.parent.mkdir(parents=True)
            grader.write_text("grader\n", encoding="utf-8")
            split = project / "splits/fold0.json"
            split.parent.mkdir(parents=True)
            split.write_text("split\n", encoding="utf-8")
            fit_view = recovery / "fit_views/cell"
            fit_view.mkdir(parents=True)
            producer = fit_view / "producer_receipt.json"
            producer.write_text("{}\n", encoding="utf-8")
            fit_receipt_path = recovery / "receipts/cell.json"
            fit_receipt_path.parent.mkdir(parents=True)
            fit_receipt_path.write_text("fit\n", encoding="utf-8")
            output = recovery / "exports/cell"
            cache = output / "cache"
            cache.mkdir(parents=True)
            (cache / "pred_cell_valid.npy").write_bytes(b"prediction")
            (cache / "pred_cell_valid.json").write_text("{}\n", encoding="utf-8")
            checkpoint_sha = "f" * 64
            fit_receipt = {
                "job": {"id": "cell", "fold": 0, "seed": 1001},
                "fit_view": str(fit_view.resolve()),
                "checkpoint_identity": {"fixed_file_sha256": checkpoint_sha},
            }
            predictor_sha = sha256_file(predictor)
            split_sha = sha256_file(split)
            argv = [
                str(predictor.resolve()),
                "cell",
                "valid",
                "--epoch",
                "5",
                "--campaign-split-manifest",
                str(split.resolve()),
                "--campaign-evaluation-only",
                "--device",
                "cuda",
                "--inference-precision",
                "fp32",
            ]
            command_path = output / "export_command.json"
            command = {
                "schema": "tabdata-b300-export-command.v1",
                "run_name": "cell",
                "fit_run_dir": str(fit_view.resolve()),
                "export_dir": str(output.resolve()),
                "fold": 0,
                "seed": 1001,
                "epoch": 5,
                "source_root": str(source.resolve()),
                "predictor": str(predictor.resolve()),
                "predictor_sha256": predictor_sha,
                "grader": str(grader.resolve()),
                "grader_sha256": "56ffc693da52668726bab1f7008e31753a76aed3ff5d79089167785079d9eb31",
                "argv": argv,
                "argv_sha256": hashlib.sha256(
                    json.dumps(argv, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "split_manifest_sha256": split_sha,
                "checkpoint_sha256": checkpoint_sha,
            }
            _write(command_path, command)
            prediction_sha = sha256_file(cache / "pred_cell_valid.npy")
            sidecar_sha = sha256_file(cache / "pred_cell_valid.json")
            export_receipt = {
                "status": "completed_fixed_epoch5_feature_only_fp32",
                "run_name": "cell",
                "fold": 0,
                "seed": 1001,
                "epoch": 5,
                "feature_only": True,
                "inference_precision": "fp32",
                "target_reads_during_export": False,
                "checkpoint_sha256": checkpoint_sha,
                "predictor_sha256": predictor_sha,
                "r4_grader_sha256": command["grader_sha256"],
                "split_manifest_sha256": split_sha,
                "command_sha256": sha256_file(command_path),
                "fit_receipt_sha256": sha256_file(producer),
                "prediction_sha256": prediction_sha,
                "sidecar_sha256": sidecar_sha,
            }
            export_receipt_path = output / "export_receipt.json"
            _write(export_receipt_path, export_receipt)
            resource_path = output / "export_cell_resource_receipt.json"
            _write(
                resource_path,
                {
                    "schema": "tabdata-b300-process-envelope.v1",
                    "status": "actor_returned",
                    "script": "export_cell.py",
                    "visible_devices": "GPU-1",
                    "gpu": "NVIDIA B300",
                    "allocator_budget_bytes": 6 * 1024**3,
                    "torch_threads": 2,
                    "cuda_peak_allocated_bytes": 100,
                    "cuda_peak_reserved_bytes": 200,
                },
            )
            log_path = recovery / "logs/cell.export.log"
            log_path.parent.mkdir(parents=True)
            log_path.write_text("export done\n", encoding="utf-8")
            actor_argv = [
                sys.executable,
                str(wrapper.resolve()),
                str(actor.resolve()),
                "--source-root",
                str(source.resolve()),
                "--run-dir",
                str(fit_view.resolve()),
                "--export-dir",
                str(output.resolve()),
                "--split-manifest",
                str(split.resolve()),
                "--run-name",
                "cell",
                "--fold",
                "0",
                "--seed",
                "1001",
                "--grader-script",
                str(grader.resolve()),
            ]
            _write(
                recovery / "receipts/cell.export.json",
                {
                    "schema": "tabdata-r01-export-recovery.v2",
                    "status": "exported_fixed_epoch5_no_refit",
                    "job_id": "cell",
                    "old_fit_outputs_modified": False,
                    "fit_recovery_receipt_sha256": sha256_file(fit_receipt_path),
                    "export_receipt_sha256": sha256_file(export_receipt_path),
                    "export_command_sha256": sha256_file(command_path),
                    "export_resource_sha256": sha256_file(resource_path),
                    "actor_log_sha256": sha256_file(log_path),
                    "visible_gpu_uuid": "GPU-1",
                    "actor_argv": actor_argv,
                    "checkpoint_sha256": checkpoint_sha,
                    "prediction_sha256": prediction_sha,
                    "sidecar_sha256": sidecar_sha,
                },
            )
            result = verify_export_receipt(recovery, project, "cell", fit_receipt, "GPU-1")
            self.assertEqual(result["status"], "exported_fixed_epoch5_no_refit")
            command["argv"][2] = "train"
            _write(command_path, command)
            with self.assertRaises(ValueError):
                verify_export_receipt(recovery, project, "cell", fit_receipt, "GPU-1")
            command["argv"] = argv
            _write(command_path, command)
            recovery_receipt_path = recovery / "receipts/cell.export.json"
            recovery_receipt = json.loads(recovery_receipt_path.read_text(encoding="utf-8"))
            recovery_receipt["export_command_sha256"] = sha256_file(command_path)
            recovery_receipt["actor_argv"][2] = str(code / "other_export_cell.py")
            _write(recovery_receipt_path, recovery_receipt)
            with self.assertRaises(ValueError):
                verify_export_receipt(recovery, project, "cell", fit_receipt, "GPU-1")


class GradeTerminalTests(unittest.TestCase):
    """Known failed pairs terminate without scoring; completed grades are not republished."""

    def test_invalid_terminal_pair_becomes_not_applicable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wide = {"id": "wide", "fold": 0, "seed": 1001}
            deep = {"id": "deep", "fold": 0, "seed": 1001}
            cells = {
                "wide": {
                    "job_id": "wide",
                    "fit": {"status": "validated"},
                    "export": {"status": "exported"},
                    "grade": {"status": "pending"},
                },
                "deep": {
                    "job_id": "deep",
                    "fit": {"status": "rejected"},
                    "export": {"status": "not_applicable"},
                    "grade": {"status": "pending"},
                },
            }
            result = _grade_pair(
                {}, root, root, wide, deep, cells, lambda *_args, **_kwargs: self.fail(), 123.0
            )
            self.assertEqual(result, 123.0)
            self.assertEqual(cells["deep"]["grade"]["status"], "not_applicable")

    def test_terminal_grade_skips_revalidation_and_publication(self) -> None:
        wide = {"id": "wide", "fold": 0, "seed": 1001}
        deep = {"id": "deep", "fold": 0, "seed": 1001}
        cells = {"wide": {}, "deep": {"job_id": "deep", "grade": {"status": "graded"}}}
        result = _grade_pair(
            {}, Path("."), Path("."), wide, deep, cells, lambda *_args: self.fail(), 42.0
        )
        self.assertEqual(result, 42.0)


if __name__ == "__main__":
    unittest.main()
