"""CPU-only checks for watcher admission, hashes, and official grade replay."""

from __future__ import annotations

import json
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
