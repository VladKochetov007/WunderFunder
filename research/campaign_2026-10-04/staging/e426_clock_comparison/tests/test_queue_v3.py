"""Stdlib-only safety fixtures for the E426 export/score queue controller."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[5]
CONTROLLER_PATH = ROOT / "research/campaign_2026-10-04/checks/run_e426_clock_queue_v3.py"
SPEC = importlib.util.spec_from_file_location("e426_clock_queue_v3", CONTROLLER_PATH)
assert SPEC is not None and SPEC.loader is not None
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


class QueueControllerTests(unittest.TestCase):
    def test_collected_successful_worker_uses_bound_teardown_record(self) -> None:
        spec = QUEUE.read_json(QUEUE.CAMPAIGN / f"specs/{QUEUE.PREFLIGHT_SPEC}.sealed.json")
        record_path = QUEUE.RUNNER / f"attempts/{QUEUE.PREFLIGHT_ATTEMPT}.json"
        record = QUEUE.read_json(record_path)
        collected = {"LoadState": "not-found", "ActiveState": "inactive",
                     "MainPID": "0", "ExecMainStatus": "0", "SubState": "dead"}
        with patch.object(QUEUE, "properties", return_value=collected):
            self.assertEqual(QUEUE.successful_attempt(QUEUE.PREFLIGHT_ATTEMPT, spec), record)
        with patch.object(QUEUE, "properties", return_value={**collected, "MainPID": "123"}):
            with self.assertRaises(RuntimeError):
                QUEUE.successful_attempt(QUEUE.PREFLIGHT_ATTEMPT, spec)
        bad_record = {**record, "finished_unix": QUEUE.time.time() + 1000}
        with patch.object(QUEUE, "properties", return_value=collected):
            with patch.object(QUEUE, "read_json", return_value=bad_record):
                with self.assertRaises(RuntimeError):
                    QUEUE.successful_attempt(QUEUE.PREFLIGHT_ATTEMPT, spec)

    def test_export_validator_consumes_sealed_descriptors_not_draft_strings(self) -> None:
        report = QUEUE.read_json(QUEUE.PREFLIGHT_REPORT)
        specs = {}
        base = ROOT / "research/campaign_2026-10-04/specs"
        for name, _attempt, _output, _run, _fit in QUEUE.EXPORTS:
            draft = QUEUE.read_json(base / f"{name}.draft.json")
            by_run = {row["run"]: row for row in report["fits"]}
            hashes = by_run[_run]["artifacts"]
            draft["inputs"] = [
                {"path": relative,
                 "sha256": QUEUE.sha256_file(QUEUE.PREFLIGHT_REPORT)
                 if relative == QUEUE.PREFLIGHT_REPORT.relative_to(ROOT).as_posix()
                 else hashes.get(relative, "0" * 64)}
                for relative in draft["inputs"]
            ]
            specs[base / f"{name}.sealed.json"] = draft
        original_read = QUEUE.read_json
        def fixture_read(path):
            return specs[path] if path in specs else original_read(path)
        with patch.object(QUEUE, "read_json", side_effect=fixture_read):
            self.assertEqual(set(QUEUE.require_sealed_export_specs()), set(row[0] for row in QUEUE.EXPORTS))
            first = next(iter(specs.values()))
            first["inputs"] = [item for item in first["inputs"]
                               if item["path"] != QUEUE.PREFLIGHT_REPORT.relative_to(ROOT).as_posix()]
            with self.assertRaises(ValueError):
                QUEUE.require_sealed_export_specs()

    def test_completed_oneshot_handles_active_exited_and_inactive(self) -> None:
        self.assertTrue(QUEUE.terminal_service_success({
            "LoadState": "loaded", "ActiveState": "active", "SubState": "exited",
            "ExecMainStatus": "0",
        }))
        self.assertTrue(QUEUE.terminal_service_success({
            "LoadState": "loaded", "ActiveState": "inactive", "SubState": "dead",
            "ExecMainStatus": "0",
        }))
        self.assertFalse(QUEUE.terminal_service_success({
            "LoadState": "loaded", "ActiveState": "active", "SubState": "running",
            "ExecMainStatus": "0",
        }))
        self.assertFalse(QUEUE.terminal_service_success({
            "LoadState": "loaded", "ActiveState": "failed", "SubState": "failed",
            "ExecMainStatus": "1",
        }))

    def test_controller_orders_preflight_exports_seal_and_score(self) -> None:
        source = inspect.getsource(QUEUE.main)
        markers = (
            "gates = wait_for_gates()",
            "run_job(PREFLIGHT_SPEC, PREFLIGHT_ATTEMPT, \"cpu\"",
            "sealed_specs = require_sealed_export_specs()",
            "for spec_name, attempt_name, *_ in EXPORTS:",
            "score_spec = seal_score_after_exports()",
            "run_job(SCORE_SPEC, SCORE_ATTEMPT, \"cpu\"",
        )
        positions = [source.index(marker) for marker in markers]
        self.assertEqual(positions, sorted(positions))

    def test_future_inputs_and_source_pin_fail_closed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                QUEUE.require_paths_ready({"future/prediction.npy"}, root)
            target = root / "future/prediction.npy"
            target.parent.mkdir()
            target.write_bytes(b"complete")
            QUEUE.require_paths_ready({"future/prediction.npy"}, root)
            expected = hashlib.sha256(b"complete").hexdigest()
            self.assertEqual(QUEUE.require_pinned_file(target, expected), expected)
            with self.assertRaises(ValueError):
                QUEUE.require_pinned_file(target, "0" * 64)

    def test_failed_attempts_are_never_automatically_retried(self) -> None:
        for status in ("failed", "stopped", "canceled", "unknown"):
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                QUEUE.require_attempt_status(status)
        QUEUE.require_attempt_status("finished")
        QUEUE.require_attempt_status("reserved")

    def test_exact_root_incident_release_only_allows_science_scope(self) -> None:
        release = QUEUE.read_json(QUEUE.TIMING_RELEASE)
        terminal = release["terminal_services"]
        timing = {"status": "failed_closed"}
        with patch.object(QUEUE, "properties", side_effect=lambda unit: terminal[unit]):
            result = QUEUE.verify_timing_incident_release(
                timing, terminal[QUEUE.TIMING_SERVICE]
            )
        self.assertFalse(result["zip_readiness_authorized"])
        self.assertFalse(result["old_attempt_retry_authorized"])
        self.assertEqual(result["scope"], "package_timing_deferred_science_only")

    def test_release_hash_or_live_worker_mismatch_blocks_timing_override(self) -> None:
        release = QUEUE.read_json(QUEUE.TIMING_RELEASE)
        terminal = release["terminal_services"]
        timing_service = terminal[QUEUE.TIMING_SERVICE]
        original_sha256_file = QUEUE.sha256_file
        with patch.object(QUEUE, "sha256_file", side_effect=lambda path: "0" * 64
                          if path == QUEUE.TIMING_RELEASE else original_sha256_file(path)):
            with patch.object(QUEUE, "properties", side_effect=lambda unit: terminal[unit]):
                with self.assertRaises((RuntimeError, ValueError)):
                    QUEUE.verify_timing_incident_release({"status": "failed_closed"}, timing_service)
        bad_terminal = dict(terminal)
        bad_worker = dict(terminal["wnn-c04-e450-seed4-i16-timing-v2.service"])
        bad_worker["MainPID"] = "123"
        bad_terminal["wnn-c04-e450-seed4-i16-timing-v2.service"] = bad_worker
        with patch.object(QUEUE, "properties", side_effect=lambda unit: bad_terminal[unit]):
            with self.assertRaises(RuntimeError):
                QUEUE.verify_timing_incident_release({"status": "failed_closed"}, timing_service)

    def test_all_drafts_are_path_only_and_score_is_pinned(self) -> None:
        spec_paths = [
            ROOT / "research/campaign_2026-10-04/specs/e426_fit_provenance_preflight_v1.draft.json",
            *sorted((ROOT / "research/campaign_2026-10-04/specs").glob(
                "e426_clock_export_*_ep3_fp32_v2.draft.json"
            )),
            ROOT / "research/campaign_2026-10-04/specs/e426_clock_fold0_paired_score_v2.draft.json",
        ]
        import json

        for path in spec_paths:
            value = json.loads(path.read_text())
            self.assertTrue(all(isinstance(item, str) for item in value["code"] + value["inputs"]))
        score_path = spec_paths[-1]
        self.assertEqual(QUEUE.sha256_file(score_path), QUEUE.EXPECTED_SCORE_DRAFT_SHA256)
        self.assertEqual(QUEUE.QUEUE_STAGES,
                         ("timing_gate", "provenance_preflight", "three_exports", "score_seal", "score"))


if __name__ == "__main__":
    unittest.main()
