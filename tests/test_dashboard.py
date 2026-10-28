from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from scripts.create_dashboard_demo import create_demo
from wnn.dashboard.reader import MAX_METADATA_CACHE_ENTRIES, DashboardReader
from wnn.dashboard.schema import parse_event


def event(step: int, wall_time: datetime | None = None) -> dict[str, object]:
    """Build a small valid monitor event for parser and API tests."""
    return {
        "schema": "wnn.monitor.v1",
        "kind": "train_step",
        "step": step,
        "epoch": step / 10,
        "wall_time": (wall_time or datetime.now(UTC)).isoformat(),
        "metrics": {"train/loss": 1.0 / (step + 1), "optimizer/lr": 0.001},
        "context": {"split": "train"},
    }


def evaluation_event(step: int, epoch: float) -> dict[str, object]:
    """Build one combined evaluation event for the bounded epoch journal."""
    recorded = event(step)
    recorded["kind"] = "evaluation"
    recorded["epoch"] = epoch
    recorded["context"] = {"cohort_hash": "fixture-cohort", "cohort_count": 4}
    return recorded


class DashboardParserTests(unittest.TestCase):
    """Check event validation, incremental appends, rotation, and redaction."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run-one"
        self.run_dir.mkdir()
        (self.run_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "source": "local fixture",
                    "config": {"width": 128, "api_token": "do-not-return"},
                    "provenance": {"note": "token=abc123 safe"},
                }
            ),
            encoding="utf-8",
        )
        (self.run_dir / "monitor_events.jsonl").write_bytes(b"")
        self.reader = DashboardReader(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_event_rejects_nonfinite_metrics_and_naive_times(self) -> None:
        invalid = event(1)
        invalid["metrics"] = {"train/loss": float("nan")}
        self.assertIsNone(parse_event(json.dumps(invalid).encode()))
        naive = event(1)
        naive["wall_time"] = "2026-10-28T10:30:00"
        self.assertIsNone(parse_event(json.dumps(naive).encode()))
        large = event(1)
        large["metrics"] = {"train/loss": 1}
        huge_json = json.dumps(large).replace('"train/loss": 1', f'"train/loss": {"9" * 5_000}', 1)
        huge_integer = huge_json.encode()
        self.assertIsNone(parse_event(huge_integer))
        self.assertIsNotNone(parse_event(json.dumps(event(1)).encode()))

    def test_partial_append_and_truncation_reset_stream(self) -> None:
        payload = (json.dumps(event(1)) + "\n").encode()
        log_path = self.run_dir / "monitor_events.jsonl"
        log_path.write_bytes(payload[:28])
        first = self.reader.get_runs(["run-one"])["runs"][0]
        self.assertEqual(first["event_count"], 0)
        with log_path.open("ab") as handle:
            handle.write(payload[28:])
        second = self.reader.get_runs(["run-one"])["runs"][0]
        self.assertEqual(second["event_count"], 1)
        self.assertEqual(second["events"][0]["step"], 1)
        log_path.write_bytes((json.dumps(event(0)) + "\n").encode())
        third = self.reader.get_runs(["run-one"])["runs"][0]
        self.assertEqual(third["event_count"], 1)
        self.assertEqual(third["events"][0]["step"], 0)
        self.assertTrue(third["history_limited"])

    def test_invalid_lines_are_visible_while_later_events_continue(self) -> None:
        log_path = self.run_dir / "monitor_events.jsonl"
        log_path.write_text("not-json\n" + json.dumps(event(2)) + "\n", encoding="utf-8")
        snapshot = self.reader.get_runs(["run-one"])["runs"][0]
        self.assertEqual(snapshot["status"], "error")
        self.assertEqual(snapshot["invalid_lines"], 1)
        self.assertEqual(snapshot["events"][0]["step"], 2)

    def test_metadata_is_redacted_and_missing_values_are_not_added(self) -> None:
        result = self.reader.get_runs(["run-one"])
        encoded = json.dumps(result)
        self.assertNotIn("do-not-return", encoded)
        self.assertNotIn("abc123", encoded)
        run = result["runs"][0]
        self.assertEqual(run["config"]["api_token"], "[redacted]")
        self.assertNotIn("valid/loss", run["metric_keys"])

    def test_transient_metadata_error_clears_on_unchanged_cached_file(self) -> None:
        initial = self.reader.get_runs(["run-one"])["runs"][0]
        self.assertEqual(initial["metadata_status"], "ok")
        with patch(
            "wnn.dashboard.reader.read_run_metadata", side_effect=OSError("temporary read error")
        ):
            failed = self.reader.get_runs(["run-one"])["runs"][0]
        self.assertEqual(failed["metadata_status"], "incomplete")
        recovered = self.reader.get_runs(["run-one"])["runs"][0]
        self.assertEqual(recovered["metadata_status"], "ok")
        self.assertEqual(recovered["source"], "local fixture")

    def test_metadata_lru_stays_bounded_when_run_ids_churn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_ids = [f"r{index:03d}" for index in range(MAX_METADATA_CACHE_ENTRIES + 24)]
            for run_id in run_ids:
                run_dir = root / run_id
                run_dir.mkdir()
                (run_dir / "run_metadata.json").write_text(
                    json.dumps({"source": run_id, "config": {}, "provenance": {}}),
                    encoding="utf-8",
                )
            reader = DashboardReader(root)
            for offset in range(0, len(run_ids), 8):
                reader.get_runs(run_ids[offset : offset + 8])
                self.assertLessEqual(len(reader._metadata), MAX_METADATA_CACHE_ENTRIES)
            self.assertNotIn(run_ids[0], reader._metadata)
            self.assertEqual(reader.get_runs([run_ids[0]])["runs"][0]["source"], run_ids[0])
            self.assertLessEqual(len(reader._metadata), MAX_METADATA_CACHE_ENTRIES)

    def test_event_retention_and_api_downsampling_are_bounded(self) -> None:
        log_path = self.run_dir / "monitor_events.jsonl"
        log_path.write_text(
            "".join(json.dumps(event(step)) + "\n" for step in range(2_000)), encoding="utf-8"
        )
        snapshot = self.reader.get_runs(["run-one"], limit=50)["runs"][0]
        self.assertLessEqual(snapshot["retained_count"], 4_000)
        self.assertEqual(len(snapshot["events"]), 50)
        self.assertEqual(snapshot["events"][-1]["step"], 1_999)

    def test_evaluation_journal_preserves_epoch_records_past_train_tail(self) -> None:
        evaluation_events = [evaluation_event(step * 900, float(step)) for step in range(5)]
        (self.run_dir / "monitor_evaluations.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in evaluation_events), encoding="utf-8"
        )
        train_log = self.run_dir / "monitor_events.jsonl"
        with train_log.open("w", encoding="utf-8") as handle:
            for step in range(4_500):
                handle.write(json.dumps(event(step)) + "\n")
            handle.write(json.dumps(evaluation_events[0]) + "\n")

        snapshot = self.reader.get_runs(["run-one"])["runs"][0]
        retained_evaluations = [item for item in snapshot["events"] if item["kind"] == "evaluation"]
        self.assertEqual(
            {item["step"] for item in retained_evaluations}, {0, 900, 1_800, 2_700, 3_600}
        )
        self.assertEqual(len(retained_evaluations), 5)
        self.assertEqual(snapshot["evaluation_duplicates"], 1)
        self.assertTrue(snapshot["history_limited"])
        self.assertFalse(snapshot["evaluation_history_limited"])
        self.assertLessEqual(len(snapshot["events"]), 1_200)

    def test_evaluation_journal_retention_limit_is_visible(self) -> None:
        eval_log = self.run_dir / "monitor_evaluations.jsonl"
        eval_log.write_text(
            "".join(json.dumps(evaluation_event(step, float(step))) + "\n" for step in range(520)),
            encoding="utf-8",
        )
        snapshot = self.reader.get_runs(["run-one"])["runs"][0]
        self.assertEqual(snapshot["evaluation_retained_count"], 512)
        self.assertEqual(snapshot["evaluation_dropped_count"], 8)
        self.assertTrue(snapshot["evaluation_history_limited"])
        self.assertEqual(sum(item["kind"] == "evaluation" for item in snapshot["events"]), 512)

    def test_empty_and_old_event_logs_have_explicit_status(self) -> None:
        empty = self.reader.get_runs(["run-one"])["runs"][0]
        self.assertEqual(empty["status"], "no_data")
        old_time = datetime.now(UTC) - timedelta(minutes=5)
        log_path = self.run_dir / "monitor_events.jsonl"
        log_path.write_text(json.dumps(event(4, old_time)) + "\n", encoding="utf-8")
        stale = self.reader.get_runs(["run-one"])["runs"][0]
        self.assertEqual(stale["status"], "stale")


class DashboardFixtureTests(unittest.TestCase):
    """Confirm the opt-in demo generator labels its output as synthetic."""

    def test_generated_demo_is_labeled_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            created = create_demo(root)
            reader = DashboardReader(root)
            listed = reader.list_runs()["runs"]
            self.assertEqual(len(listed), 2)
            self.assertTrue(all(run["synthetic"] for run in listed))
            snapshot = reader.get_runs([created[0].name])["runs"][0]
            self.assertTrue(snapshot["synthetic"])
            self.assertIn("train_eval/loss", snapshot["metric_keys"])
            with self.assertRaises(FileExistsError):
                create_demo(root)


if __name__ == "__main__":
    unittest.main()
