from __future__ import annotations

import json
import socket
import tempfile
import threading
import unittest
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts.create_dashboard_demo import create_demo
from wnn.dashboard.reader import MAX_METADATA_CACHE_ENTRIES, DashboardReader
from wnn.dashboard.schema import parse_event
from wnn.dashboard.server import MAX_CONCURRENT_REQUESTS, create_server


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

    def test_empty_and_old_event_logs_have_explicit_status(self) -> None:
        empty = self.reader.get_runs(["run-one"])["runs"][0]
        self.assertEqual(empty["status"], "no_data")
        old_time = datetime.now(UTC) - timedelta(minutes=5)
        log_path = self.run_dir / "monitor_events.jsonl"
        log_path.write_text(json.dumps(event(4, old_time)) + "\n", encoding="utf-8")
        stale = self.reader.get_runs(["run-one"])["runs"][0]
        self.assertEqual(stale["status"], "stale")


class DashboardApiTests(unittest.TestCase):
    """Exercise loopback routes, filtering, read-only behavior, and path guards."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        run_dir = self.root / "api-run"
        run_dir.mkdir()
        (run_dir / "run_metadata.json").write_text(
            json.dumps(
                {"source": "unit fixture", "config": {"architecture": "GRU"}, "provenance": {}}
            ),
            encoding="utf-8",
        )
        (run_dir / "monitor_events.jsonl").write_text(json.dumps(event(3)) + "\n", encoding="utf-8")
        self.server = create_server(0, self.root)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_offline_assets_filters_and_event_api(self) -> None:
        with urlopen(self.base + "/", timeout=2) as response:
            html = response.read().decode("utf-8")
            self.assertEqual(
                response.headers["Content-Security-Policy"].split(";")[0], "default-src 'self'"
            )
        self.assertIn("Local Research Monitor", html)
        self.assertNotIn("https://", html)
        with urlopen(self.base + "/api/runs?source=unit&config=GRU", timeout=2) as response:
            runs = json.loads(response.read())["runs"]
        self.assertEqual([run["run_id"] for run in runs], ["api-run"])
        with urlopen(self.base + "/api/run?ids=api-run&limit=2", timeout=2) as response:
            snapshot = json.loads(response.read())["runs"][0]
        self.assertEqual(snapshot["events"][0]["step"], 3)
        self.assertEqual(snapshot["status"], "live")

    def test_path_traversal_and_mutation_routes_are_rejected(self) -> None:
        with self.assertRaises(HTTPError) as traversal:
            urlopen(self.base + "/api/run?ids=..%2F..%2Fsecret", timeout=2)
        self.assertEqual(traversal.exception.code, 400)
        request = Request(self.base + "/api/run?ids=api-run", method="POST")
        with self.assertRaises(HTTPError) as mutation:
            urlopen(request, timeout=2)
        self.assertEqual(mutation.exception.code, 405)

    def test_request_workers_are_capped_and_slots_release_after_completion(self) -> None:
        all_started = threading.Event()
        release_workers = threading.Event()
        active_lock = threading.Lock()
        active_count = [0]
        original_finish = self.server.finish_request

        def hold_worker(request: socket.socket, client_address: tuple[str, int]) -> None:
            with active_lock:
                active_count[0] += 1
                if active_count[0] == MAX_CONCURRENT_REQUESTS:
                    all_started.set()
            release_workers.wait(timeout=4)
            original_finish(request, client_address)

        self.server.finish_request = hold_worker
        clients: list[socket.socket] = []
        excess: socket.socket | None = None
        try:
            for _ in range(MAX_CONCURRENT_REQUESTS):
                client = socket.create_connection(self.server.server_address, timeout=2)
                client.settimeout(4)
                client.sendall(b"GET / HTTP/1.0\r\nHost: localhost\r\n\r\n")
                clients.append(client)
            self.assertTrue(all_started.wait(timeout=3), "all worker slots should be occupied")
            excess = socket.create_connection(self.server.server_address, timeout=2)
            excess.settimeout(2)
            excess.sendall(b"GET / HTTP/1.0\r\nHost: localhost\r\n\r\n")
            response = excess.recv(256)
            self.assertTrue(response.startswith(b"HTTP/1.1 503 Service Unavailable"))
            excess.close()
            excess = None
        finally:
            release_workers.set()
            for client in clients:
                with suppress(OSError):
                    while client.recv(4_096):
                        pass
                client.close()
            if excess is not None:
                excess.close()
        slots = self.server._request_workers
        acquired = 0
        try:
            for _ in range(MAX_CONCURRENT_REQUESTS):
                self.assertTrue(
                    slots.acquire(timeout=2), "completed workers should release their slots"
                )
                acquired += 1
        finally:
            for _ in range(acquired):
                slots.release()
        with urlopen(self.base + "/", timeout=2) as response:
            self.assertEqual(response.status, 200)

    def test_server_rejects_non_loopback_binding_and_symlink_run(self) -> None:
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        with self.assertRaises(ValueError):
            create_server(0, self.root, "0.0.0.0")
        outside_temp = tempfile.TemporaryDirectory()
        outside = Path(outside_temp.name)
        try:
            (self.root / "linked-run").symlink_to(outside, target_is_directory=True)
        except OSError:
            outside_temp.cleanup()
            self.skipTest("directory symlinks are unavailable in this environment")
        listing = DashboardReader(self.root).list_runs()["runs"]
        self.assertNotIn("linked-run", [run["run_id"] for run in listing])
        outside_temp.cleanup()


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
