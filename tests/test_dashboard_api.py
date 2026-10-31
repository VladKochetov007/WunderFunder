from __future__ import annotations

import json
import socket
import tempfile
import threading
import unittest
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from wnn.dashboard.reader import DashboardReader
from wnn.dashboard.server import MAX_CONCURRENT_REQUESTS, create_server


def _event() -> dict[str, object]:
    """Build a valid fresh event for local API checks."""
    return {
        "schema": "wnn.monitor.v1",
        "kind": "train_step",
        "step": 3,
        "epoch": 0.3,
        "wall_time": datetime.now(UTC).isoformat(),
        "metrics": {"train/loss": 0.25},
        "context": {"split": "train"},
    }


class DashboardApiTests(unittest.TestCase):
    """Exercise loopback routes, request bounds, and path guards."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        run_dir = self.root / "api-run"
        run_dir.mkdir()
        metadata = {"source": "unit fixture", "config": {"architecture": "GRU"}, "provenance": {}}
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (run_dir / "monitor_events.jsonl").write_text(json.dumps(_event()) + "\n", encoding="utf-8")
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
        try:
            (self.root / "linked-run").symlink_to(outside_temp.name, target_is_directory=True)
        except OSError:
            outside_temp.cleanup()
            self.skipTest("directory symlinks are unavailable in this environment")
        listing = DashboardReader(self.root).list_runs()["runs"]
        self.assertNotIn("linked-run", [run["run_id"] for run in listing])
        outside_temp.cleanup()


if __name__ == "__main__":
    unittest.main()
