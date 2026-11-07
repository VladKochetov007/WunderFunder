"""Rate-limited resource checks and exact receipt-owned process termination."""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path

from wnn.multistage_search import atomic_write_json, read_json
from wnn.multistage_search_runtime import is_same_process, terminate_owned_process
from wnn.training.sequence_observer_resources import process_table, tracked_processes


def optional_json(path: Path) -> dict[str, object]:
    """Read optional atomically written progress without treating absence as success."""
    try:
        return read_json(path)
    except FileNotFoundError:
        return {}


def owned_records(panel: dict[str, object]) -> list[dict[str, object]]:
    """Match only the two declared command/receipt paths for each frozen cell."""
    records = []
    for cell in panel["cells"]:
        for mode in ("profile", "fit"):
            record = optional_json(Path(cell[f"{mode}_owned_receipt"]))
            if not record:
                continue
            command = record.get("command", [])
            if (
                record.get("schema") != "wnn.multistage-owned-process.v1"
                or cell["config"] not in command
                or cell[f"{mode}_output"] not in command
                or record.get("mode") != mode
            ):
                raise ValueError("owned process receipt does not match its declared cell command")
            records.append(record)
    return records


def directory_bytes(path: Path) -> int:
    """Count all new physical output files, never traversing historical-data links."""
    total = 0
    for folder, directories, files in os.walk(path, followlinks=False):
        directories[:] = [name for name in directories if not (Path(folder) / name).is_symlink()]
        for name in files:
            file = Path(folder) / name
            if not file.is_symlink():
                try:
                    total += file.stat().st_size
                except FileNotFoundError:
                    continue
    return total


class ResourceGuard:
    """Share one ten-second disk/RAM/deadline check across all eight owned workers."""

    def __init__(self, panel: dict[str, object]) -> None:
        self.panel = panel
        self.lock = threading.RLock()
        self.last_check = 0.0
        self.error: str | None = None
        self.stop = threading.Event()
        self.identities: dict[int, int] = {}
        self.snapshot: dict[str, object] = {}

    def terminate(self) -> None:
        """Signal only helper receipts whose command, PID, birth and PGID still match."""
        for record in owned_records(self.panel):
            child = record.get("child")
            if isinstance(child, dict) and is_same_process(child):
                terminate_owned_process(child, grace_seconds=0.5)

    def check(self, *, force: bool = False) -> None:
        """Fail closed before collection reserve or a storage/host-memory breach."""
        with self.lock:
            if self.stop.is_set() and self.error is None:
                self.error = "ROOT stop signal received"
            if self.error is not None:
                self.terminate()
                raise RuntimeError(self.error)
            if self.last_check and time.monotonic() - self.last_check < 10:
                return
            self.last_check = time.monotonic()
            authority = self.panel["authority"]
            runtime = Path(self.panel["runtime"])
            output = (runtime / "runs").resolve().parent
            for record in owned_records(self.panel):
                child = record.get("child")
                if isinstance(child, dict) and is_same_process(child):
                    self.identities[int(child["pid"])] = int(child["birth"])
            processes = tracked_processes(process_table(), self.identities)
            rss = sum(item.resident_bytes for item in processes.values() if item.is_alive)
            memory = {
                line.split()[0].rstrip(":"): int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith(("MemTotal:", "MemAvailable:"))
            }
            self.snapshot = {
                "at_epoch": time.time(),
                "output_bytes": directory_bytes(output),
                "output_free_bytes": shutil.disk_usage(output).free,
                "data2_free_bytes": shutil.disk_usage(runtime).free,
                "owned_worker_rss_bytes": rss,
                "host_available_bytes": memory["MemAvailable"],
                "configured_worker_threads": 64,
                "owned_live_processes": sum(item.is_alive for item in processes.values()),
            }
            checks = (
                (
                    time.time() < authority["fit_deadline_epoch"],
                    "one-hour collection reserve has started",
                ),
                (
                    self.snapshot["output_bytes"] <= authority["output_cap_bytes"],
                    "new output exceeds8GB cap",
                ),
                (
                    self.snapshot["output_free_bytes"] >= authority["output_free_floor_bytes"],
                    "/data free space below4GB floor",
                ),
                (
                    self.snapshot["data2_free_bytes"] >= authority["data2_free_floor_bytes"],
                    "/data2 free space below3GB floor",
                ),
                (rss <= authority["worker_rss_cap_bytes"], "owned worker RSS exceeds384GiB cap"),
                (
                    memory["MemAvailable"] >= authority["page_cache_headroom_bytes"],
                    "host available RAM below64GiB page-cache reserve",
                ),
            )
            atomic_write_json(Path(self.panel["control"]) / "resources.json", self.snapshot)
            for passed, reason in checks:
                if not passed:
                    self.error = reason
                    self.terminate()
                    raise RuntimeError(reason)
