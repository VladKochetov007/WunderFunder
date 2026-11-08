from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time
from contextlib import suppress
from pathlib import Path

import torch

from wnn.multistage_search import atomic_write_json, file_lock, file_sha256, read_json
from wnn.multistage_search_runtime import is_same_process
from wnn.training.sequence_observer_resources import process_table, tracked_processes
from wnn.training.sequence_provenance import canonical_sha256

SOURCE_SHA = "5bbbf68eccdc8c8773ff4b4eda2c4026d9c3f31b145dfc0d683b17d20a1987d0"
SPLIT_SHA = "0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c"
CONFIG_HASHES = {
    "event211_d02_s101101": {
        "raw": "40ac84be2a95395355f67c31420547da2bfb650010831afc53b5f469bdb285a5",
        "normalized": "245e85b44664807515407ddd6ced14501013fa365267f04d28a1bb2cb6d69fbb",
    },
    "event211_d02_s101102": {
        "raw": "67e76c6a860c82325af96d4c6edcacada7325f1f3a55c805e9be4738a64120f7",
        "normalized": "7e536e62e3ad3123b30fac0a39c27fee879ae382270baa0d0ebc1f8bb9394985",
    },
    "raw112_d02_s101101": {
        "raw": "b7b5f377f1b9f4b2a92538fa14ff4e6919148db628da312190dc3223d2f74e8a",
        "normalized": "c148aec9b9d41e46c1f17ac0947298c8404bb2f4a3a2ac32ba1cfc52785c549f",
    },
    "raw112_d02_s101102": {
        "raw": "c7f415ca0d96033bbf7844a22a5ea2b8a01bbc61be1c0e0a8237d0f8cf2b4164",
        "normalized": "79dc45cb5979ad35d892df5441e91a0ce4b7ace98b2c0f6c661811add2eab588",
    },
    "raw112_d0_s101101": {
        "raw": "d9ff7b1115a6c368e3911005fa47193af54def021e6e1256a4dab93020cf3465",
        "normalized": "4c7217b49191deb7add9b93da43a76c787dc594f3d0eb7b94ac1dbec4d398883",
    },
    "raw112_d0_s101102": {
        "raw": "e7bad04eee5ce5622880ed03004f290f3f0ca56e65684728ba890b7f01d7da6e",
        "normalized": "940875cf8987a066f66632e447dfe43669c2e90e87ad50187570cec1727cba0d",
    },
}
ARMS = (
    ("raw112_d0", "raw112", 0.0),
    ("raw112_d02", "raw112", 0.2),
    ("event211_d02", "event211", 0.2),
)


def clock(runtime: Path) -> dict[str, object]:
    """Persist the scalar first actual-profile clock once; never extend it on restart."""
    path = runtime / "receipts/first-profile-r1.json"
    with file_lock(path.with_suffix(".lock")):
        if not path.exists():
            first = time.time()
            atomic_write_json(
                path,
                {
                    "first_profile_epoch": first,
                    "fit_deadline_epoch": first + 10800,
                    "wave_deadline_epoch": first + 14400,
                },
                replace=False,
            )
        return wave_limits(runtime)


def wave_limits(runtime: Path) -> dict[str, object]:
    """Derive the original envelope without rewriting ROOT's scalar clock."""
    value = read_json(runtime / "receipts/first-profile-r1.json")
    first = float(value["first_profile_epoch"])
    if not math.isfinite(first) or first > time.time() + 30:
        raise ValueError("first-profile clock is not a finite actual start")
    expected = {
        "first_profile_epoch": first,
        "fit_deadline_epoch": first + 10800,
        "wave_deadline_epoch": first + 14400,
    }
    if any(key in value and value[key] != item for key, item in expected.items()):
        raise ValueError("wave clock attempted to extend the original allocation")
    return expected


def profile_identity(receipt: dict[str, object], cell: dict[str, object]) -> dict[str, object]:
    """Admit30 successful native updates with exact config/source/device/fit ancestry."""
    identity = receipt["identity"]
    data = identity["data"]
    devices = identity["runtime"]["devices"]
    if (
        receipt.get("status") != "profile_complete"
        or receipt.get("updates") != 30
        or receipt.get("errors") != []
        or receipt.get("scientific_checkpoint_written") is not False
        or identity["source_sha256"] != SOURCE_SHA
        or identity["config_sha256"] != cell["config_sha256"]
        or identity.get("world_size") != 1
        or len(devices) != 1
        or devices[0]["uuid"].removeprefix("GPU-") != cell["gpu_uuid"].removeprefix("GPU-")
        or data["split_sha256"] != SPLIT_SHA
        or len(data["fit_members"]) != 10325
        or any(member[0] != "train" for member in data["fit_members"])
        or len(data["evaluation_members"]) != 1873
        or data.get("teacher") is not None
        or data["normalization"]["scope"] != "all_fit"
        or data["normalization"]["sequence_count"] != 10325
        or identity["data_sha256"] != canonical_sha256(data)
    ):
        raise ValueError("native profile differs from the exact fresh-cell contract")
    return identity


def full_valid(output: Path, config_sha256: str) -> None:
    """Fail closed unless all six native full-validation events have exact coverage."""
    events = [
        json.loads(line)
        for line in (output / "monitor_evaluations.jsonl").read_bytes().splitlines()
    ]
    if len(events) != 6:
        raise ValueError("six-epoch fit lacks six full-validation events")
    for epoch, event in enumerate(events, start=1):
        metrics = event["metrics"]
        if (
            event["kind"] != "evaluation"
            or event["epoch"] != epoch
            or metrics["valid/sequence_count"] != 1873
            or metrics["valid/selected_rows"] != 3528273
        ):
            raise ValueError("full-validation epoch/member/row coverage differs")
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in metrics.values()
        ):
            raise ValueError("full-validation metrics contain nonfinite values")
    terminal = read_json(output / "terminal_status.json")
    identity = terminal["identity"]
    if (
        terminal.get("status") != "completed"
        or terminal.get("completed_epochs") != 6
        or terminal.get("requested_epochs") != 6
        or terminal.get("global_updates") != 38760
        or terminal.get("completion_kind") != "max_epochs"
        or identity.get("config_sha256") != config_sha256
        or identity.get("source_sha256") != SOURCE_SHA
        or file_sha256(output / "model.pt") != terminal.get("model_sha256")
    ):
        raise ValueError("terminal six-epoch state or actual checkpoint SHA/identity differs")
    payload = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    if (
        payload.get("checkpoint_schema") != "wnn.sequence-fit.v1"
        or payload.get("completed_epochs") != 6
        or payload.get("epoch") != 5
        or payload.get("provenance", {}).get("identity") != identity
    ):
        raise ValueError("actual model checkpoint is not the qualified fixed6E endpoint")


def directory_bytes(root: Path) -> int:
    """Count only physical owned output files, without traversing historical symlinks."""
    total = 0
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = [n for n in directories if not (Path(folder) / n).is_symlink()]
        for name in files:
            path = Path(folder) / name
            if not path.is_symlink():
                with suppress(FileNotFoundError):
                    total += path.stat().st_size
    return total


class Guard:
    """Rate-limit resource checks; canonical owned-command cleanup handles each failure."""

    def __init__(self, runtime: Path, control: Path) -> None:
        self.runtime, self.control = runtime, control
        self.lock = threading.RLock()
        self.last = 0.0
        self.identities: dict[int, int] = {}
        self.error: str | None = None

    def check(self, stage: str) -> None:
        """Bound fitting to3h and collection/readouts to4h, with exact owned RSS checks."""
        bound = wave_limits(self.runtime)
        deadline = bound["wave_deadline_epoch" if stage == "readout" else "fit_deadline_epoch"]
        if time.time() >= deadline:
            raise TimeoutError("original wave deadline/reserve reached")
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if time.monotonic() - self.last < 10:
                return
            self.last = time.monotonic()
            for path in (self.control / "owned").glob("*.json"):
                record = read_json(path)
                child = record.get("child")
                if isinstance(child, dict) and is_same_process(child):
                    self.identities[int(child["pid"])] = int(child["birth"])
            processes = tracked_processes(process_table(), self.identities)
            rss = sum(item.resident_bytes for item in processes.values() if item.is_alive)
            available = next(
                int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
            owned = (self.runtime / "runs").resolve()
            snapshot = {
                "at_epoch": time.time(),
                "output_bytes": directory_bytes(owned),
                "output_free_bytes": shutil.disk_usage(owned).free,
                "data2_free_bytes": shutil.disk_usage(self.runtime).free,
                "owned_rss_bytes": rss,
                "host_available_bytes": available,
            }
            if (
                snapshot["output_bytes"] > 4_000_000_000
                or snapshot["output_free_bytes"] < 3_000_000_000
                or snapshot["data2_free_bytes"] < 3_000_000_000
                or rss > 128 * 1024**3
                or available < 64 * 1024**3
            ):
                self.error = "wave output/disk/RSS/page-cache-headroom resource boundary reached"
                raise RuntimeError(self.error)
            atomic_write_json(self.control / "resource_status.json", snapshot)
