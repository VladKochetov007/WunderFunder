from __future__ import annotations

import fcntl
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
STATE = Path(__file__).resolve().parent
JOBS = ROOT / "research/campaign_2026-09-29/jobs"
FAULT = re.compile(
    r"oom[-_ ]kill|out of memory|killed process|NVRM.*Xid|failed with result|timed out", re.I
)


def event_id(event: dict[str, Any]) -> str:
    """Identify incidents independently of monitor restart time."""
    content = {key: value for key, value in event.items() if key not in ("time", "id")}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Publish complete metadata."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def capture(*args: str) -> str:
    """Failed inspection must never be interpreted as an idle system."""
    result = subprocess.run(args, text=True, capture_output=True, timeout=10)
    # journalctl --grep uses exit1 for a successfully inspected empty interval.
    if result.returncode and not (
        args[0] == "journalctl" and result.returncode == 1 and not result.stderr.strip()
    ):
        raise subprocess.CalledProcessError(result.returncode, args, result.stdout, result.stderr)
    return result.stdout


def record_incident(event: dict[str, Any]) -> None:
    """Persist evidence and serialize failure publication with CPU admission."""
    with (STATE / "locks/admission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (STATE / "logs/incidents.jsonl").open("a") as output:
            output.write(json.dumps(event) + "\n")
            output.flush()
        atomic_json(STATE / "RESOURCE_FAILURE", event)
        (ROOT / "research/campaign_2026-10-01/QUEUE_PAUSE").write_text(
            "Watcher: diagnose and acknowledge incidents before retry.\n"
        )


class Monitor:
    """Durable incidents, journal cursors and aggregate OOM counters."""

    def __init__(self) -> None:
        self.degraded: list[str] = []
        self.seen: set[str] = set()
        self.acknowledged: set[str] = set()
        self.saved = self.read_json(STATE / "watch_memory.json", {})
        self.seen.update(self.saved.get("seen", []))
        self.acknowledged = set(
            self.read_json(STATE / "acknowledgements.json", {}).get("incident_ids", [])
        )
        incidents = STATE / "logs/incidents.jsonl"
        if incidents.exists():
            for line in incidents.read_text().splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("id", event_id(event)) not in self.acknowledged:
                    atomic_json(STATE / "RESOURCE_FAILURE", event)

    def read_json(self, path: Path, default: Any) -> Any:
        """Malformed metadata is preserved as a blocking diagnostic."""
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            event = {"type": "invalid_metadata", "path": str(path), "error": str(error)}
            self.emit(event)
            return default

    def emit(self, event: dict[str, Any]) -> None:
        """Retain each failure once, with explicit acknowledgement required."""
        identity = event_id(event)
        if identity not in self.acknowledged and identity not in self.seen:
            record_incident(
                {**event, "id": identity, "time": datetime.now().astimezone().isoformat()}
            )
            self.seen.add(identity)

    def query(self, *args: str) -> str | None:
        """Expose failed observation as degraded monitoring coverage."""
        try:
            return capture(*args)
        except (OSError, subprocess.SubprocessError) as error:
            self.degraded.append(f"{args!r}: {error}")
            return None

    def journals(self, stream: str, *extra: str) -> None:
        """Consume saved cursors to catch faults in vanished scopes."""
        cursor = self.saved.get(f"{stream}_cursor")
        options = (
            [f"--after-cursor={cursor}"]
            if cursor
            else ["--since", self.saved.get("started", "2 minutes ago")]
        )
        pattern = "wnn-|oom|Out of memory|Killed process|NVRM|Xid"
        output = self.query(
            "journalctl", *extra, *options, "--grep", pattern, "--no-pager", "-o", "json"
        )
        if output is None:
            return
        kept = []
        for line in output.splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                self.degraded.append(f"invalid {stream} journal entry")
                continue
            if entry.get("__CURSOR"):
                self.saved[f"{stream}_cursor"] = entry["__CURSOR"]
            message = str(entry.get("MESSAGE", ""))
            unit = str(entry.get("_SYSTEMD_USER_UNIT", ""))
            if (
                "wnn-" in message
                or "wnn-" in unit
                or (stream == "kernel" and FAULT.search(message))
            ):
                kept.append(line)
                if FAULT.search(message):
                    self.emit(
                        {
                            "type": "journal_fault",
                            "stream": stream,
                            "cursor": entry.get("__CURSOR"),
                            "message": message,
                            "unit": unit,
                        }
                    )
        with (STATE / f"logs/{stream}_events.jsonl").open("a") as output_file:
            output_file.write("".join(line + "\n" for line in kept))
        (STATE / f"logs/{stream}_latest.txt").write_text("\n".join(kept[-80:]))

    def poll(self) -> None:
        """Reconcile running units and persist the next polling boundary."""
        self.degraded = []
        now = datetime.now().astimezone().isoformat()
        active = []
        for path in JOBS.glob("*.json"):
            job = self.read_json(path, {})
            status = job.get("status")
            if status in ("failed", "interrupted") or (
                status == "finished" and job.get("exit_code") != 0
            ):
                self.emit(
                    {
                        "type": "gpu_job_failed",
                        "record": str(path),
                        "failure": job.get("failure"),
                        "exit_code": job.get("exit_code"),
                        "unit": job.get("unit"),
                    }
                )
            if status in ("running", "waiting"):
                active.append(
                    {
                        key: job.get(key)
                        for key in ("name", "unit", "elapsed_seconds", "peak_observed_vram_bytes")
                    }
                )
                unit = str(job.get("unit", ""))
                unit = unit if unit.endswith(".service") else unit + ".service"
                properties = self.query(
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "-p",
                    "LoadState",
                    "-p",
                    "ActiveState",
                    "-p",
                    "Result",
                    "-p",
                    "ExecMainStatus",
                )
                if properties is not None and time.time() - path.stat().st_mtime > 30:
                    values = dict(
                        line.split("=", 1) for line in properties.splitlines() if "=" in line
                    )
                    if values.get("LoadState") == "not-found" or values.get("ActiveState") not in (
                        "active",
                        "activating",
                        "deactivating",
                    ):
                        self.emit(
                            {"type": "stale_running_job", "record": str(path), "properties": values}
                        )
        group = self.query(
            "systemctl", "--user", "show", "wnn-campaign.slice", "-p", "ControlGroup", "--value"
        )
        if group:
            events = Path("/sys/fs/cgroup") / group.strip().lstrip("/") / "memory.events"
            try:
                values = dict(line.split() for line in events.read_text().splitlines())
                kills = int(values.get("oom_kill", 0))
                previous = self.saved.get("slice_oom_kill")
                if previous is not None and kills > previous:
                    self.emit(
                        {"type": "aggregate_oom", "previous": previous, "memory_events": values}
                    )
                self.saved["slice_oom_kill"] = kills
            except (OSError, ValueError) as error:
                self.degraded.append(f"aggregate memory counters: {error}")
        self.journals("journal", "--user")
        self.journals("kernel", "-k")
        resources = self.query(
            "systemctl",
            "--user",
            "show",
            "wnn-campaign.slice",
            "-p",
            "CPUQuotaPerSecUSec",
            "-p",
            "MemoryMax",
            "-p",
            "MemoryCurrent",
        )
        gpu = self.query(
            "nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"
        )
        atomic_json(
            STATE / "watch_status.json",
            {
                "time": now,
                "active_gpu_jobs": active,
                "resources": resources,
                "gpu": gpu,
                "barrier": (STATE / "RESOURCE_FAILURE").exists(),
                "monitoring_degraded": self.degraded,
            },
        )
        if self.degraded:
            with (STATE / "logs/watch_errors.log").open("a") as output:
                output.write(json.dumps({"time": now, "errors": self.degraded}) + "\n")
        self.saved.update({"seen": sorted(self.seen), "started": self.saved.get("started", now)})
        atomic_json(STATE / "watch_memory.json", self.saved)


def main() -> None:
    """Poll each minute; systemd restarts unexpected watcher failures."""
    monitor = Monitor()
    while True:
        monitor.poll()
        time.sleep(60)


if __name__ == "__main__":
    main()
