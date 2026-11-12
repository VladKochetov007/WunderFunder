from __future__ import annotations

import csv
import io
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_PROCESSES = 16_384
MAX_NVIDIA_OUTPUT = 131_072
MIB = 1_048_576


@dataclass(frozen=True)
class ProcessReading:
    """Bind Linux liveness to a PID's start tick so PID reuse is never adopted."""

    pid: int
    parent_pid: int
    start_ticks: int
    state: str
    cpu_ticks: int
    resident_bytes: int

    @property
    def is_alive(self) -> bool:
        """Exclude exited zombies from the live-process count."""
        return self.state not in ("Z", "X", "x")


def read_process(pid: int, proc_root: Path = Path("/proc")) -> ProcessReading | None:
    """Read process identity and resources without signals or command-line disclosure."""
    try:
        payload = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
        # comm may contain spaces or parentheses; fields after the last ')' are stable.
        fields = payload[payload.rindex(")") + 2 :].split()
        return ProcessReading(
            pid,
            int(fields[1]),
            int(fields[19]),
            fields[0],
            int(fields[11]) + int(fields[12]),
            max(0, int(fields[21])) * int(os.sysconf("SC_PAGE_SIZE")),
        )
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def process_table(proc_root: Path = Path("/proc")) -> dict[int, ProcessReading]:
    """Read a bounded process table for explicit descendant discovery."""
    result: dict[int, ProcessReading] = {}
    try:
        with os.scandir(proc_root) as entries:
            for count, entry in enumerate(entries):
                if count >= MAX_PROCESSES:
                    raise RuntimeError("observer process scan exceeded its explicit entry bound")
                if entry.name.isdigit():
                    reading = read_process(int(entry.name), proc_root)
                    if reading is not None:
                        result[reading.pid] = reading
    except OSError as exc:
        raise RuntimeError(
            "independent process observation requires readable Linux procfs"
        ) from exc
    return result


def tracked_processes(
    table: dict[int, ProcessReading], identities: dict[int, int]
) -> dict[int, ProcessReading]:
    """Keep bound identities and discover their children, including torchrun ranks."""
    tracked = {
        pid: reading
        for pid, ticks in identities.items()
        if (reading := table.get(pid)) is not None and reading.start_ticks == ticks
    }
    for _ in range(len(table)):
        children = {
            pid: reading
            for pid, reading in table.items()
            if pid not in identities
            and reading.parent_pid in tracked
            and reading.start_ticks >= tracked[reading.parent_pid].start_ticks
        }
        if not children:
            break
        tracked.update(children)
        identities.update({pid: reading.start_ticks for pid, reading in children.items()})
    return tracked


def _query_nvidia(query: str, domain: str) -> list[list[str]]:
    """Query all visible devices or compute applications with a bounded timeout."""
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{domain}={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if len(result.stdout) > MAX_NVIDIA_OUTPUT:
        return []
    try:
        return list(csv.reader(io.StringIO(result.stdout)))
    except csv.Error:
        return []


def _number(value: str) -> float | None:
    try:
        number = float(value.strip())
    except ValueError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def gpu_resources(tracked_pids: set[int]) -> tuple[dict[str, int | float], dict[str, object]]:
    """Return all GPU sensors and UUID/PID mappings, preserving foreign processes."""
    metrics: dict[str, int | float] = {}
    devices: list[dict[str, object]] = []
    query = "index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
    for row in _query_nvidia(query, "gpu"):
        if len(row) != 8 or not row[0].strip().isdigit():
            continue
        index = int(row[0])
        devices.append({"index": index, "uuid": row[1].strip(), "name": row[2].strip()})
        for key, value, scale in zip(
            (
                "utilization_pct",
                "memory_used_bytes",
                "memory_total_bytes",
                "power_w",
                "temperature_c",
            ),
            row[3:],
            (1, MIB, MIB, 1, 1),
            strict=True,
        ):
            number = _number(value)
            if number is not None:
                metrics[f"gpu/device_{index}/{key}"] = number * scale
    processes: list[dict[str, object]] = []
    for row in _query_nvidia("gpu_uuid,pid,used_gpu_memory", "compute-apps"):
        if len(row) != 3 or not row[1].strip().isdigit():
            continue
        pid = int(row[1])
        memory = _number(row[2])
        processes.append(
            {
                "gpu_uuid": row[0].strip(),
                "pid": pid,
                "tracked": pid in tracked_pids,
                "memory_used_bytes": None if memory is None else round(memory * MIB),
            }
        )
    metrics["observer/gpu_device_count"] = len(devices)
    metrics["observer/gpu_process_count"] = len(processes)
    metrics["observer/gpu_query_available"] = int(bool(devices))
    return metrics, {
        "devices": devices,
        "gpu_processes": processes,
        "gpu_query_status": "available" if devices else "unavailable or no visible devices",
    }


def host_resources(output: Path) -> dict[str, int | float]:
    """Read host RAM and output-filesystem headroom without importing the trainer."""
    metrics: dict[str, int | float] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    except OSError:
        lines = []
    for line in lines:
        fields = line.split()
        if len(fields) == 3 and fields[0] in ("MemTotal:", "MemAvailable:"):
            key = "total" if fields[0] == "MemTotal:" else "available"
            if fields[1].isdigit() and fields[2] == "kB":
                metrics[f"cpu/memory_{key}_bytes"] = int(fields[1]) * 1024
    try:
        disk = shutil.disk_usage(output if output.exists() else output.parent)
    except OSError:
        return metrics
    metrics["disk/output_free_bytes"] = disk.free
    metrics["disk/output_total_bytes"] = disk.total
    return metrics
