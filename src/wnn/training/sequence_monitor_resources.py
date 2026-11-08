from __future__ import annotations

import csv
import io
import math
import os
import re
import shutil
import subprocess
import time
from contextlib import suppress
from pathlib import Path

import torch

NVIDIA_UUID = r"(?:[0-9A-Fa-f]{32}|[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12})"
GPU_IDENTIFIER = re.compile(
    rf"(?:GPU-{NVIDIA_UUID}|MIG-(?:GPU-)?{NVIDIA_UUID}(?:/\d{{1,4}}/\d{{1,4}})?)\Z"
)
NVIDIA_QUERY = "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
NVIDIA_TIMEOUT_SECONDS = 1.5
MAX_NVIDIA_OUTPUT_CHARS = 4_096
MAX_PROC_FILE_BYTES = 65_536
MIB_BYTES = 1_048_576


def _read_proc_file(path: str) -> bytes | None:
    """Read a small Linux proc entry, returning None when procfs is unavailable."""
    try:
        with open(path, "rb") as handle:
            return handle.read(MAX_PROC_FILE_BYTES)
    except OSError:
        return None


def _parse_rss_bytes(status: bytes | None, statm: bytes | None) -> int | None:
    """Read resident memory from status, falling back to resident pages in statm."""
    if status is not None:
        for line in status.decode("ascii", errors="ignore").splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) == 3 and fields[2] == "kB" and fields[1].isdigit():
                    return int(fields[1]) * 1_024
                break
    if statm is None:
        return None
    fields = statm.decode("ascii", errors="ignore").split()
    if len(fields) < 2 or not fields[1].isdigit():
        return None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
    return int(fields[1]) * page_size if page_size > 0 else None


def _parse_host_memory(meminfo: bytes | None) -> dict[str, int]:
    """Extract available and total host RAM from Linux meminfo when present."""
    if meminfo is None:
        return {}
    wanted = {"MemAvailable": "cpu/memory_available_bytes", "MemTotal": "cpu/memory_total_bytes"}
    values: dict[str, int] = {}
    for line in meminfo.decode("ascii", errors="ignore").splitlines():
        name, separator, payload = line.partition(":")
        metric = wanted.get(name)
        if metric is None or not separator:
            continue
        fields = payload.split()
        if len(fields) == 2 and fields[0].isdigit() and fields[1] == "kB":
            values[metric] = int(fields[0]) * 1_024
    return values


def _finite_number(value: str) -> float | None:
    """Parse one finite sensor value while treating unavailable text as absent."""
    try:
        parsed = float(value.strip())
    except (AttributeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _query_gpu(gpu_identifier: str | None) -> dict[str, int | float]:
    """Read NVIDIA sensors only for a validated physical GPU or MIG UUID."""
    if gpu_identifier is None or not GPU_IDENTIFIER.fullmatch(gpu_identifier):
        return {}
    command = [
        "nvidia-smi",
        "--id",
        gpu_identifier,
        f"--query-gpu={NVIDIA_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=NVIDIA_TIMEOUT_SECONDS,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    output = result.stdout
    if not isinstance(output, str) or len(output) > MAX_NVIDIA_OUTPUT_CHARS:
        return {}
    try:
        rows = list(csv.reader(io.StringIO(output)))
    except csv.Error:
        return {}
    if len(rows) != 1 or len(rows[0]) != 5:
        return {}
    utilization, memory_used, memory_total, power, temperature = (
        _finite_number(value) for value in rows[0]
    )
    metrics: dict[str, int | float] = {}
    if utilization is not None and 0 <= utilization <= 100:
        metrics["gpu/utilization_pct"] = utilization
    if memory_used is not None and memory_used >= 0:
        metrics["gpu/device_memory_used_bytes"] = round(memory_used * MIB_BYTES)
    if memory_total is not None and memory_total > 0:
        metrics["gpu/device_memory_total_bytes"] = round(memory_total * MIB_BYTES)
    if power is not None and power >= 0:
        metrics["gpu/power_w"] = power
    if temperature is not None:
        metrics["gpu/temperature_c"] = temperature
    return metrics


class HostResourceSampler:
    """Collect bounded host, output-disk, and UUID-addressed GPU telemetry."""

    def __init__(self, output: Path) -> None:
        """Create a sampler for the filesystem containing a run's output."""
        self.output = Path(output)
        self._previous_process_time: float | None = None
        self._previous_monotonic: float | None = None

    def sample(self, gpu_identifier: str | None = None) -> dict[str, int | float]:
        """Return available resource readings without failing training setup."""
        metrics: dict[str, int | float] = {}
        process_now = time.process_time()
        monotonic_now = time.monotonic()
        if self._previous_process_time is not None and self._previous_monotonic is not None:
            elapsed = monotonic_now - self._previous_monotonic
            consumed = process_now - self._previous_process_time
            if elapsed > 0 and consumed >= 0:
                utilization = consumed / elapsed * 100
                if math.isfinite(utilization):
                    metrics["cpu/process_utilization_pct"] = utilization
        self._previous_process_time = process_now
        self._previous_monotonic = monotonic_now

        rss = _parse_rss_bytes(
            _read_proc_file("/proc/self/status"), _read_proc_file("/proc/self/statm")
        )
        if rss is not None:
            metrics["cpu/rss_bytes"] = rss
        metrics.update(_parse_host_memory(_read_proc_file("/proc/meminfo")))
        try:
            disk = shutil.disk_usage(self.output)
            if disk.free >= 0:
                metrics["disk/free_bytes"] = disk.free
            if disk.total > 0 and 0 <= disk.used <= disk.total:
                metrics["disk/used_pct"] = disk.used / disk.total * 100
        except OSError:
            pass
        metrics.update(_query_gpu(gpu_identifier))
        return metrics


def resource_metrics(device: torch.device) -> dict[str, int | float]:
    """Return only process and device resource readings that the host exposes."""
    metrics: dict[str, int | float] = {"cpu/process_time_seconds": time.process_time()}
    if device.type == "cuda" and torch.cuda.is_available():
        with suppress(AssertionError, RuntimeError):
            metrics.update(
                {
                    "gpu/allocated_bytes": torch.cuda.memory_allocated(device),
                    "gpu/reserved_bytes": torch.cuda.memory_reserved(device),
                    "gpu/peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "gpu/peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }
            )
    return metrics
