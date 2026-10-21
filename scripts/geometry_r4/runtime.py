"""Cgroup/GPU telemetry and bounded subprocess execution for R4."""

from __future__ import annotations

import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .contract import (
    MIN_GPU_FREE_MIB,
    PROGRESS,
    RESOURCE_SAMPLE_SECONDS,
    RESOURCES,
    ROOT,
    append_jsonl,
    command_environment,
    now_utc,
    verify_frozen_inputs,
)


# Cgroup and GPU telemetry are emitted as schema-versioned JSON values.
def cgroup_directory() -> Path:
    """Resolve this service's cgroup v2 directory."""
    entries = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    relative = next((line.split("::", 1)[1] for line in entries if line.startswith("0::")), None)
    if relative is None:
        raise RuntimeError("unified cgroup v2 path is unavailable")
    directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
    if not (directory / "memory.max").is_file():
        raise RuntimeError(f"cgroup memory controller is unavailable at {directory}")
    return directory


def cgroup_values(directory: Path) -> dict[str, Any]:
    """Read bounded memory and CPU evidence for this research service."""
    result: dict[str, Any] = {"cgroup": str(directory)}
    for name in ("memory.current", "memory.peak", "memory.max", "memory.swap.max", "cpu.max"):
        path = directory / name
        if path.is_file():
            result[name.replace(".", "_")] = path.read_text(encoding="utf-8").strip()
    events_path = directory / "memory.events"
    if events_path.is_file():
        result["memory_events"] = {
            key: int(value)
            for key, value in (line.split() for line in events_path.read_text().splitlines())
        }
    return result


def gpu_values() -> dict[str, str]:
    """Read concise GPU utilization, temperature and memory values."""
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    columns = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
    if len(columns) != 6:
        raise RuntimeError(f"unexpected nvidia-smi output: {completed.stdout.strip()}")
    keys = (
        "timestamp",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_pct",
        "temperature_c",
        "power_w",
    )
    return dict(zip(keys, columns, strict=True))


def validate_service_budget(directory: Path) -> None:
    """Require the declared 5 GiB, one-core, no-swap cgroup before fitting."""
    values = cgroup_values(directory)
    if int(values["memory_max"]) > 5_000_000_000:
        raise RuntimeError(f"R4 service memory cap exceeds 5 GB: {values['memory_max']}")
    if values["memory_swap_max"] != "0":
        raise RuntimeError("R4 service must have swap disabled")
    quota, period = (int(value) for value in values["cpu_max"].split())
    if quota > period:
        raise RuntimeError(f"R4 service CPU quota exceeds one core: {values['cpu_max']}")


def sample_resources(run_name_value: str, phase: str, directory: Path) -> dict[str, Any]:
    """Append one bounded RAM/CPU/GPU sample to the resource ledger."""
    record = {
        "timestamp_utc": now_utc(),
        "run": run_name_value,
        "phase": phase,
        **cgroup_values(directory),
        "gpu": gpu_values(),
    }
    append_jsonl(RESOURCES, record)
    return record


def execute_phase(
    run_name_value: str,
    phase: str,
    command: list[str],
    log_path: Path,
    timeout_seconds: int,
    directory: Path,
) -> None:
    """Run one bounded phase, retaining logs and live resource telemetry."""
    verify_frozen_inputs()
    baseline_events = cgroup_values(directory).get("memory_events", {})
    gpu_before = gpu_values()
    free_mib = int(gpu_before["memory_total_mib"]) - int(gpu_before["memory_used_mib"])
    if free_mib < MIN_GPU_FREE_MIB:
        raise RuntimeError(f"only {free_mib} MiB of GPU memory is free before {phase}")
    append_jsonl(
        PROGRESS,
        {
            "timestamp_utc": now_utc(),
            "event": f"{phase}_start",
            "run": run_name_value,
            "command": command,
            "log": str(log_path.relative_to(ROOT)),
            "gpu_before": gpu_before,
            "memory_events_before": baseline_events,
        },
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_sample = 0.0
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=command_environment(),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                    raise TimeoutError(f"{phase} exceeded {timeout_seconds} seconds")
                if elapsed - last_sample >= RESOURCE_SAMPLE_SECONDS:
                    sample_resources(run_name_value, phase, directory)
                    last_sample = elapsed
                time.sleep(5)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=30)
        status = process.returncode
    final = cgroup_values(directory)
    final_events = final.get("memory_events", {})
    for key in ("oom", "oom_kill", "oom_group_kill"):
        if int(final_events.get(key, 0)) > int(baseline_events.get(key, 0)):
            raise RuntimeError(f"service {key} counter increased during {run_name_value}")
    elapsed = round(time.monotonic() - started, 1)
    append_jsonl(
        PROGRESS,
        {
            "timestamp_utc": now_utc(),
            "event": f"{phase}_complete" if status == 0 else f"{phase}_failed",
            "run": run_name_value,
            "exit_status": status,
            "elapsed_seconds": elapsed,
            "service_final": final,
        },
    )
    print(f"{phase} {run_name_value}: exit={status}, elapsed={elapsed:.1f}s", flush=True)
    if status != 0:
        raise RuntimeError(f"{phase} failed for {run_name_value}; see {log_path}")
