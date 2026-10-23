"""Run one admitted command with thermal and host-memory cooling pauses."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import NamedTuple


class Reading(NamedTuple):
    """Host observations used by the single-child admission guard."""

    temperature: int
    available_bytes: int
    memory_stall: float
    cpu_package_temperature: float | None = None


CPU_HWMON_ROOT = Path("/sys/class/hwmon")


def pause_reason(reading: Reading, paused: bool) -> str | None:
    """Apply hysteresis so a warm or pressured laptop gets a cooling interval."""
    if reading.temperature >= (75 if paused else 80):
        return "temperature"
    if reading.cpu_package_temperature is not None:
        cpu_is_hot = (
            reading.cpu_package_temperature > 90.0
            if paused
            else reading.cpu_package_temperature >= 97.0
        )
        if cpu_is_hot:
            return "cpu_package_temperature"
    if reading.available_bytes < (10_000_000_000 if paused else 8_000_000_000):
        return "host_available_memory"
    if reading.memory_stall >= (0.2 if paused else 2.0):
        return "host_memory_pressure"
    return None


def read_cpu_package_temperature(hwmon_root: Path | None = None) -> float:
    """Return the hottest readable Linux hwmon CPU package temperature in Celsius."""
    root = CPU_HWMON_ROOT if hwmon_root is None else hwmon_root
    temperatures: list[float] = []
    for label_path in sorted(root.glob("hwmon*/temp*_label")):
        try:
            label = label_path.read_text().strip()
        except OSError as error:
            raise RuntimeError(f"cannot read CPU temperature sensor label {label_path}") from error
        if not label.startswith("Package id "):
            continue
        input_path = label_path.with_name(f"{label_path.name.removesuffix('_label')}_input")
        try:
            temperatures.append(int(input_path.read_text().strip()) / 1000.0)
        except (OSError, ValueError) as error:
            raise RuntimeError(f"cannot read CPU package temperature {input_path}") from error
    if not temperatures:
        raise RuntimeError(
            f"CPU package temperature guard is enabled, but no readable Package id sensor was found under {root}"
        )
    return max(temperatures)


def read_resources(*, include_cpu_package_temperature: bool = False) -> Reading:
    """Read GPU0, optional CPU-package temperature and host pressure without a GPU framework."""
    cpu_temperature = read_cpu_package_temperature() if include_cpu_package_temperature else None
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    available = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    full = next(
        line
        for line in Path("/proc/pressure/memory").read_text().splitlines()
        if line.startswith("full ")
    )
    average = float(next(field[6:] for field in full.split() if field.startswith("avg10=")))
    return Reading(int(result.stdout.strip()), available, average, cpu_temperature)


def write_report(path: Path, report: dict[str, object]) -> None:
    """Replace the mutable operational receipt without exposing a partial JSON file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def run_command(spec_path: Path, *, cpu_package_temperature_guard: bool = False) -> int:
    """Execute exactly one explicit argv while preserving its numeric recipe."""
    spec = json.loads(spec_path.read_text())
    if set(spec) != {"argv", "cwd", "report"}:
        raise ValueError("guard spec requires only argv, cwd and report")
    argv = spec["argv"]
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) or "\0" in value for value in argv)
    ):
        raise ValueError("command argv must contain explicit non-NUL strings")
    report_path = Path(spec["report"]).resolve()
    if report_path.exists() or not report_path.parent.is_dir():
        raise ValueError("guard report must be new with an existing parent")
    started = time.monotonic()
    report: dict[str, object] = {
        "spec": str(spec_path.resolve()),
        "argv": argv,
        "status": "waiting_for_safe_start",
        "events": [],
        "paused_seconds": 0.0,
    }
    events: list[dict[str, object]] = []
    report["events"] = events
    write_report(report_path, report)
    try:
        while pause_reason(
            read_resources(include_cpu_package_temperature=cpu_package_temperature_guard), True
        ):
            time.sleep(1)
    except RuntimeError as error:
        report.update(status="guard_failed", error=str(error))
        write_report(report_path, report)
        raise
    child = subprocess.Popen(argv, cwd=spec["cwd"], start_new_session=True)
    report.update(status="running", pid=child.pid)
    write_report(report_path, report)
    paused = False
    pause_started = 0.0
    paused_seconds = 0.0
    last_heartbeat = 0.0
    try:
        while child.poll() is None:
            reading = read_resources(include_cpu_package_temperature=cpu_package_temperature_guard)
            reason = pause_reason(reading, paused)
            if bool(reason) != paused:
                os.killpg(child.pid, signal.SIGSTOP if reason else signal.SIGCONT)
                if reason:
                    pause_started = time.monotonic()
                else:
                    paused_seconds += time.monotonic() - pause_started
                paused = bool(reason)
                event = {
                    "event": "paused" if paused else "resumed",
                    "reason": reason,
                    "elapsed_seconds": time.monotonic() - started,
                    "temperature_celsius": reading.temperature,
                    "available_bytes": reading.available_bytes,
                    "host_memory_stall_avg10": reading.memory_stall,
                }
                if cpu_package_temperature_guard:
                    event["cpu_package_temperature_celsius"] = reading.cpu_package_temperature
                events.append(event)
                print(json.dumps(event), flush=True)
                report["status"] = "cooling" if paused else "running"
                write_report(report_path, report)
            if time.monotonic() - last_heartbeat >= 60:
                snapshot = reading._asdict()
                if not cpu_package_temperature_guard:
                    snapshot.pop("cpu_package_temperature")
                print(json.dumps({"guard": snapshot, "paused": paused}), flush=True)
                last_heartbeat = time.monotonic()
            time.sleep(1)
        if paused:
            paused_seconds += time.monotonic() - pause_started
        report.update(
            status="complete" if child.returncode == 0 else "failed",
            returncode=child.returncode,
            wall_seconds=time.monotonic() - started,
            paused_seconds=paused_seconds,
        )
    except BaseException as error:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGCONT)
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        report.update(status="guard_failed", error=str(error), returncode=child.returncode)
        raise
    finally:
        write_report(report_path, report)
    return int(child.returncode)


def main() -> None:
    """Run the root-admitted single command described by a small JSON spec."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--cpu-package-temperature-guard",
        action="store_true",
        help="pause at 97 C and resume at or below 90 C using CPU package hwmon sensors",
    )
    args = parser.parse_args()
    raise SystemExit(
        run_command(args.spec, cpu_package_temperature_guard=args.cpu_package_temperature_guard)
    )


if __name__ == "__main__":
    main()
