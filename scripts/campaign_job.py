"""Durable, resource-limited campaign jobs with GPU wall-time accounting."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "research" / "campaign_2026-09-29"
JOBS = CAMPAIGN / "jobs"
GPU_CAP = 40_000.0  # 2026-10-01: user allowed continuous GPU use until 10:00 Kyiv on top of 10,877 s charged
VRAM_CAP = 5_000_000_000
STOP_NEW_JOBS = CAMPAIGN / "STOP_NEW_JOBS"


def request_stop(reason: str) -> None:
    """Persist the launch barrier until an intentional campaign resumption."""
    try:
        with STOP_NEW_JOBS.open("x") as marker:
            marker.write(f"{timestamp()}: {reason}\n")
            marker.flush()
            os.fsync(marker.fileno())
    except FileExistsError:
        pass


def check_launch_allowed() -> None:
    """Reject launches and delayed service starts while the barrier exists."""
    if STOP_NEW_JOBS.exists():
        raise RuntimeError(f"campaign launch blocked by {STOP_NEW_JOBS.relative_to(ROOT)}")


def timestamp() -> str:
    """Return a UTC timestamp for auditable job records."""
    return datetime.now(UTC).isoformat()


def write_record(path: Path, record: dict) -> None:
    """Atomically publish a job's latest durable accounting record."""
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2) + "\n")
    temp.replace(path)


def records() -> list[dict]:
    """Read accounting records without discarding malformed evidence."""
    return [json.loads(path.read_text()) for path in sorted(JOBS.glob("*.json"))]


def source_fingerprints() -> dict[str, str]:
    """Record the exact research/runtime source bytes seen when a job starts."""
    paths = [
        *sorted((ROOT / "src" / "wnn").glob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "submission").glob("*.py")),
        *sorted((ROOT / "native" / "src").glob("*.rs")),
        ROOT / "scripts" / "limited",
        ROOT / "scripts" / "ft_gclp.sh",
        ROOT / "utils.py",
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }


def charged_seconds(record: dict) -> float:
    """Conservatively account jobs whose monitor ended unexpectedly."""
    elapsed = float(record.get("elapsed_seconds", 0))
    if record["status"] == "running":
        active = unit_active(record["unit"])
        if active:
            elapsed = max(elapsed, time.time() - record["started_unix"])
        else:
            elapsed = max(elapsed + 5, float(record.get("timeout_seconds", elapsed)) + 15)
    return elapsed


def budget() -> dict:
    """Return cumulative GPU-job wall time, including failures and active work."""
    jobs = records()
    used = sum(charged_seconds(job) for job in jobs if job["kind"] == "gpu")
    return {
        "gpu_seconds_used": used,
        "gpu_seconds_remaining": max(0.0, GPU_CAP - used),
        "gpu_seconds_cap": GPU_CAP,
        "jobs": jobs,
    }


def device_memory() -> int:
    """Read total observed device memory, including non-PyTorch allocations."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return max(int(line.strip()) for line in result.stdout.splitlines()) * 1024**2


def unit_active(unit: str) -> bool:
    """Check a known campaign unit with a bounded systemctl call."""
    return (
        subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit],
            check=False,
            timeout=5,
        ).returncode
        == 0
    )


def stop_scope(unit: str) -> None:
    """Stop a campaign-owned scope, then confirm it has exited."""
    if unit_active(unit):
        subprocess.run(["systemctl", "--user", "stop", unit], check=True, timeout=10)
    if unit_active(unit):
        raise RuntimeError(f"campaign scope remains active: {unit}")


def stop_child(child: subprocess.Popen) -> None:
    """Reap the leader and stop its group, including surviving descendants."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait(timeout=5)
        return
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        child.poll()
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            child.wait(timeout=2)
            return
        time.sleep(0.1)
    with suppress(ProcessLookupError):
        os.killpg(child.pid, signal.SIGKILL)
    child.wait(timeout=2)


def run_job(args: argparse.Namespace) -> int:
    """Run one job under a durable service and preserve failure evidence."""
    received_signals: list[int] = []

    def on_signal(signum: int, frame: FrameType | None) -> None:
        received_signals.append(signum)
        request_stop(f"{args.name} received {signal.Signals(signum).name}")

    previous_handlers = {
        signum: signal.signal(signum, on_signal) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    check_launch_allowed()
    JOBS.mkdir(parents=True, exist_ok=True)
    lock = (CAMPAIGN / "gpu.lock").open("a+")
    if args.kind == "gpu":
        while True:
            check_launch_allowed()
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.5)
        for old in records():
            scope = old.get("scope_unit")
            if old["kind"] == "gpu" and scope and unit_active(scope):
                raise RuntimeError(f"previous GPU scope remains active: {scope}")
    check_launch_allowed()
    record_path = JOBS / f"{args.name}.json"
    if record_path.exists():
        raise ValueError(f"job evidence already exists: {record_path}")
    remaining = budget()["gpu_seconds_remaining"]
    allowance = min(args.timeout, remaining - 15) if args.kind == "gpu" else args.timeout
    if allowance <= 0:
        raise ValueError("GPU budget exhausted")
    record = {
        "name": args.name,
        "unit": args.unit,
        "scope_unit": f"wnn-top1-{args.name}-scope.scope",
        "phase": args.phase,
        "kind": args.kind,
        "command": args.command,
        "started_at": timestamp(),
        "started_unix": time.time(),
        "timeout_seconds": allowance,
        "status": "running",
        "elapsed_seconds": 0,
        "peak_observed_vram_bytes": 0,
        "source_sha256": source_fingerprints(),
    }
    write_record(record_path, record)
    log_path = CAMPAIGN / "logs" / f"{args.name}.log"
    start = time.monotonic()
    child = None
    failure = None
    try:
        with log_path.open("x") as log:
            if received_signals:
                failure = f"signal:{signal.Signals(received_signals[0]).name}"
            else:
                check_launch_allowed()
            if failure is not None:
                return 1
            child = subprocess.Popen(
                args.command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            while child.poll() is None:
                elapsed = time.monotonic() - start
                if received_signals:
                    failure = f"signal:{signal.Signals(received_signals[0]).name}"
                    break
                if args.kind == "gpu":
                    observed = device_memory()
                    record["peak_observed_vram_bytes"] = max(
                        record["peak_observed_vram_bytes"], observed
                    )
                    if observed > VRAM_CAP:
                        failure = "observed_vram_limit"
                        break
                record["elapsed_seconds"] = elapsed
                write_record(record_path, record)
                if elapsed >= allowance:
                    failure = "time_budget_limit"
                    break
                time.sleep(0.5)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        failure = repr(exc)
    finally:
        if received_signals:
            failure = f"signal:{signal.Signals(received_signals[0]).name}"
        if child is not None:
            try:
                stop_child(child)
            except (OSError, subprocess.SubprocessError) as exc:
                failure = f"{failure}; child_stop={exc!r}"
        try:
            stop_scope(record["scope_unit"])
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            failure = f"{failure}; scope_stop={exc!r}"
        record.update(
            status=(
                "interrupted"
                if received_signals
                else "finished"
                if failure is None and child is not None and child.returncode == 0
                else "failed"
            ),
            finished_at=timestamp(),
            elapsed_seconds=time.monotonic() - start,
            exit_code=None if child is None else child.returncode,
            failure=failure,
            log=str(log_path.relative_to(ROOT)),
        )
        write_record(record_path, record)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        lock.close()
    return 0 if record["status"] == "finished" else 1


def launch(args: argparse.Namespace) -> None:
    """Launch a transient service that remains alive after the shell exits."""
    check_launch_allowed()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.name):
        raise ValueError("job names may contain only letters, digits, underscore and dash")
    unit = f"wnn-top1-{args.name}"
    if (JOBS / f"{args.name}.json").exists():
        raise ValueError("use a new job name; prior evidence is immutable")
    remaining = budget()["gpu_seconds_remaining"]
    allowance = min(args.timeout, remaining - 15) if args.kind == "gpu" else args.timeout
    if allowance <= 0:
        raise ValueError("GPU budget exhausted")
    env = {
        "WNN_SLICE": "wnn-campaign.slice",
        "WNN_CPU_PCT": "75",
        "WNN_MEM_MAX": "36G",
        "WNN_GPU_FRACTION": "0.70",
        "WNN_LOADER_RAM_GB": "6",
        "WNN_LOADER_WORKERS": "4",
        "WNN_THREADS": str(args.threads),
        "WNN_CPU_QUOTA": f"{args.threads * 100}%",
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "NUMBA_NUM_THREADS": str(args.threads),
        "POLARS_MAX_THREADS": str(args.threads),
        "ARROW_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
        "WNN_RUNTIME_MAX": str(math.ceil(allowance + 5)),
        "WNN_SCOPE_UNIT": f"wnn-top1-{args.name}-scope",
    }
    command = [
        "systemd-run",
        "--user",
        f"--unit={unit}",
        "--slice=wnn-campaign.slice",
        "--quiet",
        "-p",
        "MemorySwapMax=0",
        "-p",
        f"RuntimeMaxSec={math.ceil(allowance + 8)}",
        "-p",
        "TimeoutStopSec=15s",
        "-p",
        "KillMode=control-group",
        "-p",
        f"CPUQuota={args.threads * 100}%",
        "-p",
        "MemoryMax=36G",
    ]
    command.extend(f"--setenv={key}={value}" for key, value in env.items())
    command.extend(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "run",
            "--unit",
            unit,
            "--name",
            args.name,
            "--kind",
            args.kind,
            "--phase",
            args.phase,
            "--timeout",
            str(args.timeout),
            "--threads",
            str(args.threads),
            "--",
            *args.command,
        ]
    )
    check_launch_allowed()
    subprocess.run(command, check=True)
    print(json.dumps({"unit": unit, "name": args.name, "command": args.command}))


def main() -> None:
    """Parse launcher, monitor and read-only accounting commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("launch", "run"):
        part = sub.add_parser(action)
        part.add_argument("--name", required=True)
        part.add_argument("--kind", choices=("gpu", "cpu"), required=True)
        part.add_argument("--phase", default="controls")
        part.add_argument("--timeout", type=float, required=True)
        part.add_argument("--threads", type=int, choices=range(1, 14), default=12)
        if action == "run":
            part.add_argument("--unit", required=True)
        part.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.action == "status":
        print(json.dumps(budget(), indent=2))
        return
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("a command and positive timeout are required")
    if args.action == "launch":
        launch(args)
    else:
        raise SystemExit(run_job(args))


if __name__ == "__main__":
    main()
