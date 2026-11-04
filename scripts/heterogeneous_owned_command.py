"""Run one reviewed command durably, without retries or resource allocation."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import resource
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

# Requests contain heterogeneous, caller-reviewed source identities; never executable JSON.
JsonObject = dict[str, Any]
ENVIRONMENT_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "CUDA_MODULE_LOADING",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUBLAS_EMULATE_SINGLE_PRECISION",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "WNN_PROJECT_ROOT",
    "WNN_THREADS",
    "WNN_LOADER_WORKERS",
    "WNN_LOADER_RAM_GB",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ARROW_NUM_THREADS",
    "MALLOC_ARENA_MAX",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "PYTORCH_CUDA_ALLOC_CONF",
    "PYTORCH_ALLOC_CONF",
}


def utc_now() -> str:
    """Return an explicit UTC timestamp for durable phase records."""
    return datetime.datetime.now(datetime.UTC).isoformat()


def write_status(path: Path, value: JsonObject) -> None:
    """Replace only this command's status atomically after flushing its bytes."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def start_ticks(pid: int) -> str | None:
    """Read the Linux process birth identity, handling names containing spaces."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def stop_owned(child: subprocess.Popen[bytes], ticks: str | None) -> None:
    """Stop only the live process group created for this exact child instance."""
    if child.poll() is not None:
        return
    if ticks is None or start_ticks(child.pid) != ticks or os.getpgid(child.pid) != child.pid:
        raise RuntimeError("refusing to signal a process without matching child ownership")
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if start_ticks(child.pid) != ticks or os.getpgid(child.pid) != child.pid:
            raise RuntimeError("child identity changed before bounded SIGKILL") from None
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=30)
    except ProcessLookupError:
        child.wait(timeout=30)


def run(request_path: Path, output: Path) -> int:
    """Execute one fixed request and retain running/terminal ownership and timing."""
    raw = request_path.read_bytes()
    request = json.loads(raw)
    command, environment = request.get("command"), request.get("env", {})
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(x, str) or not x for x in command)
    ):
        raise ValueError("command must be a nonempty argument list")
    if (
        not isinstance(environment, dict)
        or set(environment) - ENVIRONMENT_KEYS
        or any(not isinstance(v, str) for v in environment.values())
    ):
        raise ValueError("env must contain only approved non-sensitive string variables")
    wall_seconds = request.get("wall_seconds")
    if type(wall_seconds) not in (float, int) or not 0 < wall_seconds < float("inf"):
        raise ValueError("wall_seconds must be finite and positive")
    cwd, log = Path(request["cwd"]).resolve(strict=True), Path(request["log"]).resolve()
    if not cwd.is_dir() or output.exists() or log.exists():
        raise ValueError("cwd must exist and status/log paths must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    started, child, ticks = time.monotonic(), None, None
    status = {
        "schema": "wnn.owned-command.v1",
        "status": "launching",
        "started_utc": utc_now(),
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "command_sha256": hashlib.sha256(
            json.dumps(command, separators=(",", ":")).encode()
        ).hexdigest(),
        "phase": request["phase"],
        "role": request["role"],
        "source_identity": request["source_identity"],
        "gpu_uuid": request.get("gpu_uuid"),
        "wall_seconds": wall_seconds,
        "cwd": str(cwd),
        "log": str(log),
        "environment": environment,
        "runner_pid": os.getpid(),
        "runner_start_ticks": start_ticks(os.getpid()),
    }
    write_status(output, status)

    def interrupted(signum: int, frame: object) -> None:
        del frame
        raise InterruptedError(f"runner received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with log.open("xb", buffering=0) as stream:
            child = subprocess.Popen(
                command,
                cwd=cwd,
                env=os.environ | environment,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            ticks = start_ticks(child.pid)
            try:
                process_group, session = os.getpgid(child.pid), os.getsid(child.pid)
            except ProcessLookupError:
                process_group, session = None, None
            status.update(
                status="running",
                pid=child.pid,
                start_ticks=ticks,
                process_group=process_group,
                session=session,
                command_started_utc=utc_now(),
            )
            write_status(output, status)
            try:
                code = child.wait(timeout=max(0.001, wall_seconds - (time.monotonic() - started)))
                status.update(status="complete" if code == 0 else "failed", return_code=code)
            except (subprocess.TimeoutExpired, InterruptedError) as error:
                stop_owned(child, ticks)
                status.update(
                    status="timed_out"
                    if isinstance(error, subprocess.TimeoutExpired)
                    else "interrupted",
                    return_code=child.returncode,
                    error="wall-clock allocation exhausted"
                    if isinstance(error, subprocess.TimeoutExpired)
                    else str(error),
                )
    except (OSError, ValueError, RuntimeError, InterruptedError) as error:
        if child is not None:
            try:
                stop_owned(child, ticks)
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as stop_error:
                status["ownership_stop_error"] = f"{type(stop_error).__name__}: {stop_error}"
        status.update(
            status="failed",
            return_code=None if child is None else child.returncode,
            error=f"{type(error).__name__}: {error}",
        )
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    status.update(
        finished_utc=utc_now(),
        elapsed_seconds=time.monotonic() - started,
        child_max_rss_kib=usage.ru_maxrss,
        child_user_seconds=usage.ru_utime,
        child_system_seconds=usage.ru_stime,
    )
    write_status(output, status)
    return 0 if status["status"] == "complete" else 1


def main() -> None:
    """Run exactly one explicit request; the caller owns detachment and admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.request, args.output))


if __name__ == "__main__":
    main()
