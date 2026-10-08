from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import psutil


def aggregate_rss(process: psutil.Process) -> int:
    """Conservatively sum RSS for this supervisor and every live descendant."""
    total = 0
    for member in [process, *process.children(recursive=True)]:
        try:
            total += member.memory_info().rss
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return total


def stop_group(child: subprocess.Popen) -> None:
    """Stop every isolated trainer group as well as the worker, with finite grace."""
    try:
        descendants = psutil.Process(child.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    groups = {child.pid}
    for member in descendants:
        try:
            groups.add(os.getpgid(member.pid))
        except ProcessLookupError:
            continue
    groups.discard(os.getpgrp())
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            continue
    with suppress(subprocess.TimeoutExpired):
        child.wait(timeout=10)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            continue
    child.wait(timeout=10)


def main() -> int:
    output = Path(os.environ["WNN_OUTPUT_DIR"])
    limit = 28_000_000_000
    receipt = {
        "scope": "supervisor plus worker and all descendants; summed RSS, shared pages counted conservatively",
        "limit_bytes": limit,
        "sample_interval_seconds": 1,
        "peak_rss_bytes": 0,
        "status": "running",
    }
    path = output / "worker_resource_receipt.json"
    process = psutil.Process()
    child = subprocess.Popen(
        [sys.executable, str(Path(__file__).with_name("worker.py"))], start_new_session=True
    )

    def interrupted(signum: int, _frame: object) -> None:
        receipt["status"] = "supervisor_interrupted"
        receipt["signal"] = signum
        stop_group(child)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while child.poll() is None:
            observed = aggregate_rss(process)
            receipt["peak_rss_bytes"] = max(receipt["peak_rss_bytes"], observed)
            if observed > limit:
                receipt["status"] = "aggregate_rss_limit_exceeded"
                path.write_text(json.dumps(receipt, indent=2) + "\n")
                stop_group(child)
                break
            path.write_text(json.dumps(receipt, indent=2) + "\n")
            time.sleep(1)
    finally:
        if child.poll() is None:
            stop_group(child)
        receipt["child_exit_code"] = child.returncode
        if receipt["status"] == "running":
            receipt["status"] = "complete"
        path.write_text(json.dumps(receipt, indent=2) + "\n")
    return child.returncode if receipt["status"] == "complete" else 137


if __name__ == "__main__":
    raise SystemExit(main())
