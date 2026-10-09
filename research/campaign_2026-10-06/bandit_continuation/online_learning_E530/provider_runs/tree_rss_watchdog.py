#!/usr/bin/env python3
"""Run one external CLI with a wall-time and summed process-tree RSS ceiling."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

PROVIDER_RUNS = Path(__file__).resolve().parent


def _group_rss_kib(group_id: int) -> int:
    total = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
            if int(fields[2]) != group_id:
                continue
            status = (entry / "status").read_text(encoding="ascii")
        except (IndexError, OSError, ValueError):
            continue
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                total += int(line.split()[1])
                break
    return total


def _stop_group(group_id: int) -> None:
    try:
        os.killpg(group_id, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(2)
    with suppress(ProcessLookupError):
        os.killpg(group_id, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--max-rss-kib", type=int, default=1_048_576)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    receipt_path = args.receipt.resolve()
    working_directory = args.cwd.resolve()
    if not command or not working_directory.is_dir():
        parser.error("a command and existing isolated working directory are required")
    if args.timeout_seconds <= 0 or args.max_rss_kib <= 0:
        parser.error("resource ceilings must be positive")
    if not receipt_path.is_relative_to(PROVIDER_RUNS):
        parser.error("receipt must be written inside this E530 provider_runs directory")
    if not working_directory.is_relative_to(PROVIDER_RUNS):
        parser.error("CLI working directory must stay inside this E530 provider_runs directory")
    try:
        os.sched_setaffinity(0, {0})
    except (AttributeError, OSError):
        parser.error("could not pin this process tree to one CPU")

    started = datetime.now(UTC).isoformat(timespec="seconds")
    process = subprocess.Popen(command, cwd=working_directory, start_new_session=True)
    deadline = time.monotonic() + args.timeout_seconds
    peak = 0
    outcome = "completed"
    while True:
        return_code = process.poll()
        rss = _group_rss_kib(process.pid)
        peak = max(peak, rss)
        if return_code is not None and rss == 0:
            break
        if rss > args.max_rss_kib:
            outcome = "rss_limit_exceeded"
            _stop_group(process.pid)
            break
        if time.monotonic() >= deadline:
            outcome = "wall_time_limit_exceeded"
            _stop_group(process.pid)
            break
        time.sleep(0.25)
    return_code = process.wait()
    receipt = {
        "schema_version": 1,
        "started_at_utc": started,
        "finished_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": command,
        "cwd": str(working_directory),
        "cpu_affinity": [0],
        "wall_time_limit_seconds": args.timeout_seconds,
        "rss_sum_limit_kib": args.max_rss_kib,
        "peak_rss_sum_kib": peak,
        "outcome": outcome,
        "return_code": return_code,
        "transcript_saved": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"watchdog: outcome={outcome} return_code={return_code} peak_rss_sum_kib={peak}",
        file=sys.stderr,
    )
    return return_code if outcome == "completed" else 124


if __name__ == "__main__":
    raise SystemExit(main())
