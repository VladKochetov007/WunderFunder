from __future__ import annotations

import argparse
import logging
import os
import subprocess
import time
from pathlib import Path

from wnn.training.sequence_observer import SequenceObserver

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Attach read-only observation, or wrap one caller-authorized worker command.

    This helper never sends signals. Its optional wall limit stops observation
    only; the worker's existing durable controller owns execution deadlines.
    """
    parser = argparse.ArgumentParser(description="Independent sequence-worker/GPU observer")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--max-seconds", type=float, default=86_400)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if bool(command) == (args.pid is not None):
        parser.error("supply exactly one of --pid or a worker command following --")
    if not 1 <= args.interval <= 60 or not 1 <= args.max_seconds <= 604_800:
        parser.error("interval must be 1..60 and max-seconds must be 1..604800")
    process: subprocess.Popen[bytes] | None = None
    if command:
        environment = dict(os.environ, WNN_MONITOR_HEARTBEAT_SECONDS=str(min(args.interval, 5)))
        process = subprocess.Popen(command, env=environment)
    pid = process.pid if process is not None else args.pid
    observer = SequenceObserver(args.output, pid, args.interval)
    deadline = time.monotonic() + args.max_seconds
    while True:
        returncode = process.poll() if process is not None else None
        event = observer.sample(returncode)
        if event["metrics"]["observer/process_alive_count"] == 0:
            return returncode if returncode is not None else 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            LOGGER.warning(
                "observer reached its wall limit; worker lifecycle remains with its owner"
            )
            return 124
        time.sleep(min(args.interval, remaining))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
