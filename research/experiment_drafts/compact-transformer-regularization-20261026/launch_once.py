"""Claim one authorized cron launch without automatic retries."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    """Replace this independent cron child with one explicitly specified program."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a Python program and its arguments are required")
    try:
        descriptor = os.open(args.claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    with os.fdopen(descriptor, "w") as stream:
        json.dump(
            {"pid": os.getpid(), "created_unix": time.time(), "command": args.command}, stream
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.execv(sys.executable, [sys.executable, *args.command])


if __name__ == "__main__":
    main()
