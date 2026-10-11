"""Profile-only stack heartbeat for long startup validation steps."""

from __future__ import annotations

import faulthandler
import json
import os
import sys

if os.environ.get("WNN_STARTUP_DIAGNOSTICS") == "1":
    interval = float(os.environ.get("WNN_STARTUP_STACK_DUMP_SECONDS", "60"))
    if interval <= 0:
        raise ValueError("startup stack interval must be positive")
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(interval, repeat=True, file=sys.stderr)
    print(
        json.dumps(
            {"event": "startup_stack_watchdog_armed", "interval_seconds": interval},
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
