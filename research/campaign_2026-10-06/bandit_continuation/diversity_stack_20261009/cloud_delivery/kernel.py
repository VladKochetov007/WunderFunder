from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

WORK = Path(os.environ.get("KAGGLE_WORKING", "/kaggle/working"))


def _boot_marker() -> None:
    marker = {
        "event": "kernel_boot",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "input_root": os.environ.get("KAGGLE_INPUT", "/kaggle/input"),
    }
    try:
        WORK.mkdir(parents=True, exist_ok=True)
        path = WORK / "boot_marker.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump(marker, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        print(json.dumps(marker, sort_keys=True), file=sys.stderr, flush=True)
        raise
    print(json.dumps(marker, sort_keys=True), flush=True)


_boot_marker()

# CPU_STAGE_RUNTIME_INSERTION_POINT
