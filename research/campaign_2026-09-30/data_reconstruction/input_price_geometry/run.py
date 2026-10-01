"""Foreground bounded launcher; no systemd or durable resource claim."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    own = Path(__file__).resolve().parent
    root = own.parents[3]
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    command = [
        "env",
        "OPENBLAS_NUM_THREADS=1",
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "NUMBA_NUM_THREADS=1",
        "POLARS_MAX_THREADS=1",
        "ARROW_NUM_THREADS=1",
        "CUDA_VISIBLE_DEVICES=",
        "timeout",
        "60s",
        "prlimit",
        "--as=2147483648",
        "--cpu=60",
        str(root / ".venv/bin/python"),
        str(own / "probe.py"),
    ]
    started = time.monotonic()
    with (own / "run.log").open("x") as log:
        result = subprocess.run(
            command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    with (own / "exit.json").open("x") as stream:
        json.dump(
            {
                "command": command,
                "wall_seconds": time.monotonic() - started,
                "exit_code": result.returncode,
                "foreground": True,
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print("exit", result.returncode, "seconds", time.monotonic() - started)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
