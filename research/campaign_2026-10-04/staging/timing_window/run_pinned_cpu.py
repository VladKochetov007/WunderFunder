"""Set one-CPU affinity before entering an unchanged timing entry point."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[1:3] != ["--cpu", "15"]:
        raise ValueError("This fixed timing window requires CPU15 and a script")
    target = Path(sys.argv[3]).resolve()
    root = Path(__file__).resolve().parents[4]
    allowed = {
        root / "research/campaign_2026-10-04/staging/e462/time_public_q8.py",
        Path(__file__).resolve().with_name("time_seed4_i16.py"),
    }
    if target not in allowed or 15 not in os.sched_getaffinity(0):
        raise ValueError("Unrecognized timing entry point or unavailable CPU15")
    os.sched_setaffinity(0, {15})
    if os.sched_getaffinity(0) != {15}:
        raise RuntimeError("Single-CPU timing affinity was not enforced")
    sys.argv = [str(target), *sys.argv[4:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
