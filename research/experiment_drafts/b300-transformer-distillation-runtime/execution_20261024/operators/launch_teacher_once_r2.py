"""Start the reviewed finite teacher cohort once from an independent cron session."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata")
PLAN = ROOT / "runs/pr32-production-admission-20261024-r2/plan.json"
SUPERVISOR = ROOT / "runtime/pr32-operator-20261024/run_teacher_cohort.py"
PLAN_SHA = "9671e766f0217c452ba9c9c002230c784e49456a7f5edaca7759c2ae5ac384fa"
SUPERVISOR_SHA = "83b20eee85ccaf00664fd612a92985589a43c5ca4abb00057d033f1644aecfcb"


def main() -> None:
    """Claim a unique launch and replace this cron child with the verified controller."""
    for path, expected in ((PLAN, PLAN_SHA), (SUPERVISOR, SUPERVISOR_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"launch input changed after review: {path.name}")
    claim = PLAN.parent / "LAUNCHED_ONCE.json"
    try:
        descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return
    with os.fdopen(descriptor, "w") as stream:
        json.dump(
            {
                "created_unix": time.time(),
                "pid": os.getpid(),
                "plan_sha256": PLAN_SHA,
                "supervisor_sha256": SUPERVISOR_SHA,
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    python = str(ROOT / ".venv/bin/python")
    os.execv(python, [python, str(SUPERVISOR), "--plan", str(PLAN)])


if __name__ == "__main__":
    main()
