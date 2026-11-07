from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Run one explicitly selected CPU/GPU canonical pytest probe in an isolated environment."""
    environment = dict(os.environ)
    raw = Path(sys.argv[1]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sys.argv[2]:
        raise ValueError("probe input changed before isolated launch")
    if json.loads(raw)["device"] == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["WNN_GRU_RECOVERY_PROBE"] = json.dumps({"path": sys.argv[1], "sha256": sys.argv[2]})
    environment["PYTEST_ADDOPTS"] = ""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", sys.argv[3]], env=environment
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
