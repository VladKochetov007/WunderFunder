"""Run one sealed child-source Python command with its own imports and data root."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Run one source-snapshot script with that snapshot's local imports."""
    if len(sys.argv) < 2:
        raise SystemExit("usage: cleanfold_exec.py <source-relative-script> [arguments...]")
    child = Path(sys.argv[1])
    if child.is_absolute() or ".." in child.parts:
        raise SystemExit("child script must be a safe path inside the frozen source snapshot")
    script = (SOURCE_ROOT / child).resolve(strict=True)
    if not script.is_relative_to(SOURCE_ROOT):
        raise SystemExit("child script escaped the frozen source snapshot")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SOURCE_ROOT / "src")
    env.setdefault("WNN_PROJECT_ROOT", "/root/wnn")
    return subprocess.run([sys.executable, str(script), *sys.argv[2:]], env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
