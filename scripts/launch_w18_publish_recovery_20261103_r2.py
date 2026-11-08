"""Own the CPU-only publication repair and its temporary credential cleanup."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import signal
import stat
import subprocess
import time
from pathlib import Path
from types import FrameType

R = Path("/data/epanov_s/tabdata/runtime/w18-submit-recovery-20261103-r1")
F = Path("/data/epanov_s/tabdata/runtime/final-submit-all-labeled-20261031-r1")
DEADLINE = 1793673140.7359152
PUBLISH_SHA = "d6a0a5f2921d43ee780b597c89edaf50539c1a5c6803de637d8ae0d280ee9887"


def main() -> None:
    """Run no fit/profile command; the outer GNU timeout owns this process group."""
    token = R / "private/publish-r2.token"
    info = token.lstat()
    identity = (info.st_dev, info.st_ino)
    assert stat.S_ISREG(info.st_mode) and info.st_mode & 0o077 == 0

    def cleanup() -> None:
        try:
            current = token.lstat()
            if (current.st_dev, current.st_ino) == identity:
                token.unlink()
        except FileNotFoundError:
            pass

    def stop(signum: int, frame: FrameType | None) -> None:
        raise SystemExit(128 + signum)

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, stop)
    publisher = R / "control/publish_w18_r2.py"
    assert hashlib.sha256(publisher.read_bytes()).hexdigest() == PUBLISH_SHA
    assert time.time() < DEADLINE
    env = dict(
        os.environ,
        PYTHONPATH=str(R / "source") + ":" + str(R / "source/src"),
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    code = 1
    try:
        code = subprocess.call(
            [
                "/data/epanov_s/tabdata/.venv/bin/python",
                "-u",
                str(publisher),
                "--runtime-root",
                str(F),
                "--recovery-root",
                str(R),
                "--deadline-epoch",
                str(DEADLINE),
                "--state-file",
                str(R / "reports-publish-r2/w18-publication-state-20261103-r2.json"),
            ],
            env=env,
            cwd=R,
        )
    finally:
        cleanup()
        receipt = {
            "exit_code": code,
            "finished_epoch": time.time(),
            "owned_token_removed": not token.exists(),
            "scope": "CPU-only publication repair; zero fits or profiles",
        }
        with (R / "receipts/LAUNCHER_EXIT_PUBLISH_R2.json").open("x") as stream:
            json.dump(receipt, stream, indent=2)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
