from __future__ import annotations

import os
import time
from pathlib import Path

from capacity_fit_worker import run

if __name__ == '__main__':
    os.environ.setdefault('WNN_KERNEL_DEADLINE_MONOTONIC', repr(time.monotonic() + 26000))
    run(Path(__file__).resolve().parent)
