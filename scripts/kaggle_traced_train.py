"""Run scripts/train_rnn.py unchanged with periodic stack dumps (Kaggle hang diagnosis).

faulthandler writes every thread's stack to stderr every 240 s, so a silent stall inside data
setup becomes visible in the worker log. Arguments are passed through to train_rnn.
"""

from __future__ import annotations

import faulthandler
import runpy
import sys
import time
from pathlib import Path

faulthandler.enable()
faulthandler.dump_traceback_later(240, repeat=True)
print(f"TRACED_WORKER_START {time.time():.0f}", flush=True)
script = Path(__file__).resolve().parent / "train_rnn.py"
sys.argv = [str(script), *sys.argv[1:]]
runpy.run_path(str(script), run_name="__main__")
