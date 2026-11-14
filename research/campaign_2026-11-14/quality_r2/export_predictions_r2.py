"""Run the pinned prediction exporter with explicit IEEE FP32 recurrence kernels."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch

SOURCE = Path("/home/vlad/development/wunder/research_tmp/q14_strict_source")


def main() -> None:
    """Forward prediction arguments without changing the frozen exporter or grader."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.set_float32_matmul_precision("highest")
    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "scripts")]
    script = SOURCE / "scripts/predict_run.py"
    sys.argv = [str(script), *sys.argv[1:]]
    print("PREDICT_STRICT_FP32 IEEE recurrence; autocast off", flush=True)
    with torch.autocast(device_type="cuda", enabled=False):
        runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
