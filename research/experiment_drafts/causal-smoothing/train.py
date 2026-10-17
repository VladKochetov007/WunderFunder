from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch

SOURCE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE_ROOT / "src"))

from training_report import SmoothingObserver  # noqa: E402

from wnn.training.environment import TrainingEnvironment  # noqa: E402
from wnn.training.runner import run_training  # noqa: E402


def main() -> None:
    """Run the canonical trainer with strict precision and experiment receipts."""
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a GPU; CPU execution is disabled")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if sys.argv[1:2] == ["--predict"]:
        del sys.argv[1]
        sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
        runpy.run_path(str(SOURCE_ROOT / "scripts/predict_run.py"), run_name="__main__")
        return
    run_training(environment=TrainingEnvironment(observer=SmoothingObserver()))


if __name__ == "__main__":
    main()
