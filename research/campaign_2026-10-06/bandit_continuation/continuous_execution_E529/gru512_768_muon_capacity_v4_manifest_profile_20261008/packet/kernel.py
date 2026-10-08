"""Kaggle entrypoint for the reviewed GRU capacity worker."""

from __future__ import annotations

import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WNN_THREADS"] = "4"
os.environ["WNN_LOADER_WORKERS"] = "0"
os.environ["WNN_LOADER_RAM_GB"] = "0.30"
os.environ["WNN_GPU_FRACTION"] = "0.75"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["PYTHONUNBUFFERED"] = "1"

from capacity_worker import run  # noqa: E402


def main() -> None:
    """Start the one bounded profile/conditional-fit cohort."""
    receipt = run(Path(__file__).resolve().parent)
    print(
        "GRU_CAPACITY_TERMINAL "
        f"status={receipt['status']} fit_started={receipt['fit_started']} "
        f"worker_seconds={receipt['worker_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
