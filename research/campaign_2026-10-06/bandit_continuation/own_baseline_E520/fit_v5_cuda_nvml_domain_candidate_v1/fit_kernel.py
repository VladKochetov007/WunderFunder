"""Kaggle entry point for the externally reviewed E520 scratch-pair stage."""

from __future__ import annotations

import os
from pathlib import Path

# Keep the worker single-GPU and single-threaded before any numeric imports.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WNN_LOADER_WORKERS"] = "0"
os.environ["WNN_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fit_worker import run_fit  # noqa: E402


def main() -> None:
    """Run the exact admitted seed pair and print only its receipt status."""
    result = run_fit(Path(__file__).resolve().parent)
    print(
        f"E520 fit status={result['status']} elapsed_seconds={result['elapsed_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
