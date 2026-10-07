"""Kaggle script entry point for the E520 profile-only worker."""

from __future__ import annotations

import os
from pathlib import Path

# Set process-wide accelerator/thread visibility before importing Torch.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WNN_LOADER_WORKERS"] = "0"
os.environ["WNN_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

from profile_worker import run_profile  # noqa: E402


def main() -> None:
    """Run only the reviewed E520 profile packet and print a compact receipt pointer."""
    result = run_profile(Path(__file__).resolve().parent)
    print(
        f"E520 profile status={result['status']} wall_seconds={result['wall_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
