"""Kaggle entry point for the conditional H232 profile-to-fit worker."""

from __future__ import annotations

import os
from pathlib import Path

# Set process-wide accelerator/thread visibility before importing Torch.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WNN_LOADER_WORKERS"] = "2"
os.environ["WNN_LOADER_RAM_GB"] = "0.30"
os.environ["WNN_GPU_FRACTION"] = "0.75"
os.environ["WNN_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

from profile_worker import run_profile  # noqa: E402


def main() -> None:
    """Run the reviewed profile and its strict same-worker conditional fit gate."""
    result = run_profile(Path(__file__).resolve().parent)
    print(
        f"H232 conditional status={result['status']} fit_started={result.get('fit_started')} "
        f"wall_seconds={result['wall_seconds']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
