"""Resource budget for research jobs on a shared laptop: 80% CPU, 80% GPU.

The hard limits (CPU quota, memory cap, no swap) come from ``scripts/limited``;
this module keeps thread pools and GPU usage inside the same budget.
"""

from __future__ import annotations

import os
import time

CPU_FRACTION = 0.8
# GPU memory cap per process. Upper bound agreed with the user (2026-09-25: 90%); WNN_GPU_FRACTION may only lower it.
GPU_FRACTION = min(0.9, float(os.environ.get("WNN_GPU_FRACTION", "0.9")))
# GPU busy-time rate limit per job (DutyCycle). 1.0 = no throttling (the single-job fast path);
# set WNN_GPU_DUTY below 1 only to deliberately share the GPU with other work.
GPU_DUTY = min(1.0, float(os.environ.get("WNN_GPU_DUTY", "1.0")))


def cpu_threads() -> int:
    env = os.environ.get("WNN_THREADS")
    if env:
        return max(1, int(env))
    return max(1, int((os.cpu_count() or 1) * CPU_FRACTION))


class DutyCycle:
    """Sleep after each unit of work so that busy time stays at ``fraction`` of wall time."""

    def __init__(self, fraction: float, clock=time.perf_counter, sleep=time.sleep):
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction
        self._clock = clock
        self._sleep = sleep
        self._mark = clock()

    def tick(self) -> None:
        busy = self._clock() - self._mark
        pause = busy * (1.0 - self.fraction) / self.fraction
        if pause > 0:
            self._sleep(pause)
        self._mark = self._clock()


def configure_torch(gpu_fraction: float | None = None, duty: float | None = None):
    """Cap torch CPU threads and GPU memory; return a GPU DutyCycle, or None when not rate-limited.

    The memory cap comes from WNN_GPU_FRACTION, the rate limit from WNN_GPU_DUTY (default 1.0 = none).
    With a DutyCycle, call ``torch.cuda.synchronize()`` then ``duty.tick()`` at step ends.
    """
    import torch

    gpu_fraction = GPU_FRACTION if gpu_fraction is None else min(gpu_fraction, GPU_FRACTION)
    torch.set_num_threads(cpu_threads())
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_fraction)
        torch.backends.cudnn.benchmark = True
    duty = GPU_DUTY if duty is None else min(duty, GPU_DUTY)
    return DutyCycle(duty) if duty < 1.0 else None


def dataloader_budget(batch_bytes: int, ram_budget_bytes: int | None = None, max_workers: int = 4):
    """Workers and prefetch so queued batches (workers * prefetch * batch) fit the RAM budget."""
    if ram_budget_bytes is None:
        ram_budget_bytes = int(float(os.environ.get("WNN_LOADER_RAM_GB", "6")) * (1 << 30))
    max_workers = min(max_workers, int(os.environ.get("WNN_LOADER_WORKERS", max_workers)))
    workers = max(1, min(max_workers, cpu_threads() // 2))
    prefetch = max(1, min(2, ram_budget_bytes // max(1, workers * batch_bytes)))
    return workers, int(prefetch)
