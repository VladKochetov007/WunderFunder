"""Thread and allocator limits for the resource-constrained research campaign.

The hard limits (CPU quota, memory cap, no swap) come from ``scripts/limited``;
this module keeps thread pools and GPU usage inside the same budget.
"""

from __future__ import annotations

import os
import time

CPU_FRACTION = 0.75
# Leave measured free memory for CUDA context, kernels and non-PyTorch allocations.
GPU_MEMORY_SAFETY_RESERVE_BYTES = 1_000_000_000
GPU_FRACTION = min(1.0, float(os.environ.get("WNN_GPU_FRACTION", "1.0")))
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
    """Set CPU threads and a live-memory-aware CUDA allocator limit.

    WNN_GPU_MEMORY_BYTES preserves an explicit byte cap for older frozen attempts.
    Otherwise the allocator limit follows currently free device memory and leaves
    WNN_GPU_MEMORY_RESERVE_BYTES free, defaulting to 1 GB.
    The rate limit comes from WNN_GPU_DUTY (default 1.0 = none).
    With a DutyCycle, call ``torch.cuda.synchronize()`` then ``duty.tick()`` at step ends.

    Returns a GPU DutyCycle, or None when execution is not rate-limited.
    """
    import torch

    gpu_fraction = GPU_FRACTION if gpu_fraction is None else min(gpu_fraction, GPU_FRACTION)
    torch.set_num_threads(cpu_threads())
    if torch.cuda.is_available():
        memory_bytes = os.environ.get("WNN_GPU_MEMORY_BYTES")
        if memory_bytes is not None:
            total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            gpu_fraction = allocator_fraction(memory_bytes, total)
        else:
            free_bytes, total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
            available_fraction = available_allocator_fraction(
                free_bytes, total_bytes, gpu_memory_safety_reserve_bytes()
            )
            gpu_fraction = min(gpu_fraction, available_fraction)
        torch.cuda.set_per_process_memory_fraction(gpu_fraction)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    duty = GPU_DUTY if duty is None else min(duty, GPU_DUTY)
    return DutyCycle(duty) if duty < 1.0 else None


def configure_strict_fp32() -> None:
    """Disable CUDA TF32 paths for a declared full-float32 execution path."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def allocator_fraction(memory_bytes: str, total_bytes: int) -> float:
    """Validate a byte-based allocation ceiling without confusing GB and GiB.

    This limits the PyTorch allocator only; the campaign monitor separately observes
    total device usage, including the CUDA context and other processes.
    """
    if not memory_bytes.isascii() or not memory_bytes.isdigit():
        raise ValueError("WNN_GPU_MEMORY_BYTES must be an integer byte count")
    limit = int(memory_bytes)
    if not 0 < limit <= total_bytes:
        raise ValueError("WNN_GPU_MEMORY_BYTES must fit the selected device")
    return limit / total_bytes


def available_allocator_fraction(free_bytes: int, total_bytes: int, reserve_bytes: int) -> float:
    """Return an allocator fraction from current free bytes and a safety reserve."""
    if total_bytes <= 0 or free_bytes < 0 or free_bytes > total_bytes or reserve_bytes < 0:
        raise ValueError("GPU memory observations and reserve must be nonnegative and consistent")
    usable_bytes = free_bytes - reserve_bytes
    if usable_bytes <= 0:
        raise RuntimeError("free GPU memory does not exceed the configured safety reserve")
    return min(1.0, usable_bytes / total_bytes)


def gpu_memory_safety_reserve_bytes() -> int:
    """Return a validated configurable reserve for CUDA and non-PyTorch allocations."""
    value = os.environ.get("WNN_GPU_MEMORY_RESERVE_BYTES")
    if value is None:
        return GPU_MEMORY_SAFETY_RESERVE_BYTES
    if not value.isascii() or not value.isdigit():
        raise ValueError("WNN_GPU_MEMORY_RESERVE_BYTES must be a nonnegative integer")
    return int(value)


def dataloader_budget(
    batch_bytes: int, ram_budget_bytes: int | None = None, max_workers: int = 4
) -> tuple[int, int]:
    """Budget queued batches plus one consumed and one pinned/in-flight copy.

    ``batch_bytes`` must include all tensors (features, targets and teachers).
    Resident datasets, model state and worker overhead need a separate reserve.
    Zero workers uses synchronous loading; callers must omit prefetch_factor.
    """
    if ram_budget_bytes is None:
        ram_budget_bytes = int(float(os.environ.get("WNN_LOADER_RAM_GB", "6")) * (1 << 30))
    max_workers = min(max_workers, int(os.environ.get("WNN_LOADER_WORKERS", max_workers)))
    if batch_bytes <= 0 or ram_budget_bytes <= 0 or max_workers < 0:
        raise ValueError("batch and RAM budgets must be positive; workers nonnegative")
    capacity = ram_budget_bytes // batch_bytes
    if capacity < 2:
        raise ValueError("loader RAM budget cannot hold a batch and its in-flight copy")
    workers = min(max_workers, max(1, cpu_threads() // 2), capacity - 2)
    if workers == 0:
        return 0, 0
    prefetch = min(2, (capacity - 2) // workers)
    return workers, prefetch
