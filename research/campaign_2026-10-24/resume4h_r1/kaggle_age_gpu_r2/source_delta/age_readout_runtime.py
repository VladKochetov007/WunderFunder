"""Initialize deterministic PyTorch before any CUDA context can be created."""

from __future__ import annotations

import importlib
import os
from types import ModuleType

CUBLAS_WORKSPACE_CONFIG = ":4096:8"
CUDA_ALLOCATOR_LIMIT_BYTES = 4_000_000_000
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

torch: ModuleType = importlib.import_module("torch")
torch.use_deterministic_algorithms(True)


def configure_cuda_memory(device_name: str) -> None:
    """Cap the selected CUDA allocator at four GiB and reset its peak counters."""
    if not device_name.startswith("cuda:"):
        return
    device_index = int(device_name.partition(":")[2])
    total_bytes = int(torch.cuda.get_device_properties(device_index).total_memory)
    fraction = min(1.0, CUDA_ALLOCATOR_LIMIT_BYTES / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
    torch.cuda.reset_peak_memory_stats(device_index)
