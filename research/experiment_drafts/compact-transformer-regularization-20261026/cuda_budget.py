"""Run an unchanged canonical entrypoint with an explicit per-rank CUDA allocation cap."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    """Bound this worker's allocator without touching co-resident CUDA processes."""
    import torch

    entrypoint = Path(sys.argv.pop(1)).resolve(strict=True)
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    cap = int(float(os.environ["WNN_CUDA_CAP_GIB"]) * (1 << 30))
    if not 0 < cap < total:
        raise ValueError("CUDA allocation cap is outside physical memory")
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    sys.argv[0] = str(entrypoint)
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
