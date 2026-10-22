"""Observe allocator peaks before canonical profile context teardown."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

import torch


def main() -> None:
    """Run the unmodified CLI, then report live strict process resource counters."""
    entry = Path(sys.argv[1]).resolve()
    sys.argv = [str(entry), *sys.argv[2:]]
    try:
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        if torch.cuda.is_initialized():
            print(
                "RESOURCE_PROFILE",
                json.dumps(
                    {
                        "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                        "total_bytes": torch.cuda.get_device_properties(0).total_memory,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
