"""Apply the controller's L4 process envelope before importing a fit/export actor."""

from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path


def main() -> None:
    """Execute one actor in-process so the CUDA allocator limit remains effective."""
    script = Path(sys.argv[1]).resolve(strict=True)
    if script.parent != Path(__file__).resolve().parent or script.name not in {
        "train_cell.py",
        "export_cell.py",
    }:
        raise ValueError("only the bundled fit and export entrypoints may run")
    run_root = Path(os.environ["WNN_RUN_DIR"]).absolute()
    run_root.parent.mkdir(parents=True, exist_ok=True)
    run_root = run_root.parent.resolve(strict=True) / run_root.name
    if run_root.is_symlink():
        raise ValueError("process output root must not be a symlink")
    if run_root.exists():
        if not run_root.is_dir() or any(run_root.iterdir()):
            raise FileExistsError(
                f"process output root must be fresh and empty: {run_root}"
            )
    else:
        run_root.mkdir(parents=True)
    run_dir = run_root.resolve(strict=True)
    for name in (
        "WNN_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        os.environ[name] = "2"
    os.environ["NUMEXPR_NUM_THREADS"] = "2"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True

    import torch

    if torch.cuda.device_count() != 1:
        raise ValueError("each cell must see exactly one assigned CUDA device")
    device = torch.cuda.get_device_properties(0)
    if "B300" not in device.name:
        raise ValueError("this campaign entrypoint is bound to NVIDIA B300")
    budget = 6 * 1024**3
    os.environ["WNN_GPU_MEMORY_BYTES"] = str(budget)
    torch.cuda.set_per_process_memory_fraction(budget / device.total_memory, 0)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    receipt = {
        "schema": "tabdata-b300-process-envelope.v1",
        "pid": os.getpid(),
        "script": script.name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": device.name,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(),
        "allocator_budget_bytes": budget,
        "allocator_limit_is_not_a_device_memory_reservation": True,
    }
    print("PROCESS_ENVELOPE " + json.dumps(receipt), flush=True)
    started = time.monotonic()
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *sys.argv[2:]]
    try:
        runpy.run_path(str(script), run_name="__main__")
        receipt["status"] = "actor_returned"
    except BaseException as error:
        receipt.update(status="actor_failed", error_type=type(error).__name__)
        raise
    finally:
        receipt.update(
            elapsed_seconds=time.monotonic() - started,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
        )
        path = run_dir / f"{script.stem}_resource_receipt.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
