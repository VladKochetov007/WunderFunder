"""Append actual process/CUDA resources after an unchanged legacy profile command."""

from __future__ import annotations

import json
import os
import resource
import runpy
import subprocess
import sys
from pathlib import Path

import torch


def main() -> None:
    """Profile the public trainer, preserving its arguments, RNG setup and output."""
    log_path = Path(sys.argv[1])
    arguments = sys.argv[2:]
    if "--profile-steps" not in arguments:
        raise ValueError("this wrapper is for bounded engineering profiles only")
    sys.argv = ["scripts/train_rnn.py", *arguments]
    try:
        runpy.run_module("scripts.train_rnn", run_name="__main__", alter_sys=True)
    finally:
        sys.stdout.flush()
        if torch.cuda.is_initialized():
            evidence = [
                json.loads(line.partition(" ")[2])
                for line in log_path.read_text().splitlines()
                if line.startswith("PROFILE_UPDATE_EVIDENCE ")
            ]
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            visible = os.environ["CUDA_VISIBLE_DEVICES"]
            gpu_uuid = subprocess.check_output(
                ["nvidia-smi", "-i", visible, "--query-gpu=uuid", "--format=csv,noheader"],
                text=True,
                timeout=15,
            ).strip()
            payload = {
                "gpu_uuid": gpu_uuid,
                "gpu_name": properties.name,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "device_total_memory_bytes": properties.total_memory,
                "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "oom_count": torch.cuda.memory_stats(device).get("num_ooms", 0),
                "autocast": torch.is_autocast_enabled("cuda"),
                "amp_scaler": evidence[-1]["grad_scaler_enabled"] if evidence else None,
                "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                "matmul_precision": torch.get_float32_matmul_precision(),
                "scope": "actual same-process CUDA peaks and main-process RSS; no quality score",
            }
            print("PROFILE_RESOURCES", json.dumps(payload, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
