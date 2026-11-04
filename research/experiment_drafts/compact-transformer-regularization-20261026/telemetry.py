"""Record shared-node resources separately from this campaign's actual progress."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def main() -> None:
    """Sample host/GPU state without attributing foreign GPU utilization to our fits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()
    deadline = time.monotonic() + args.hours * 3600
    path = args.root / "HARDWARE.jsonl"
    while time.monotonic() < deadline:
        value = {
            "unix": time.time(),
            "loadavg": os.getloadavg(),
            "disk_free_bytes": shutil.disk_usage(args.root).free,
        }
        memory = Path("/proc/meminfo").read_text().splitlines()
        value["host_memory_kib"] = {
            line.split(":")[0]: int(line.split()[1])
            for line in memory
            if line.split(":")[0] in {"MemAvailable", "MemTotal", "SwapFree"}
        }
        for key, query in [
            (
                "shared_gpu",
                "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free,power.draw",
            ),
            ("cuda_processes", "--query-compute-apps=gpu_uuid,pid,used_memory"),
        ]:
            try:
                result = subprocess.run(
                    ["nvidia-smi", query, "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=True,
                )
                value[key] = result.stdout.strip().splitlines()
            except (subprocess.SubprocessError, OSError) as error:
                value[key] = {"error": str(error)}
        pointer = args.root / "SELECTED_PLAN.json"
        if pointer.exists():
            selected = json.loads(pointer.read_text())
            plan = json.loads(Path(selected["path"]).read_text())
            state = Path(plan["run_root"]) / "STATE.json"
            value["cohort"] = (
                json.loads(state.read_text()) if state.exists() else {"status": "not_started"}
            )
        with path.open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True) + "\n")
            stream.flush()
        if (args.root / "DELIVERY_COMPLETE.json").exists():
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
