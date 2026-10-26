"""Measure the four matched configurations concurrently on their actual assigned GPUs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    """Run one bounded engineering cohort, preserving every attempt and rank log."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument("--max-seconds", type=int, default=1200)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    workers = []
    started = time.monotonic()
    try:
        for job in plan["jobs"]:
            out = args.output / job["name"]
            out.mkdir()
            tmp = Path(plan["tmp_root"]) / args.output.name / job["name"]
            tmp.mkdir(parents=True)
            env = os.environ | {
                "CUDA_VISIBLE_DEVICES": ",".join(job["gpu_uuids"]),
                "PYTHONPATH": plan["source_root"] + "/src",
                "CUBLAS_EMULATE_SINGLE_PRECISION": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "TORCHINDUCTOR_COMPILE_THREADS": "1",
                "TORCHINDUCTOR_CACHE_DIR": str(tmp / "inductor"),
                "TMPDIR": str(tmp),
                "WNN_CUDA_CAP_GIB": str(job.get("cuda_cap_gib", 100)),
                "PYTHONUNBUFFERED": "1",
            }
            command = [
                "taskset",
                "-c",
                ",".join(map(str, job["cpus"])),
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc_per_node=" + str(len(job["gpu_uuids"])),
                plan["launcher"],
                plan["source_root"] + "/scripts/train_sequence.py",
                "--config",
                job["config"],
                "--data-root",
                plan["data_root"],
                "--split-manifest",
                job["split_manifest"],
                "--output",
                str(out / "profile"),
                "--profile-updates",
                str(args.updates),
            ]
            with (out / "console.log").open("wb") as log:
                proc = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
                )
            workers.append((job["name"], proc))
            (out / "LAUNCH.json").write_text(
                json.dumps({"pid": proc.pid, "command": command}) + "\n"
            )
        while any(p.poll() is None for _, p in workers):
            if time.monotonic() - started > args.max_seconds:
                raise TimeoutError("bounded profile cohort exceeded deadline")
            time.sleep(2)
        result = {
            "elapsed_seconds": time.monotonic() - started,
            "workers": {name: p.returncode for name, p in workers},
        }
        (args.output / "EXIT.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        for _, proc in workers:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=15)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=15)


if __name__ == "__main__":
    main()
