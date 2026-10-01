"""Execute a frozen job lane with resource caps and append-only receipts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def append(record):
    with (HERE / "job-ledger.jsonl").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(record) + "\n")
        stream.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--lane", type=int, required=True)
    ap.add_argument("--lanes", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    assert args.gpu in range(8) and 0 <= args.lane < args.lanes <= 2
    assert 1 <= args.timeout <= 1800
    plan_path = args.plan.resolve()
    assert plan_path.is_relative_to(HERE)
    plan = json.loads(plan_path.read_text())
    plan_hash = sha(plan_path)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=str(ROOT / "src"),
               WNN_GPU_FRACTION=".34", WNN_THREADS="4", WNN_LOADER_WORKERS="2",
               WNN_LOADER_RAM_GB="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", NUMBA_NUM_THREADS="4", ARROW_NUM_THREADS="4")
    log_dir = HERE / "logs"
    log_dir.mkdir(exist_ok=True)
    for job in plan["jobs"][args.lane::args.lanes]:
        assert datetime.now(timezone.utc) < datetime.fromisoformat("2026-10-01T07:00:00+00:00")
        assert sha(plan_path) == plan_hash
        for name, expected in plan.get("code_sha256", {}).items():
            assert sha(plan_path.parent / name) == expected, name
        for stage, argv in [("train", job["train_argv"]), ("predict", job["predict_argv"])]:
            memory = subprocess.check_output([
                "nvidia-smi", f"--id={args.gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"
            ], text=True).strip()
            if int(memory) < 6200:
                append({"event": "resource_blocked", "utc": stamp(), "job": job["name"],
                        "stage": stage, "gpu": args.gpu, "free_MiB": int(memory)})
                raise RuntimeError("GPU headroom below predeclared start threshold; no job started")
            log = log_dir / f"{job['name']}.{stage}.log"
            assert not log.exists(), log
            unit = f"wunder-{job['name']}-{stage}"
            command = ["systemd-run", "--user", "--scope", "--quiet", "--unit", unit, "-p", "MemoryMax=20G",
                       "-p", "MemorySwapMax=0", "-p", "CPUQuota=600%", "--", "nice", "-n", "10",
                       "/usr/bin/time", "-v", *argv]
            begin = time.monotonic()
            with log.open("x") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                append({"event": "started", "utc": stamp(), "job": job["name"], "stage": stage,
                        "gpu": args.gpu, "pid": process.pid, "plan_sha256": plan_hash,
                        "argv": command, "free_MiB": int(memory)})
                timed_out = False
                try:
                    code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    subprocess.run(["systemctl", "--user", "stop", unit + ".scope"],
                                   stdout=stream, stderr=subprocess.STDOUT, timeout=15, check=False)
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        code = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        code = process.wait(timeout=10)
            append({"event": "completed" if code == 0 else "failed", "utc": stamp(),
                    "job": job["name"], "stage": stage, "gpu": args.gpu, "exit_code": code,
                    "timed_out": timed_out, "seconds": time.monotonic() - begin,
                    "log_sha256": sha(log)})
            if code != 0:
                raise RuntimeError(f"{job['name']} {stage} failed, retained {log}")


if __name__ == "__main__":
    main()
