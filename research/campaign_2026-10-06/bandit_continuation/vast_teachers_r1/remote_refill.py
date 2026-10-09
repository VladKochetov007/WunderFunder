"""Host-side refill daemon: keep every GPU slot busy from a directory of queued job files.

Each /root/run/refill/pending/*.json holds {"name", "steps": [{label, argv, timeout}]}, run in name order.
A job starts when fewer than MAX_JOBS train_rnn.py processes run, on the GPU with fewer running jobs.
Status goes to /root/run/refill/status.json.  Exits after STOP exists and nothing is running, or on
DRAIN when the pending directory is empty.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

RUN, WNN = Path("/root/run"), Path("/root/wnn")
BASE = RUN / "refill"
PENDING, RUNNING, DONE = BASE / "pending", BASE / "running", BASE / "done"
MAX_JOBS, GPUS, THREADS = 4, (0, 1), 6
for directory in (PENDING, RUNNING, DONE, RUN / "logs"):
    directory.mkdir(parents=True, exist_ok=True)
active: dict[str, dict] = {}


def trainers() -> int:
    """Main trainers only: DataLoader workers share the command line but never hold the GPU."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True, check=False)
    return len([line for line in out.stdout.splitlines() if line.strip()])


def launch(path: Path, env_base: dict) -> None:
    job = json.loads(path.read_text())
    load = {g: sum(1 for j in active.values() if j["gpu"] == g and j["proc"].poll() is None) for g in GPUS}
    gpu = min(GPUS, key=lambda g: load[g])
    moved = RUNNING / path.name
    path.rename(moved)
    script = " && ".join(
        f"( {' '.join(map(repr, s['argv']))} ) >> /root/run/logs/{job['name']}.{s['label']}.log 2>&1" for s in job["steps"])
    env = env_base | {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS)}
    proc = subprocess.Popen(["bash", "-c", script], env=env, cwd=WNN, stdin=subprocess.DEVNULL)
    active[job["name"]] = {"proc": proc, "gpu": gpu, "file": moved, "started": time.time()}


def save() -> None:
    state = {n: {"gpu": j["gpu"], "running": j["proc"].poll() is None, "rc": j["proc"].poll(),
                 "minutes": round((time.time() - j["started"]) / 60, 1)} for n, j in active.items()}
    state["_pending"] = sorted(p.name for p in PENDING.glob("*.json"))
    (BASE / "status.json").write_text(json.dumps(state, indent=1))


def main() -> None:
    plan_env = json.loads((WNN / "vast_jobs.json").read_text())
    env_base = os.environ | {"WNN_PROJECT_ROOT": str(WNN), "PYTHONPATH": str(WNN / "src"), "PYTHONUNBUFFERED": "1",
                             **plan_env.get("env", {})}
    while True:
        for job in list(active.values()):
            rc = job["proc"].poll()
            if rc is not None and job["file"].exists():
                job["file"].rename(DONE / f"{job['file'].stem}.rc{rc}.json")
        queue = sorted(PENDING.glob("*.json"))
        free = MAX_JOBS - sum(1 for j in active.values() if j["proc"].poll() is None)
        # worker.py jobs also count: wait until the primary worker leaves a slot
        free = min(free, MAX_JOBS - trainers())
        if queue and free > 0:
            launch(queue[0], env_base)
            time.sleep(30)  # let the new trainer register before the next count
        save()
        idle = all(j["proc"].poll() is not None for j in active.values())
        if (BASE / "STOP").exists() and idle and not queue:
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
