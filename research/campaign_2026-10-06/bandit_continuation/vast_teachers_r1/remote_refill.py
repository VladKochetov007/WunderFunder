"""Host-side refill daemon: keep every GPU slot busy from a directory of queued job files.

Each /root/run/refill/pending/*.json holds {"name", "steps": [{label, argv, timeout}]}, run in name order.  A job starts
when fewer than MAX_JOBS trainers hold a GPU (MAX_JOBS comes from /root/run/refill/MAX_JOBS, default 6, re-read every
pass), on the GPU that currently holds the fewest compute processes (read from nvidia-smi, so jobs started by an earlier
daemon count).  A restarted daemon adopts jobs left in running/: it follows them by their log path and files them under
done/ as rc0 when their validation prediction exists, otherwise rc1.  Status goes to /root/run/refill/status.json.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
from pathlib import Path

RUN, WNN = Path("/root/run"), Path("/root/wnn")
BASE = RUN / "refill"
PENDING, RUNNING, DONE = BASE / "pending", BASE / "running", BASE / "done"
DEFAULT_MAX_JOBS, THREADS = 6, 6
for directory in (PENDING, RUNNING, DONE, RUN / "logs"):
    directory.mkdir(parents=True, exist_ok=True)
active: dict[str, dict] = {}


def max_jobs() -> int:
    try:
        return int((BASE / "MAX_JOBS").read_text().strip())
    except (OSError, ValueError):
        return DEFAULT_MAX_JOBS


def gpu_loads() -> dict[int, int]:
    """Compute processes per GPU index, from nvidia-smi."""
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=False).stdout.splitlines()
    index_of = {line.split(",")[1].strip(): int(line.split(",")[0]) for line in gpus if "," in line}
    loads = dict.fromkeys(index_of.values(), 0)
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=False).stdout.splitlines()
    for uuid in (line.strip() for line in apps if line.strip()):
        if uuid in index_of:
            loads[index_of[uuid]] += 1
    return loads


def launch(path: Path, env_base: dict) -> None:
    job = json.loads(path.read_text())
    loads = gpu_loads()
    gpu = min(loads, key=lambda g: loads[g]) if loads else 0
    moved = RUNNING / path.name
    path.rename(moved)
    script = " && ".join(
        f"( {' '.join(map(repr, s['argv']))} ) >> /root/run/logs/{job['name']}.{s['label']}.log 2>&1" for s in job["steps"])
    env = env_base | {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": str(THREADS), "MKL_NUM_THREADS": str(THREADS)}
    proc = subprocess.Popen(["bash", "-c", script], env=env, cwd=WNN, stdin=subprocess.DEVNULL)
    active[job["name"]] = {"name": job["name"], "proc": proc, "gpu": gpu, "file": moved, "started": time.time(), "adopted": False}


def adopt() -> None:
    """Follow jobs an earlier daemon left in running/ (found through their log path in the shell command line)."""
    for path in sorted(RUNNING.glob("*.json")):
        job = json.loads(path.read_text())
        first = job["steps"][0]["label"]
        needle = f"/root/run/logs/{job['name']}.{first}.log"
        if job["name"] in active:
            continue
        alive = subprocess.run(["pgrep", "-f", needle], capture_output=True, text=True, check=False).stdout.split()
        active[job["name"]] = {"name": job["name"], "proc": None, "gpu": -1, "file": path, "started": time.time(), "adopted": True,
                               "needle": needle, "alive_at_adopt": bool(alive)}


def finished(job: dict) -> tuple[bool, int | None]:
    if job["proc"] is not None:
        rc = job["proc"].poll()
        return rc is not None, rc
    alive = subprocess.run(["pgrep", "-f", job["needle"]], capture_output=True, text=True, check=False).stdout.split()
    if alive:
        return False, None
    ok = (WNN / f"cache/pred_{job['name']}_valid.json").exists()
    return True, 0 if ok else 1


def running_count() -> int:
    return sum(1 for job in active.values() if not finished(job)[0])


def save() -> None:
    state = {}
    for name, job in active.items():
        done, rc = finished(job)
        state[name] = {"gpu": job["gpu"], "running": not done, "rc": rc, "adopted": job["adopted"],
                       "minutes": round((time.time() - job["started"]) / 60, 1)}
    state["_pending"] = sorted(p.name for p in PENDING.glob("*.json"))
    state["_max_jobs"] = max_jobs()
    (BASE / "status.json").write_text(json.dumps(state, indent=1))


def trainers() -> int:
    return sum(gpu_loads().values())


def main() -> None:
    plan_env = json.loads((WNN / "vast_jobs.json").read_text())
    env_base = os.environ | {"WNN_PROJECT_ROOT": str(WNN), "PYTHONPATH": str(WNN / "src"), "PYTHONUNBUFFERED": "1",
                             **plan_env.get("env", {})}
    adopt()
    while True:
        for job in list(active.values()):
            done, rc = finished(job)
            if done and job["file"].exists():
                job["file"].rename(DONE / f"{job['file'].stem}.rc{rc}.json")
        queue = sorted(PENDING.glob("*.json"))
        free = min(max_jobs() - running_count(), max_jobs() - trainers())
        if queue and free > 0:
            with contextlib.suppress(FileNotFoundError):  # a vanished file means another daemon took it
                launch(queue[0], env_base)
            time.sleep(30)  # let the new trainer register on the GPU before the next count
        save()
        if (BASE / "STOP").exists() and running_count() == 0 and not queue:
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
