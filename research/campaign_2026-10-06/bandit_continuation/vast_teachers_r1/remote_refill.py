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


MIN_FREE_MIB, MAX_OOM_RETRIES = 7000, 3


def gpu_free_mib() -> dict[int, int]:
    rows = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=False).stdout.splitlines()
    return {int(r.split(",")[0]): int(r.split(",")[1]) - int(r.split(",")[2]) for r in rows if r.count(",") == 2}


def oom_failure(job: dict) -> bool:
    """True when a failed job's logs show a CUDA out-of-memory (or the async CUDA error it causes)."""
    text = ""
    for log in (RUN / "logs").glob(f"{job['name']}.*.log"):
        with contextlib.suppress(OSError):
            text += log.read_text(errors="ignore")[-6000:]
    return "OutOfMemoryError" in text or "CUDA error" in text


def launch(path: Path, env_base: dict) -> None:
    job = json.loads(path.read_text())
    free_mib = gpu_free_mib()
    candidates = {g: m for g, m in free_mib.items() if m >= MIN_FREE_MIB}
    if not candidates:
        raise FileNotFoundError("no GPU has enough free memory yet")  # swallowed by the caller; retry next cycle
    gpu = max(candidates, key=lambda g: candidates[g])
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
                stem = job["file"].stem
                tries = int(stem.rsplit(".try", 1)[1]) if ".try" in stem else 0
                if rc != 0 and tries < MAX_OOM_RETRIES and oom_failure({"name": job["name"]}):
                    base = stem.rsplit(".try", 1)[0]
                    job["file"].rename(PENDING / f"{base}.try{tries + 1}.json")  # OOM is not a scientific result: run it again
                    time.sleep(60)
                    continue
                job["file"].rename(DONE / f"{stem}.rc{rc}.json")
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
