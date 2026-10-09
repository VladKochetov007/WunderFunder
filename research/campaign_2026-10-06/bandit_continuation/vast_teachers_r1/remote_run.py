"""Remote worker for one rented Vast machine: stage inputs, run a finite teacher job list, report status.

Runs on the rented host only.  Reads plan.json (signed URLs, hashes, sizes) and jobs.json.
Every phase is recorded in /root/run/status.json; a terminal DONE.json or FAILED.json is always
written.  Downloads use parallel HTTP range requests and are checked against SHA-256 before use.
No credentials are present on the host; the controller collects outputs and destroys the instance.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUN = Path("/root/run")
WNN = Path("/root/wnn")
STATUS = RUN / "status.json"
LOCK = threading.Lock()
STATE: dict = {"phase": "start", "started": time.time(), "jobs": {}, "downloads": {}}


def save() -> None:
    with LOCK:
        STATE["updated"] = time.time()
        tmp = STATUS.with_suffix(".tmp")
        tmp.write_text(json.dumps(STATE, indent=1, default=str))
        tmp.replace(STATUS)


def phase(name: str, **extra) -> None:
    STATE["phase"] = name
    STATE.setdefault("phases", []).append({"phase": name, "t": round(time.time() - STATE["started"], 1), **extra})
    save()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_range(url: str, path: Path, start: int, end: int, progress: list) -> None:
    for attempt in range(6):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=120) as response, path.open("r+b") as out:
                out.seek(start)
                position = start
                while True:
                    chunk = response.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
                    position += len(chunk)
                    with LOCK:
                        progress[0] += len(chunk)
            if position != end + 1:
                raise OSError(f"short range {start}-{end}: got to {position}")
            return
        except Exception:  # retry the whole range on any network error
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))


def download(item: dict, parts: int = 16) -> None:
    target = WNN / item["remote_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    size, url = int(item["size_bytes"]), item["url"]
    record = STATE["downloads"].setdefault(item["remote_path"], {"bytes": size})
    if target.exists() and target.stat().st_size == size and sha256(target) == item["sha256"]:
        record["state"] = "cached"
        save()
        return
    partial = target.with_name(target.name + ".partial")
    with partial.open("wb") as handle:
        handle.truncate(size)
    progress = [0]
    started = time.time()
    record["state"] = "downloading"
    step = -(-size // parts)
    ranges = [(i, min(i + step, size) - 1) for i in range(0, size, step)]

    def report() -> None:
        while record.get("state") == "downloading":
            record["done_bytes"] = progress[0]
            record["mb_per_s"] = round(progress[0] / max(time.time() - started, 1e-3) / 1e6, 1)
            save()
            time.sleep(10)

    reporter = threading.Thread(target=report, daemon=True)
    reporter.start()
    with ThreadPoolExecutor(max_workers=parts) as pool:
        list(pool.map(lambda r: fetch_range(url, partial, r[0], r[1], progress), ranges))
    record["state"] = "verifying"
    record["download_seconds"] = round(time.time() - started, 1)
    if sha256(partial) != item["sha256"]:
        record["state"] = "hash_mismatch"
        save()
        raise ValueError(f"SHA-256 mismatch for {item['remote_path']}")
    partial.replace(target)
    record["state"] = "verified"
    save()


def run_logged(argv: list[str], log: Path, env: dict, timeout: int) -> int:
    with log.open("ab") as handle:
        process = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=WNN)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(60)
            except subprocess.TimeoutExpired:
                process.kill()
            return 124


def tail(path: Path) -> str:
    try:
        return path.read_text(errors="replace").strip().splitlines()[-1][-200:]
    except (OSError, IndexError):
        return ""


def worker(slot: dict, queue: list, base_env: dict) -> None:
    while True:
        with LOCK:
            if not queue:
                return
            job = queue.pop(0)
        name = job["name"]
        record = STATE["jobs"][name]
        env = base_env | {"CUDA_VISIBLE_DEVICES": str(slot["gpu"]), "OMP_NUM_THREADS": str(slot["threads"]),
                          "MKL_NUM_THREADS": str(slot["threads"]), "WNN_THREADS": str(slot["threads"])}
        for step in job["steps"]:
            log = RUN / "logs" / f"{name}.{step['label']}.log"
            record.update(state=step["label"], gpu=slot["gpu"], step_started=time.time())
            save()
            rc = run_logged(step["argv"], log, env, step.get("timeout", 14400))
            record[f"{step['label']}_rc"] = rc
            record[f"{step['label']}_seconds"] = round(time.time() - record["step_started"], 1)
            record["last_line"] = tail(log)
            save()
            if rc != 0:
                record["state"] = f"failed_{step['label']}"
                save()
                break
        else:
            record["state"] = "done"
            save()


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "logs").mkdir(exist_ok=True)
    plan = json.loads((RUN / "plan.json").read_text())
    try:
        phase("download")
        bundle = [item for item in plan["inputs"] if item["remote_path"] == "bundle.tar.gz"]
        for item in bundle:
            download(item)
        with tarfile.open(WNN / "bundle.tar.gz") as archive:
            archive.extractall(WNN, filter="data")
        jobs = json.loads((WNN / "vast_jobs.json").read_text())
        STATE["jobs"] = {job["name"]: {"state": "queued"} for job in jobs["jobs"]}
        save()
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(download, [item for item in plan["inputs"] if item["remote_path"] != "bundle.tar.gz"]))
        phase("pip")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *jobs["pip"]], check=True,
                       stdout=(RUN / "logs/pip.log").open("wb"), stderr=subprocess.STDOUT, timeout=900)
        base_env = os.environ | {"WNN_PROJECT_ROOT": str(WNN), "PYTHONPATH": str(WNN / "src"),
                                 "PYTHONUNBUFFERED": "1", **jobs.get("env", {})}
        phase("prepare")
        for step in jobs["prepare"]:
            rc = run_logged(step["argv"], RUN / "logs" / f"prepare.{step['label']}.log",
                            base_env | {"CUDA_VISIBLE_DEVICES": "0"}, step.get("timeout", 1800))
            if rc != 0:
                raise RuntimeError(f"prepare step {step['label']} exited {rc}: {tail(RUN / 'logs' / ('prepare.' + step['label'] + '.log'))}")
        phase("jobs")
        queue = list(jobs["jobs"])
        threads = [threading.Thread(target=worker, args=(slot, queue, base_env)) for slot in jobs["slots"]]
        for index, thread in enumerate(threads):
            thread.start()
            time.sleep(jobs.get("stagger_seconds", 45) if index < len(threads) - 1 else 0)
        for thread in threads:
            thread.join()
        phase("done")
        (RUN / "DONE.json").write_text(json.dumps(STATE, indent=1, default=str))
        return 0
    except Exception as error:  # noqa: BLE001 - every failure must leave a terminal marker
        STATE["error"] = f"{type(error).__name__}: {error}"
        STATE["traceback"] = traceback.format_exc()[-4000:]
        phase("failed")
        (RUN / "FAILED.json").write_text(json.dumps(STATE, indent=1, default=str))
        return 1


if __name__ == "__main__":
    sys.exit(main())
