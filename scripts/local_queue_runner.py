"""Advance a finite local experiment queue by at most one launch per invocation.

Run from a systemd timer.  The queue directory holds ``pending/*.json`` items
processed in filename order.  Each item names an attempt id, launch resources,
a contract path, the frozen argv and optional ``after`` attempt ids that must
have finished with exit status 0.  Every invocation:

1. Moves items whose attempt finished into ``done/`` or ``failed/``.
2. Moves items whose dependency failed into ``blocked/``.
3. If no queue attempt is running, launches the first ready item through
   ``scripts/bandit_resources.py launch`` (which owns admission, the GPU lock
   and resource limits).

An attempt id that already has a durable record is never launched again.
A ``STOP`` file in the queue directory prevents new launches.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "research/campaign_2026-10-06/bandit_continuation/runtime"
ACTIVE = {"admitted", "running", "starting", "not_started_pending", "reserved", "launch_uncertain"}
MAX_ADMISSION_RETRIES = 40


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def attempt(attempt_id: str) -> dict | None:
    path = STATE / "attempts" / f"{attempt_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def outcome(record: dict | None) -> str:
    """Return none, active, success or failure for a durable attempt record."""
    if record is None:
        return "none"
    status = record.get("status")
    if status in ACTIVE or (status not in {"finished", "failed", "not_started"} and record.get("exit_status") is None):
        return "active"
    if status == "finished" and record.get("exit_status") == 0 and not record.get("failure"):
        return "success"
    return "failure"


def any_attempt_active() -> bool:
    """True when any campaign attempt (queued here or not) still holds resources."""
    for path in (STATE / "attempts").glob("*.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("status") in ACTIVE:
            return True
    return False


def admission_closed() -> bool:
    return (STATE / "STOP_ADMISSION.json").exists()


def admission_rejected(record: dict | None) -> bool:
    return bool(record) and record.get("status") == "not_started" and record.get("phase") in {"admission_failed", "admission_recheck_failed"}


def resolve(queue: Path, dependency: str) -> str:
    """Map a base attempt id to the item's current (possibly retried) attempt id."""
    for path in queue.glob("*/*.json"):
        if path.parent.name == "contracts":
            continue
        item = json.loads(path.read_text())
        if dependency in (item.get("base_attempt"), item.get("attempt")):
            return item["attempt"]
    return dependency


def log(queue: Path, row: dict) -> None:
    with (queue / "runner_log.jsonl").open("a") as handle:
        handle.write(json.dumps({"utc": now(), **row}, sort_keys=True) + "\n")


def move(item: Path, queue: Path, bucket: str) -> None:
    target = queue / bucket
    target.mkdir(exist_ok=True)
    shutil.move(str(item), target / item.name)


SLICE = Path("/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice")
MEMORY_HIGH_GUARD = 18_000_000_000


def reclaim_file_cache_for(reserve_bytes: int, queue: Path) -> None:
    """Reclaim only file cache (swappiness=0) so measured use plus the reserve fits the guard."""
    try:
        current = int((SLICE / "memory.current").read_text())
        stat = dict(line.split() for line in (SLICE / "memory.stat").read_text().splitlines())
        file_bytes = int(stat["file"])
    except (OSError, KeyError, ValueError):
        return
    want = current + reserve_bytes - (MEMORY_HIGH_GUARD - 500_000_000)
    want = min(want, file_bytes - 500_000_000)
    if want <= 0:
        return
    try:
        (SLICE / "memory.reclaim").write_text(f"{want} swappiness=0")
    except OSError as error:
        log(queue, {"event": "reclaim_partial", "requested": want, "error": str(error)})
    log(queue, {"event": "file_cache_reclaim", "requested": want, "before": current,
                "after": int((SLICE / "memory.current").read_text())})


def launch(item: dict) -> subprocess.CompletedProcess[str]:
    env = os.environ | {
        "WNN_SLICE": "wnn-campaign.slice",
        "WNN_CPU_QUOTA": "20%",
        "WNN_THREADS": "1",
        "WNN_MEM_MAX": "1500000000",
        "WNN_RUNTIME_MAX": "600s",
    }
    command = [
        str(ROOT / "scripts/limited"),
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/bandit_resources.py"),
        "--root", str(ROOT),
        "--state", str(STATE),
        "launch",
        "--attempt", item["attempt"],
        "--kind", item["kind"],
        "--cpus", str(item["cpus"]),
        "--reserve-bytes", str(item["reserve_bytes"]),
        "--profile", item["profile"],
        "--contract", str(ROOT / item["contract"]),
        "--admission-timeout", str(item.get("admission_timeout", 600)),
        "--",
        *item["argv"],
    ]
    return subprocess.run(command, capture_output=True, text=True, timeout=900, check=False, env=env, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    queue = args.queue.resolve()
    pending = queue / "pending"
    items = sorted(pending.glob("*.json"))

    running = False
    for path in items:
        item = json.loads(path.read_text())
        record = attempt(item["attempt"])
        state = outcome(record)
        if admission_rejected(record) and item.get("admission_retries", 0) < MAX_ADMISSION_RETRIES:
            # Capacity was busy; the experiment never started.  Retry under a new attempt id.
            retries = item.get("admission_retries", 0) + 1
            base_id = item.get("base_attempt", item["attempt"])
            item |= {"base_attempt": base_id, "admission_retries": retries, "attempt": f"{base_id}-a{retries}"}
            path.write_text(json.dumps(item, indent=1))
            log(queue, {"event": "admission_retry", "attempt": item["attempt"]})
            continue
        if state == "success":
            move(path, queue, "done")
            log(queue, {"event": "done", "attempt": item["attempt"]})
        elif state == "failure":
            move(path, queue, "failed")
            log(queue, {"event": "failed", "attempt": item["attempt"]})
        elif state == "active":
            running = True

    if running or (queue / "STOP").exists() or any_attempt_active() or admission_closed():
        return 0
    for path in sorted(pending.glob("*.json")):
        item = json.loads(path.read_text())
        deps = [outcome(attempt(resolve(queue, a))) for a in item.get("after", [])]
        if any(d == "failure" for d in deps):
            move(path, queue, "blocked")
            log(queue, {"event": "blocked", "attempt": item["attempt"], "after": item.get("after")})
            continue
        if any(d != "success" for d in deps):
            continue
        if attempt(item["attempt"]) is not None:
            continue
        reclaim_file_cache_for(int(item["reserve_bytes"]), queue)
        result = launch(item)
        log(queue, {"event": "launch", "attempt": item["attempt"], "rc": result.returncode,
                    "stderr_tail": result.stderr[-800:]})
        if "new admission is closed" in result.stderr:
            # The campaign barrier is closed after a resource incident; wait for its review.
            return 0
        if result.returncode != 0 and attempt(item["attempt"]) is None:
            move(path, queue, "launch_failed")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
