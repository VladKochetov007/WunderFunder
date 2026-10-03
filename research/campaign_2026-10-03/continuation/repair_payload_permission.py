"""Acknowledge only the diagnosed exit126 wrapper permission failure."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

CONT = Path(__file__).resolve().parent
ROOT = CONT.parents[2]
JOB = "c20261003_state_drift_v2_2220"


def main() -> None:
    job_path = ROOT / f"research/campaign_2026-09-29/jobs/{JOB}.json"
    log_path = ROOT / f"research/campaign_2026-09-29/logs/{JOB}.log"
    job = json.loads(job_path.read_text())
    if job["status"] != "failed" or job["exit_code"] != 126 or job["peak_observed_vram_bytes"] != 27262976:
        raise ValueError("Different failure requires separate diagnosis")
    if log_path.read_text().strip() != "ionice: failed to execute research/campaign_2026-10-03/continuation/round6/controller_state_drift_payload.sh: Permission denied":
        raise ValueError("Expected pre-Python permission failure absent")
    wrapper = CONT / "round6/controller_state_drift_payload.sh"
    inventory = json.loads((CONT / "round6/state_drift_review_v2_binding/candidate_runtime_hashes.json").read_text())
    if not os.access(wrapper, os.X_OK) or hashlib.sha256(wrapper.read_bytes()).hexdigest() != inventory["runtime_sha256"][str(wrapper.relative_to(ROOT))]:
        raise ValueError("Executable permission or reviewed bytes not preserved")
    for unit, expected in (("wnn-campaign.slice", 11), ("wnn-campaign-gpu.slice", 0)):
        group = subprocess.run(["systemctl", "--user", "show", unit, "-p", "ControlGroup", "--value"], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
        events = (Path("/sys/fs/cgroup") / group.lstrip("/") / "memory.events").read_text()
        if int(dict(line.split() for line in events.splitlines())["oom_kill"]) != expected:
            raise ValueError("New OOM requires separate diagnosis")
    if subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], check=True, capture_output=True, text=True, timeout=10).stdout.strip():
        raise ValueError("GPU still active")
    with (CONT / "locks/admission.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ack_path = CONT / "acknowledgements.json"
        ack = json.loads(ack_path.read_text())
        events = [json.loads(line) for line in (CONT / "logs/incidents.jsonl").read_text().splitlines() if line.strip()]
        pending = [event for event in events if event["id"] not in ack["incident_ids"]]
        if len(pending) != 2 or any(not ((event["type"] == "gpu_job_failed" and event["record"] == str(job_path)) or (event["type"] == "journal_fault" and event["message"] == f"wnn-top1-{JOB}.service: Failed with result 'exit-code'.")) for event in pending):
            raise ValueError("Unexplained incident; refusing barrier clear")
        output = CONT / "runtime_incidents/payload_permission_2220"
        output.mkdir(exist_ok=False)
        for name, path in (("failed_job.json", job_path), ("failed_job.log", log_path), ("RESOURCE_FAILURE", CONT / "RESOURCE_FAILURE"), ("QUEUE_PAUSE", ROOT / "research/campaign_2026-10-01/QUEUE_PAUSE")):
            (output / name).write_bytes(path.read_bytes())
        report = {"status": "DIAGNOSED_PERMISSION_FIXED_BYTES_UNCHANGED", "time": datetime.now().astimezone().isoformat(), "Python_started": False, "CUDA_work_started": False, "new_OOM": False, "wrapper_mode": oct(wrapper.stat().st_mode & 0o777), "wrapper_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(), "incidents_acknowledged": pending, "retry_condition": "fresh receipt/new job name; exact unchanged science/runtime bytes and resource ceilings"}
        (output / "REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
        ack["incident_ids"].extend(event["id"] for event in pending)
        ack["historical_job_records"].append(f"{JOB}.json")
        ack.setdefault("diagnosed_retries", []).append(str((output / "REPORT.json").relative_to(ROOT)))
        temporary = CONT / "acknowledgements.json.tmp"
        temporary.write_text(json.dumps(ack, indent=2) + "\n")
        temporary.replace(ack_path)
        (CONT / "RESOURCE_FAILURE").unlink()
        (ROOT / "research/campaign_2026-10-01/QUEUE_PAUSE").unlink()
    print(json.dumps(report))


if __name__ == "__main__":
    main()
