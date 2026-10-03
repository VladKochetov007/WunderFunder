from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATE = Path(__file__).resolve().parent
JOBS = ROOT / "research/campaign_2026-09-29/jobs"
LOGS = ROOT / "research/campaign_2026-09-29/logs"
now = datetime.now().astimezone()
active = []
for path in JOBS.glob("*.json"):
    job = json.loads(path.read_text())
    if job.get("status") != "running":
        continue
    log = LOGS / f"{job['name']}.log"
    text = log.read_text()[-160000:] if log.exists() else ""
    completed = [json.loads(row) for row in re.findall(r"^FIT (\{.*\})$", text, re.M)]
    progress = re.findall(r"^ep (\d+) batch (\d+)/(\d+)", text, re.M)
    if progress and completed:
        epoch, batch, total = map(int, progress[-1])
        durations = [float(row["train_seconds"]) for row in completed[-2:]]
        seconds_per_epoch = sum(durations) / len(durations)
        remaining = (max(0, 4 - epoch - 1) + 1 - batch / total) * seconds_per_epoch
        low = max(0, remaining * 0.8) + 60
        high = remaining * 1.35 + 480
        estimate = {
            "training_epoch": epoch,
            "batch": batch,
            "batches_per_epoch": total,
            "recent_epoch_seconds": durations,
            "estimated_graceful_finish_local": [
                (now + timedelta(seconds=low)).isoformat(timespec="minutes"),
                (now + timedelta(seconds=high)).isoformat(timespec="minutes"),
            ],
            "assumptions": "Four epochs; recent batch pace persists; export adds1-8min. Hardware/load dependent, not a deadline.",
        }
    else:
        estimate = {
            "assumptions": "Export or insufficient fit progress: use logged service timeout as ceiling, not predicted completion."
        }
    active.append(
        {
            "name": job["name"],
            "elapsed_seconds": job.get("elapsed_seconds"),
            "peak_vram_bytes": job.get("peak_observed_vram_bytes"),
            **estimate,
        }
    )
pending = [
    line
    for line in (ROOT / "research/campaign_2026-10-01/jobs/queue6.txt").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]
result = {
    "time": now.isoformat(),
    "running": active,
    "pending_gpu_queue": pending,
    "queue_assumptions": "No queued fits remain when both lists are empty. Completed width320 and combo folds retain terminal job logs; any new bounded measurement has its own deadline and is not an active fit ETA.",
}
(STATE / "ETA.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
