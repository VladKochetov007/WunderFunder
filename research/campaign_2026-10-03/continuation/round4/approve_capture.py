"""Bind terminal independent review and live queue state to one CUDA measurement."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CONT = ROOT / "research/campaign_2026-10-03/continuation"
DEADLINE = 1791057600


def sha(path: Path) -> str:
    """Hash unchanged regular files with bounded buffers."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024**2), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"still being written: {path}")
    return digest.hexdigest()


def command(arguments: list[str]) -> str:
    """Read live machine state and reject unavailable observations."""
    return subprocess.run(arguments, check=True, capture_output=True, text=True, timeout=15).stdout


def unit(name: str) -> dict[str, str]:
    """Observe terminal queue or score service properties."""
    body = command(
        ["systemctl", "--user", "show", name, "-p", "ActiveState", "-p", "Result", "-p", "MainPID"]
    )
    values = dict(line.split("=", 1) for line in body.splitlines())
    if values != {"ActiveState": "inactive", "MainPID": "0", "Result": "success"}:
        raise ValueError(f"queue/score service not terminal successful: {name}: {values}")
    return values


def main() -> None:
    """Issue no receipt until reviewer completion and live admission prerequisites pass."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-verdict", required=True)
    parser.add_argument("--reviewer-completed-at", required=True)
    args = parser.parse_args()
    candidate, review = args.candidate.resolve(strict=True), args.review.resolve(strict=True)
    candidate.relative_to(CONT / "round4")
    review.relative_to(CONT / "round4")
    if candidate == review or args.receipt.exists() or time.time() + 605 >= DEADLINE:
        raise ValueError("separate scopes, fresh receipt and605seconds remaining required")
    completed_at = datetime.fromisoformat(args.reviewer_completed_at)
    if completed_at.tzinfo is None or not 0 <= time.time() - completed_at.timestamp() < 900:
        raise ValueError("recent timezone-aware reviewer completion observation required")
    state = json.loads((review / "STATE.json").read_text())
    if (
        not str(state.get("phase", "")).startswith("frozen")
        or state.get("verdict") != args.expected_verdict
        or not args.expected_verdict.startswith("PASS")
    ):
        raise ValueError("terminal independent PASS required; no blocked-review workaround")
    manifest = json.loads((review / "HASHES.json").read_text())
    for name, expected in manifest.items():
        path = (review / name).resolve(strict=True)
        path.relative_to(review)
        if sha(path) != expected:
            raise ValueError(f"final review manifest mismatch: {name}")
    inventory = json.loads((review / "candidate_runtime_hashes.json").read_text())
    if inventory["verdict"] != args.expected_verdict:
        raise ValueError("review inventory verdict differs")
    for name, expected in inventory["candidate_sha256"].items():
        path = (candidate / name).resolve(strict=True)
        path.relative_to(candidate)
        if sha(path) != expected:
            raise ValueError(f"reviewed candidate changed: {name}")
    for section in ("source", "runtime_sha256"):
        for name, expected in inventory[section].items():
            path = (ROOT / name).resolve(strict=True)
            path.relative_to(ROOT)
            if sha(path) != expected:
                raise ValueError(f"reviewed {section} changed: {name}")
    if not (CONT / "READY").is_file() or (CONT / "RESOURCE_FAILURE").exists():
        raise ValueError("continuation admission blocked")
    watch = json.loads((CONT / "watch_status.json").read_text())
    if (
        watch["barrier"]
        or watch["monitoring_degraded"]
        or watch["active_gpu_jobs"]
        or not 0 <= time.time() - datetime.fromisoformat(watch["time"]).timestamp() < 120
    ):
        raise ValueError("fresh healthy watcher with no active GPU job required")
    units = {
        name: unit(name)
        for name in ("wnn-codex-research-queue-v2.service", "wnn-codex-fold23-scores.service")
    }
    pending = [
        line
        for line in (ROOT / "research/campaign_2026-10-01/jobs/queue6.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    jobs = [
        json.loads(path.read_text())
        for path in (ROOT / "research/campaign_2026-09-29/jobs").glob("*.json")
    ]
    processes = command(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).strip()
    if (
        pending
        or processes
        or any(job.get("kind") == "gpu" and job.get("status") == "running" for job in jobs)
    ):
        raise ValueError("queue or GPU device not drained")
    receipt = {
        "status": "approved",
        "queue_finished": True,
        "reviewer": "Averroes/01a1019d-b660-75c1-bcfe-56fc8a15f897",
        "reviewer_completion_observed_at": args.reviewer_completed_at,
        "controller_attested_at": datetime.now().astimezone().isoformat(),
        "review_verdict": args.expected_verdict,
        "scope": "Bounded synthetic actual-architecture then fixed-fit-panel CUDA default/skip parity only; no estimator intervention, public score or historical full-fit recovery",
        "candidate_sha256": inventory["candidate_sha256"],
        "source": inventory["source"],
        "runtime_sha256": inventory["runtime_sha256"],
        "review_bindings": {
            name: sha(review / name)
            for name in ("REPORT.md", "STATE.json", "HASHES.json", "candidate_runtime_hashes.json")
        },
        "live_queue_units": units,
        "watch": watch,
        "GPU_compute_processes": processes,
        "execution_seconds": 600,
        "child_TERM_seconds": 585,
        "child_KILL_grace_seconds": 5,
        "absolute_deadline_UTC": "2026-10-03T20:00:00Z",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x") as output:
        output.write(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(args.receipt),
                "sha256": sha(args.receipt),
                "status": "approved",
                "GPU_launched": False,
            }
        )
    )


if __name__ == "__main__":
    main()
