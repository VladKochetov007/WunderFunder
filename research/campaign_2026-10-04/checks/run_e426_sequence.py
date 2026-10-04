"""Run the sealed event-clock comparison after the seed-4 GPU sequence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from run_seed4_sequence import guard, properties, read_json, require_success

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
RUNNER = CAMPAIGN / "runner"
OUTPUT = CAMPAIGN / "checks/e426_sequence_v1.json"
PREDECESSOR = "wnn-e450-sequence-v1.service"
SEED4_JOBS = (
    "e450-combo-seed4-final-v1",
    "e450-combo-head-seed4-final-v1",
    "e450-predict-c20261004-combo-s4-final-v1",
    "e450-predict-c20261004-combo-head-s4-final-v1",
)
ARMS = (
    ("e426_smoke_shared_f0_s3_2b_v2", "e426-smoke-shared-f0-s3-2b-v2"),
    ("e426_f0_constant_s3_4ep_v2", "e426-f0-constant-s3-4ep-v2"),
    ("e426_f0_aligned_s3_4ep_v2", "e426-f0-aligned-s3-4ep-v2"),
    ("e426_f0_shuffled_s3_4ep_v2", "e426-f0-shuffled-s3-4ep-v2"),
)


def save(status: str, completed: list[str], **details: object) -> None:
    """Update this queue snapshot; preserve immutable worker reports."""
    temporary = OUTPUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"status": status, "completed": completed, **details}, indent=2) + "\n"
    )
    temporary.replace(OUTPUT)


def wait_predecessor(completed: list[str]) -> None:
    """Proceed only after the observed seed-4 handle and jobs finish successfully."""
    while True:
        guard()
        observed = properties(PREDECESSOR)
        predecessor = read_json(CAMPAIGN / "checks/e450_sequence_v1.json")
        if predecessor["status"] == "fits_and_exports_complete":
            if observed.get("ActiveState") in ("active", "activating", "deactivating"):
                time.sleep(5)
                continue
            if observed.get("ExecMainStatus") != "0":
                raise RuntimeError(f"Seed-4 controller failed: {observed}")
            for attempt in SEED4_JOBS:
                require_success(attempt)
            return
        if predecessor["status"] not in ("waiting_for_e424", "running"):
            raise RuntimeError(f"Seed-4 sequence stopped: {predecessor}")
        if observed.get("ActiveState") not in ("active", "activating"):
            raise RuntimeError(f"Seed-4 handle disappeared: {observed}")
        save("waiting_for_seed4", completed, observed=observed)
        time.sleep(15)


def run_job(name: str, attempt: str, completed: list[str]) -> None:
    """Consume one sealed identity, observing its exact handle without blind retries."""
    guard()
    spec_path = CAMPAIGN / f"specs/{name}.sealed.json"
    spec = read_json(spec_path)
    record_path = RUNNER / f"attempts/{attempt}.json"
    if record_path.exists():
        if read_json(record_path)["spec"] != spec:
            raise ValueError(f"Attempt identity changed: {attempt}")
    else:
        subprocess.run(
            [
                sys.executable,
                "scripts/run_campaign.py",
                "launch",
                str(spec_path),
                "--attempt",
                attempt,
            ],
            cwd=ROOT,
            check=True,
        )
    deadline = time.monotonic() + int(spec["limits"]["timeout_seconds"]) + 240
    while True:
        guard()
        record = read_json(record_path)
        if record["status"] == "finished":
            require_success(attempt)
            break
        if record["status"] not in ("running", "reserved"):
            raise RuntimeError(f"Job stopped: {attempt}: {record['status']}")
        observed = properties(record["unit"])
        if observed.get("ActiveState") not in ("active", "activating"):
            raise RuntimeError(f"Job handle disappeared: {attempt}: {observed}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"Observation deadline for {attempt}; do not restart")
        save("running", completed, active_attempt=attempt, observed=observed)
        time.sleep(5)
    validate_result(name)
    if attempt not in completed:
        completed.append(attempt)
    save("running", completed)


def validate_result(name: str) -> None:
    """Check fixed epochs and actual updates before continuing this comparison."""
    history = read_json(ROOT / f"artifacts/runs/{name}/history.json")
    is_smoke = name.startswith("e426_smoke_")
    if len(history) != (1 if is_smoke else 4):
        raise ValueError(f"Wrong epoch count: {name}")
    for epoch, result in enumerate(history):
        if result["epoch"] != epoch or result.get("evaluation") != "withheld":
            raise ValueError(f"Epoch/evaluation policy changed: {name}")
        updates = result["update_diagnostics"]["epoch"]
        expected_batches = 2 if is_smoke else 373
        expected_attempts = expected_batches * 20
        if (
            updates["completed_batches"] != expected_batches
            or updates["optimizer_attempts"] != expected_attempts
            or updates["ema_updates"] != expected_attempts
            or updates["optimizer_completed_updates"] + updates["amp_skipped_updates"]
            != expected_attempts
        ):
            raise ValueError(f"Update counters changed: {name}: {updates}")
    if not is_smoke:
        provenance = read_json(ROOT / f"artifacts/runs/{name}/provenance.json")
        mode = name.split("_")[2]
        clock = provenance["teacher_ema_clock"]
        if clock["mode"] != mode or len(clock["epochs"]) != 4:
            raise ValueError(f"Teacher-clock identity changed: {name}")
        for epoch, row in enumerate(clock["epochs"]):
            count_hash = row["applied_count_sha256"]
            if (
                row["epoch"] != epoch
                or row["updates_per_epoch"] != 7460
                or row["total_train_events_epoch"] <= 0
                or len(count_hash) != 64
                or any(char not in "0123456789abcdef" for char in count_hash)
            ):
                raise ValueError(f"Teacher schedule changed: {name}: {row}")
        if provenance["cumulative_updates"] != history[-1]["update_diagnostics"]["cumulative"]:
            raise ValueError(f"Provenance/history update counters differ: {name}")


def main() -> None:
    """Keep the one-GPU queue moving without changing the sealed science."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists() and not args.resume:
        raise FileExistsError(OUTPUT)
    completed = read_json(OUTPUT)["completed"] if OUTPUT.exists() else []
    try:
        for name, _attempt in ARMS:
            read_json(CAMPAIGN / f"specs/{name}.sealed.json")
        wait_predecessor(completed)
        for name, attempt in ARMS:
            run_job(name, attempt, completed)
        save("fits_complete", completed, paired_score="pending")
    except Exception as error:
        save("stopped", completed, reason=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
