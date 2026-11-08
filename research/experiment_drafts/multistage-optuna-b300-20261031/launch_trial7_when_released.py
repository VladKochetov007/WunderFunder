"""Finish the eighth Stage1 candidate after the exact owned GALAM worker releases GPU4."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from wnn.multistage_receiver import process_is_owned_and_live
from wnn.multistage_search_runtime import base_environment, launch_detached_worker

ROOT = Path("/data/epanov_s/tabdata/runtime/multistage-optuna-20261031-r1")


def main() -> None:
    """Wait without signalling another process, then profile, independently review and fit."""
    root, campaign = ROOT, ROOT / "campaign-r2"
    deadline = json.loads((campaign / "campaign_deadline.json").read_bytes())
    owner = json.loads((root / "receipts/fit-r6-galam64.owner.json").read_bytes())
    identity = {key: owner[key] for key in ("pid", "birth", "pgid")}
    while process_is_owned_and_live(identity):
        if time.time() >= deadline["reserve_start_epoch"]:
            raise RuntimeError("collection reserve started before the owned GPU4 slot released")
        time.sleep(5)
    operator = root / "operator-r4"
    admission = json.loads((root / "receipts/ADMISSION_R2.json").read_bytes())
    device = admission["profile_policy"]["devices"]["4"]
    worker_id = "stage1-final-fresh-000008"
    command = [
        sys.executable,
        str(operator / "scripts/multistage_search.py"),
        "worker",
        "--run-dir",
        str(campaign),
        "--mode",
        "profile",
        "--worker-id",
        worker_id,
        "--gpu-key",
        "4",
        "--stage",
        "1",
        "--trial-number",
        "7",
    ]
    environment = base_environment(device["cuda_visible_devices"], source_root=operator)
    owner_path = campaign / "workers" / f"{worker_id}.json"
    if owner_path.exists():
        raise FileExistsError("prepared trial7 already has an owned worker")
    record = launch_detached_worker(
        command,
        cwd=operator,
        env=environment,
        receipt_path=owner_path,
        metadata={
            "worker_id": worker_id,
            "mode": "profile",
            "stage": 1,
            "gpu_key": "4",
            "trial_number": 7,
        },
    )
    launch_path = owner_path.with_name(f"{worker_id}.launch_receipt.json")
    with launch_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print("TRIAL7_PROFILE_STARTED", record["pid"], flush=True)
    while process_is_owned_and_live(record):
        if time.time() >= deadline["reserve_start_epoch"]:
            raise RuntimeError("collection reserve started while waiting for the bounded profile")
        time.sleep(3)
    trial = campaign / "stage1/trials/000007"
    request_path = trial / "trial_request.json"
    request = json.loads(request_path.read_bytes())
    auditor = root / "operator-r3/research/experiment_drafts/multistage-optuna-b300-20261031"
    review = trial / "profile/profile_review_receipt_r3.json"
    audit_command = [
        sys.executable,
        str(auditor / "review_multistage_profile_r3.py"),
        "--profile",
        request["profile_receipt_path"],
        "--trial-request",
        str(request_path),
        "--attempt-receipt",
        request["profile_attempt_receipt_path"],
        "--worker-receipt",
        request["worker_receipt_path"],
        "--deadline-receipt",
        str(campaign / "campaign_deadline.json"),
        "--policy",
        str(auditor / "PROFILE_REVIEW_POLICY_AUTOMATED_R3.json"),
        "--contract",
        str(campaign / "frozen/CONTRACT.json"),
        "--receipt",
        str(review),
    ]
    subprocess.run(audit_command, env=environment, timeout=120, check=True)
    if json.loads(review.read_bytes())["status"] != "approved":
        raise RuntimeError("independent R3 audit rejected the eighth candidate")
    subprocess.run(
        [
            sys.executable,
            str(root / "requests/launch_reviewed_native_fit.py"),
            "--runtime-root",
            str(root),
            "--trial-number",
            "7",
            "--review",
            str(review),
            "--review-sha256",
            hashlib.sha256(review.read_bytes()).hexdigest(),
        ],
        env=environment,
        timeout=120,
        check=True,
    )
    print("TRIAL7_SCIENTIFIC_FIT_LAUNCHED", flush=True)


if __name__ == "__main__":
    main()
