from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

from wnn.multistage_search import atomic_write_json, read_json
from wnn.multistage_search_runtime import base_environment, launch_detached_worker


def main() -> None:
    """Profile one prepared successor, preserve R2 rejection, then consume an R4 review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--operator-root", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=(2, 3), required=True)
    parser.add_argument("--trial-number", type=int, required=True)
    parser.add_argument("--gpu-key", required=True)
    args = parser.parse_args()
    root, operator = args.runtime_root.resolve(strict=True), args.operator_root.resolve(strict=True)
    campaign = root / "campaign-r2"
    admission = read_json(root / "receipts/ADMISSION_R2.json")
    device = admission["profile_policy"]["devices"][args.gpu_key]
    worker_id = f"r4-stage{args.stage}-trial-{args.trial_number:06d}"
    owner_path = campaign / "workers" / f"{worker_id}.json"
    if owner_path.exists():
        raise FileExistsError("profile owner already exists; retain this attempt")
    env = base_environment(str(device["cuda_visible_devices"]), source_root=operator)
    env["PYTHONPATH"] = f"{operator / 'src'}:{root / 'python-deps'}"
    env["WNN_PORTABLE_DATA_MANIFEST"] = str(root / "inputs/portable-data-r1.json")
    env["WNN_PROJECT_ROOT"] = str(root / "data")
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
        args.gpu_key,
        "--stage",
        str(args.stage),
        "--trial-number",
        str(args.trial_number),
    ]
    record = launch_detached_worker(
        command,
        cwd=operator,
        env=env,
        receipt_path=owner_path,
        metadata={
            "worker_id": worker_id,
            "mode": "profile",
            "stage": args.stage,
            "trial_number": args.trial_number,
            "gpu_key": args.gpu_key,
        },
    )
    atomic_write_json(
        owner_path.with_name(f"{worker_id}.launch_receipt.json"), record, replace=False
    )
    print("SUCCESSOR_PROFILE_STARTED", args.stage, args.trial_number, record["pid"], flush=True)
    # Reap our own detached child; /proc existence alone also counts zombies as live.
    _, status = os.waitpid(int(record["pid"]), 0)
    print("ORIGINAL_PROFILE_WORKER_EXIT", os.waitstatus_to_exitcode(status), flush=True)
    trial = campaign / f"stage{args.stage}/trials/{args.trial_number:06d}"
    request_path = trial / "trial_request.json"
    request = read_json(request_path)
    review_root = operator / "research/experiment_drafts/multistage-optuna-b300-20261031"
    review = trial / "profile/profile_review_receipt_r4.json"
    audit = [
        sys.executable,
        str(review_root / "review_multistage_profile_r4.py"),
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
        str(review_root / "PROFILE_REVIEW_POLICY_AUTOMATED_R4.json"),
        "--contract",
        str(campaign / "frozen/CONTRACT.json"),
        "--receipt",
        str(review),
    ]
    subprocess.run(audit, env=env, timeout=150, check=True)
    if read_json(review).get("status") != "approved":
        raise RuntimeError("actual successor profile was rejected by R4")
    launch = [
        sys.executable,
        str(review_root / "launch_reviewed_successor_fit.py"),
        "launch",
        "--run-dir",
        str(campaign),
        "--stage",
        str(args.stage),
        "--trial-number",
        str(args.trial_number),
        "--review",
        str(review),
        "--review-sha256",
        hashlib.sha256(review.read_bytes()).hexdigest(),
        "--operator-root",
        str(operator),
    ]
    result = subprocess.run(
        launch, env=env, timeout=150, capture_output=True, text=True, check=True
    )
    print(result.stdout, flush=True)
    print("SUCCESSOR_FIT_LAUNCHED", args.stage, args.trial_number, flush=True)


if __name__ == "__main__":
    main()
