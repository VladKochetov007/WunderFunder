from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_search import append_event, file_sha256, read_json
from wnn.multistage_search_budget import utc_now


def run_profile_reviewer(run_dir: Path, trial_number: int, *, stage: int = 1) -> dict[str, object]:
    """Run the pinned deterministic profile checker for one immutable trial bundle."""
    if stage not in {1, 2, 3}:
        raise ValueError("profile review stage must be one, two, or three")
    run_manifest = read_json(run_dir / "run.json")
    trial_dir = run_dir / f"stage{stage}" / "trials" / f"{trial_number:06d}"
    request_path = trial_dir / "trial_request.json"
    request = read_json(request_path)
    if stage > 1 and (request.get("stage") != stage or request.get("trial_number") != trial_number):
        raise ValueError("successor review request does not match its stage/trial directory")
    policy_path = Path(str(run_manifest["profile_checker_policy"]["path"]))
    reviewer_path = Path(str(run_manifest["profile_reviewer_script"]["path"]))
    if file_sha256(reviewer_path) != run_manifest["profile_reviewer_script"]["sha256"]:
        raise ValueError("independent reviewer source changed after campaign preparation")
    receipt_name = (
        "profile_review_receipt.json" if stage == 1 else "successor_profile_review_receipt.json"
    )
    receipt_path = trial_dir / "profile" / receipt_name
    if receipt_path.exists():
        raise FileExistsError("profile reviewer receipt already exists; reruns are disabled")
    attempt_path = Path(str(request["profile_attempt_receipt_path"]))
    worker_path = Path(str(request["worker_receipt_path"]))
    deadline_path = (
        Path(str(request["deadline_receipt_path"]))
        if stage > 1
        else run_dir / "campaign_deadline.json"
    )
    argv = [
        sys.executable,
        str(reviewer_path),
        "--profile",
        str(request["profile_receipt_path"]),
        "--trial-request",
        str(request_path),
        "--attempt-receipt",
        str(attempt_path),
        "--worker-receipt",
        str(worker_path),
        "--deadline-receipt",
        str(deadline_path),
        "--policy",
        str(policy_path),
        "--contract",
        str(run_manifest["contract"]["path"]),
        "--receipt",
        str(receipt_path),
    ]
    environment = os.environ.copy()
    thread_environment = request.get("effective_thread_environment")
    if not isinstance(thread_environment, Mapping):
        raise ValueError("trial request is missing its effective CPU thread environment")
    environment.update({str(key): str(value) for key, value in thread_environment.items()})
    completed = subprocess.run(
        argv,
        cwd=reviewer_path.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    log_path = trial_dir / "profile" / "independent_review.log"
    log_path.write_text(
        f"exit_code={completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode != 0 or not receipt_path.is_file():
        raise RuntimeError("independent profile checker failed; inspect its captured log")
    receipt = read_json(receipt_path)
    status = receipt.get("status")
    if stage == 1 and status not in {"approved", "pending_anchor_review"}:
        raise ValueError("independent profile checker rejected the exact profile")
    if stage > 1 and (
        status != "approved"
        or receipt.get("stage") != stage
        or receipt.get("trial_number") != trial_number
        or receipt.get("reviewer") != "independent"
    ):
        raise ValueError("independent reviewer did not approve this exact successor profile")
    append_event(
        run_dir / "events.jsonl",
        {
            "event": "profile_independently_reviewed",
            "at_utc": utc_now(),
            "stage": stage,
            "trial_number": trial_number,
            "review_receipt_path": str(receipt_path.resolve()),
            "review_receipt_sha256": file_sha256(receipt_path),
            "status": status,
        },
    )
    return receipt | {"review_receipt_sha256": file_sha256(receipt_path)}
