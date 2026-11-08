from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from wnn.multistage_gpu_identity import normalize_gpu_uuid
from wnn.multistage_search import (
    append_event,
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    read_json,
)
from wnn.multistage_search_budget import utc_now


def approve_profile(
    run_dir: Path, trial_number: int, review_receipt_path: Path
) -> dict[str, object]:
    """Create a reviewed approval copy without changing the native profile bytes."""
    trial_dir = run_dir / "stage1" / "trials" / f"{trial_number:06d}"
    request_path = trial_dir / "trial_request.json"
    request, review = read_json(request_path), read_json(review_receipt_path.resolve(strict=True))
    native_path = Path(str(request["profile_receipt_path"]))
    attempt_path = Path(str(request["profile_attempt_receipt_path"]))
    plan_path = Path(str(request["trial_plan_path"]))
    native = read_json(native_path)
    attempt = read_json(attempt_path)
    native_sha, review_sha = file_sha256(native_path), file_sha256(review_receipt_path)
    worker_receipt_path = Path(str(request["worker_receipt_path"]))
    worker_receipt = read_json(worker_receipt_path)
    if native.get("approved") is not False or native.get("status") != "profile_complete":
        raise ValueError("approval requires one completed, still-unapproved native profile")
    if (
        attempt.get("schema") != "wnn.multistage-profile-attempt.v1"
        or attempt.get("status") != "profile_complete"
        or attempt.get("exit_code") != 0
        or attempt.get("trial_number") != trial_number
        or attempt.get("native_profile_sha256") != native_sha
        or attempt.get("profile_id") != f"stage1-trial-{trial_number:06d}"
        or attempt.get("trial_plan_sha256") != request.get("trial_plan_sha256")
        or attempt.get("worker_receipt_sha256") != file_sha256(worker_receipt_path)
    ):
        raise ValueError("profile attempt receipt is incomplete or belongs to another trial")
    profile_identity = native.get("identity")
    runtime = profile_identity.get("runtime") if isinstance(profile_identity, Mapping) else None
    devices = runtime.get("devices") if isinstance(runtime, Mapping) else None
    expected_uuids = request.get("expected_device_uuid_set")
    nvml = attempt.get("nvml_pre_profile")
    if (
        not isinstance(devices, list)
        or len(devices) != 1
        or not isinstance(devices[0], Mapping)
        or not isinstance(expected_uuids, list)
        or len(expected_uuids) != 1
        or normalize_gpu_uuid(devices[0].get("uuid")) != normalize_gpu_uuid(expected_uuids[0])
        or devices[0].get("total_memory_bytes")
        != request.get("expected_device_torch_total_memory_bytes")
        or not isinstance(nvml, Mapping)
        or nvml.get("status") != "observed"
        or nvml.get("device_uuid") not in expected_uuids
        or nvml.get("total_memory_bytes") != request.get("expected_device_total_memory_bytes")
    ):
        raise ValueError("Torch and NVML device totals do not match their separate observations")
    run_manifest = read_json(run_dir / "run.json")
    limits = run_manifest["limits"]
    checker_policy_record = run_manifest["profile_checker_policy"]
    checker_policy_path = Path(str(checker_policy_record["path"]))
    if file_sha256(checker_policy_path) != checker_policy_record.get("sha256"):
        raise ValueError("frozen independent profile policy changed after preparation")
    if (
        plan_path != trial_dir / "trial_plan.json"
        or file_sha256(plan_path) != request.get("trial_plan_sha256")
        or attempt_path != trial_dir / "profile" / "profile_attempt_receipt.json"
    ):
        raise ValueError("profile request does not bind its immutable plan and attempt paths")
    if file_sha256(Path(str(request["worker_owner_receipt_path"]))) != request.get(
        "worker_owner_receipt_sha256"
    ):
        raise ValueError("immutable worker owner receipt changed after profile completion")
    if worker_receipt.get("cpu_thread_env") != request.get("cpu_thread_env"):
        raise ValueError("profile process thread environment differs from the request")
    reviewer_record = run_manifest.get("profile_reviewer_script")
    if (
        not isinstance(reviewer_record, Mapping)
        or request.get("reviewer_script_sha256") != reviewer_record.get("sha256")
        or file_sha256(Path(str(reviewer_record.get("path")))) != reviewer_record.get("sha256")
        or review.get("reviewer") != "multistage_independent_critic"
    ):
        raise ValueError("review receipt is not bound to the frozen independent checker")
    actual_wall = attempt.get("actual_wall_seconds")
    if (
        not isinstance(actual_wall, (int, float))
        or not math.isfinite(float(actual_wall))
        or actual_wall <= 0
        or actual_wall > limits["max_profile_wall_seconds"]
    ):
        raise ValueError("profile exceeded its admitted wall bound")
    if file_sha256(Path(str(attempt["campaign_deadline_path"]))) != attempt.get(
        "campaign_deadline_sha256"
    ):
        raise ValueError("campaign deadline identity changed after the profile attempt")
    if file_sha256(Path(str(attempt["budget_snapshot_path"]))) != attempt.get(
        "budget_snapshot_sha256"
    ):
        raise ValueError("profile attempt budget snapshot changed")
    terminal = attempt.get("terminal_attempt")
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("status") != "profile_complete"
        or terminal.get("exit_code") != 0
        or terminal.get("profile_id") != attempt.get("profile_id")
    ):
        raise ValueError("profile attempt terminal ledger row is not successful")
    if (
        attempt.get("profile_attempt_count_at_completion") > limits["max_profile_attempts"]
        or attempt.get("fit_attempt_count_at_completion") > limits["max_fit_attempts"]
    ):
        raise ValueError("profile attempt completed beyond a frozen campaign cap")
    deadline = read_json(run_dir / "campaign_deadline.json")
    reserve_start = deadline.get("reserve_start_epoch")
    finished_epoch = worker_receipt.get("finished_at_epoch")
    started_epoch = worker_receipt.get("started_at_epoch")
    if (
        not isinstance(reserve_start, (int, float))
        or not isinstance(finished_epoch, (int, float))
        or not isinstance(started_epoch, (int, float))
        or finished_epoch > reserve_start
        or abs((finished_epoch - started_epoch) - actual_wall) > 0.01
    ):
        raise ValueError("profile wall receipt is outside the campaign deadline or inconsistent")
    rank_logs = request.get("rank_log_refs")
    if not isinstance(rank_logs, list) or len(rank_logs) != request.get("requested_world_size"):
        raise ValueError("profile trial request does not bind every rank log")
    for ref in rank_logs:
        if not isinstance(ref, Mapping) or file_sha256(Path(str(ref.get("path")))) != ref.get(
            "sha256"
        ):
            raise ValueError("a rank log changed after profile review")
    checker_policy = read_json(Path(str(run_manifest["profile_checker_policy"]["path"])))
    anchor_ids = checker_policy.get("manual_anchor_ids", [])
    if not isinstance(anchor_ids, list):
        raise ValueError("frozen profile policy has malformed manual anchor ids")
    anchor_id = request.get("anchor_id")
    requires_anchor_receipt = isinstance(anchor_id, str) and anchor_id in anchor_ids
    anchor_path = trial_dir / "profile" / "anchor_review_receipt.json"
    anchor_sha: str | None = None
    if requires_anchor_receipt:
        if not anchor_path.is_file():
            raise ValueError("root anchor review receipt is required for this anchor")
        anchor = read_json(anchor_path)
        anchor_sha = file_sha256(anchor_path)
        anchor_bindings = {
            "schema": "wnn.multistage-anchor-review.v1",
            "status": "approved",
            "anchor_id": anchor_id,
            "native_profile_sha256": native_sha,
            "review_receipt_sha256": review_sha,
            "trial_request_sha256": file_sha256(request_path),
            "reviewer": "root",
        }
        for key, value in anchor_bindings.items():
            if anchor.get(key) != value:
                raise ValueError(f"root anchor review receipt does not bind {key}")
        if not isinstance(anchor.get("reviewed_at_utc"), str):
            raise ValueError("root anchor receipt is missing its review timestamp")
    elif anchor_path.exists():
        raise ValueError("unexpected root anchor review receipt for a nonanchor trial")
    expected = {
        "schema": "wnn.multistage-profile-review.v1",
        "status": "pending_anchor_review" if requires_anchor_receipt else "approved",
        "native_profile_sha256": native_sha,
        "identity_sha256": canonical_json_sha256(native.get("identity")),
        "trial_request_sha256": file_sha256(request_path),
        "config_sha256": request.get("config_sha256"),
        "aux_split_sha256": request.get("auxiliary_split_sha256"),
        "primary_split_sha256": request.get("primary_split_sha256"),
        "profile_policy_sha256": request.get("profile_policy_sha256"),
        "worker_receipt_sha256": file_sha256(Path(str(request["worker_receipt_path"]))),
        "deadline_receipt_sha256": file_sha256(run_dir / "campaign_deadline.json"),
        "profile_attempt_receipt_sha256": file_sha256(attempt_path),
        "expected_device_uuid_set": request.get("expected_device_uuid_set"),
        "expected_device_total_memory_bytes": request.get("expected_device_total_memory_bytes"),
        "expected_device_torch_total_memory_bytes": request.get(
            "expected_device_torch_total_memory_bytes"
        ),
        "cpu_thread_env": request.get("cpu_thread_env"),
        "effective_thread_environment": request.get("effective_thread_environment"),
        "worker_owned_process": request.get("worker_owned_process"),
        "worker_owned_process_command_sha256": request.get("worker_owned_process_command_sha256"),
        "profile_child_process": request.get("profile_child_process"),
        "profile_child_process_command_sha256": request.get("profile_child_process_command_sha256"),
        "worker_owner_receipt_sha256": request.get("worker_owner_receipt_sha256"),
        "anchor_id": anchor_id,
    }
    for key, value in expected.items():
        if review.get(key) != value:
            raise ValueError(f"independent profile review receipt does not bind {key}")
    approved = dict(native)
    approved.update(
        {
            "approved": True,
            "external_review": {
                "schema": review["schema"],
                "review_receipt_path": str(review_receipt_path.resolve()),
                "review_receipt_sha256": review_sha,
                "native_profile_sha256": native_sha,
                "anchor_review_receipt_path": str(anchor_path.resolve()) if anchor_sha else None,
                "anchor_review_receipt_sha256": anchor_sha,
                "approved_at_utc": utc_now(),
            },
        }
    )
    approved_path = trial_dir / "profile" / "approved_profile_receipt.json"
    atomic_write_json(approved_path, approved, replace=False)
    approval_record = {
        "schema": "wnn.multistage-profile-approval.v1",
        "trial_number": trial_number,
        "native_profile_path": str(native_path),
        "native_profile_sha256": native_sha,
        "approved_profile_path": str(approved_path),
        "approved_profile_sha256": file_sha256(approved_path),
        "review_receipt_path": str(review_receipt_path.resolve()),
        "review_receipt_sha256": review_sha,
        "profile_attempt_receipt_path": str(attempt_path.resolve()),
        "profile_attempt_receipt_sha256": file_sha256(attempt_path),
        "anchor_review_receipt_path": str(anchor_path.resolve()) if anchor_sha else None,
        "anchor_review_receipt_sha256": anchor_sha,
        "approved_at_utc": utc_now(),
        "status": "approved_for_exact_fit_identity",
    }
    atomic_write_json(
        trial_dir / "profile" / "approval_record.json", approval_record, replace=False
    )
    append_event(run_dir / "events.jsonl", {"event": "profile_approved", **approval_record})
    return approval_record
