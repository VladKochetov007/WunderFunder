from __future__ import annotations

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


def approve_successor_profile(
    run_dir: Path, stage: int, trial_number: int, review_path: Path
) -> dict[str, object]:
    """Consume external review and make the exact native-compatible approved copy."""
    if stage not in {2, 3}:
        raise ValueError("successor approval supports stage2 and stage3 only")
    trial_dir = run_dir / f"stage{stage}" / "trials" / f"{trial_number:06d}"
    profile_dir = trial_dir / "profile"
    request_path = trial_dir / "trial_request.json"
    native_path = Path(str(read_json(request_path)["profile_receipt_path"]))
    attempt_path = profile_dir / "profile_attempt_receipt.json"
    native = read_json(native_path)
    request = read_json(request_path)
    attempt = read_json(attempt_path)
    review = read_json(review_path)
    native_sha, review_sha = file_sha256(native_path), file_sha256(review_path)
    identity = native.get("identity")
    if not isinstance(identity, Mapping) or native.get("approved") is not False:
        raise ValueError("successor native profile must remain complete and unapproved")
    identity_data = identity.get("data")
    runtime = identity.get("runtime")
    devices = runtime.get("devices") if isinstance(runtime, Mapping) else None
    expected_devices = request.get("expected_device_uuid_set")
    nvml = attempt.get("nvml_pre_profile")
    if (
        native.get("status") != "profile_complete"
        or native.get("updates") != request.get("requested_updates")
        or identity.get("config_sha256") != request.get("config_identity_sha256")
        or identity.get("source_sha256") != request.get("source_sha256")
        or not isinstance(identity_data, Mapping)
        or identity_data.get("split_sha256") != request.get("auxiliary_split_sha256")
        or identity_data.get("primary_split_sha256") != request.get("primary_split_sha256")
        or identity.get("world_size") != 1
        or not isinstance(expected_devices, list)
        or len(expected_devices) != 1
        or not isinstance(devices, list)
        or len(devices) != 1
        or not isinstance(devices[0], Mapping)
        or normalize_gpu_uuid(devices[0].get("uuid")) != normalize_gpu_uuid(expected_devices[0])
        or devices[0].get("total_memory_bytes")
        != request.get("expected_device_torch_total_memory_bytes")
        or not isinstance(nvml, Mapping)
        or nvml.get("status") != "observed"
        or nvml.get("device_uuid") not in expected_devices
        or nvml.get("total_memory_bytes") != request.get("expected_device_total_memory_bytes")
        or request.get("profile_receipt_sha256") != native_sha
    ):
        raise ValueError("native successor profile identity differs from the exact trial request")
    bindings: dict[str, object] = {
        "schema": "wnn.multistage-successor-profile-review.v1",
        "status": "approved",
        "stage": stage,
        "trial_number": trial_number,
        "native_profile_sha256": native_sha,
        "identity_sha256": canonical_json_sha256(identity),
        "trial_request_sha256": file_sha256(request_path),
        "profile_attempt_receipt_sha256": file_sha256(attempt_path),
        "config_sha256": request["config_sha256"],
        "expected_device_total_memory_bytes": request.get("expected_device_total_memory_bytes"),
        "expected_device_torch_total_memory_bytes": request.get(
            "expected_device_torch_total_memory_bytes"
        ),
        "initializer_parent_trial_result_sha256": request.get(
            "initializer_parent_trial_result_sha256"
        ),
        "auxiliary_split_sha256": request["auxiliary_split_sha256"],
        "primary_split_sha256": request["primary_split_sha256"],
        "source_sha256": request["source_sha256"],
        "initializer_path": request.get("initializer_path"),
        "initializer_sha256": request.get("initializer_sha256"),
        "teacher_cache_manifest_sha256": request.get("teacher_cache_manifest_sha256"),
    }
    if any(review.get(key) != value for key, value in bindings.items()):
        raise ValueError(
            "external successor review receipt does not bind the exact profile request"
        )
    reviewer = review.get("reviewer")
    if reviewer not in {"root", "independent"} or not isinstance(
        review.get("reviewed_at_utc"), str
    ):
        raise ValueError("successor profile requires a recorded external reviewer and timestamp")
    if attempt.get("status") != "profile_complete" or attempt.get("exit_code") != 0:
        raise ValueError("successor profile attempt is not terminal-successful")
    terminal = attempt.get("terminal_attempt")
    if (
        attempt.get("stage") != stage
        or attempt.get("trial_number") != trial_number
        or attempt.get("native_profile_sha256") != native_sha
        or attempt.get("profile_id") != f"stage{stage}-trial-{trial_number:06d}"
        or not isinstance(terminal, Mapping)
        or terminal.get("status") != "profile_complete"
        or terminal.get("exit_code") != 0
    ):
        raise ValueError("successor attempt ledger does not bind its successful native profile")
    approved = dict(native)
    approved.update(
        {
            "approved": True,
            "external_review": {
                "schema": bindings["schema"],
                "review_receipt_path": str(review_path.resolve()),
                "review_receipt_sha256": review_sha,
                "native_profile_sha256": native_sha,
                "approved_at_utc": utc_now(),
            },
        }
    )
    approved_path = profile_dir / "approved_profile_receipt.json"
    atomic_write_json(approved_path, approved, replace=False)
    record = {
        "schema": "wnn.multistage-profile-approval.v1",
        "stage": stage,
        "trial_number": trial_number,
        "native_profile_path": str(native_path.resolve()),
        "native_profile_sha256": native_sha,
        "approved_profile_path": str(approved_path.resolve()),
        "approved_profile_sha256": file_sha256(approved_path),
        "review_receipt_path": str(review_path.resolve()),
        "review_receipt_sha256": review_sha,
        "profile_attempt_receipt_path": str(attempt_path.resolve()),
        "profile_attempt_receipt_sha256": file_sha256(attempt_path),
        "reviewer": reviewer,
        "approved_at_utc": utc_now(),
        "status": "approved_for_exact_fit_identity",
    }
    approval_path = profile_dir / "approval_record.json"
    atomic_write_json(approval_path, record, replace=False)
    append_event(run_dir / "events.jsonl", {"event": "profile_approved", **record})
    return record
