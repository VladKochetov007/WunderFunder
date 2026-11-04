"""Independently verify exact-identity B300 engineering profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GIB = 1 << 30
POLICY_NAME = "PROFILE_REVIEW_POLICY_R1.json"


def sha256_file(path: Path) -> str:
    """Hash a regular file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash JSON data using the repository's canonical serialization."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from an existing non-symlink file."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _same_path(path_text: object, expected: Path) -> bool:
    return isinstance(path_text, str) and Path(path_text).resolve() == expected.resolve()


def _add(checks: list[dict[str, object]], name: str, passed: bool, evidence: object = None) -> None:
    checks.append({"check": name, "passed": passed, "evidence": evidence})


def review_profile(
    *,
    profile_path: Path,
    request_path: Path,
    attempt_path: Path,
    worker_path: Path,
    deadline_path: Path,
    policy_path: Path,
    contract_path: Path,
) -> dict[str, object]:
    """Return a deterministic engineering decision with exact artifact bindings."""
    profile = read_json(profile_path)
    request = read_json(request_path)
    attempt = read_json(attempt_path)
    worker = read_json(worker_path)
    worker_owner_path_text = request.get("worker_owner_receipt_path")
    worker_owner_path = (
        Path(worker_owner_path_text)
        if isinstance(worker_owner_path_text, str)
        else Path(".") / "missing-worker-owner"
    )
    worker_owner = _read_optional_json(str(worker_owner_path))
    deadline = read_json(deadline_path)
    policy = read_json(policy_path)
    contract = read_json(contract_path)
    checks: list[dict[str, object]] = []

    profile_sha = sha256_file(profile_path)
    request_sha = sha256_file(request_path)
    attempt_sha = sha256_file(attempt_path)
    worker_sha = sha256_file(worker_path)
    deadline_sha = sha256_file(deadline_path)
    policy_sha = sha256_file(policy_path)
    identity = _mapping(profile.get("identity"))
    identity_sha = canonical_sha256(identity)
    data = _mapping(identity.get("data"))
    runtime = _mapping(identity.get("runtime"))
    precision = _mapping(runtime.get("precision"))
    devices = runtime.get("devices")
    expected_uuids = request.get("expected_device_uuid_set")
    expected_uuid_set = set(expected_uuids) if isinstance(expected_uuids, list) else set()
    errors: list[str] = []

    def check(name: str, passed: bool, evidence: object = None) -> None:
        _add(checks, name, passed, evidence)
        if not passed:
            errors.append(name)

    check("native_profile_schema", profile.get("receipt_schema") == "wnn.sequence-profile.v1")
    check("native_profile_complete", profile.get("status") == "profile_complete")
    check("native_profile_unapproved", profile.get("approved") is False)
    check("native_profile_hash", request.get("profile_receipt_sha256") == profile_sha)
    check("native_profile_path", _same_path(request.get("profile_receipt_path"), profile_path))
    check("identity_digest", request.get("profile_identity_sha256") == identity_sha)
    check("trial_request_schema", request.get("schema") == "wnn.multistage-trial-plan.v1")
    check("request_trial_number", request.get("trial_number") == attempt.get("trial_number"))
    check("config_file_hash", _hash_matches(request.get("config_path"), request.get("config_sha256")))
    config = _read_optional_json(request.get("config_path"))
    check("native_config_identity", identity.get("config_sha256") == canonical_sha256(config))
    check("primary_split_contract", _contract_primary_sha(contract) == request.get("primary_split_sha256"))
    check("primary_fit_boundary", _contract_fit_counts(contract) == (10325, 0, 1873))
    check(
        "profile_policy_hash",
        request.get("profile_checker_policy_sha256") == policy_sha
        and request.get("profile_policy_sha256") == policy_sha,
    )
    check("policy_schema", policy.get("schema") == "wnn.multistage-profile-review-policy.v1")

    aux_path_text = request.get("auxiliary_split_manifest")
    aux_path = Path(aux_path_text) if isinstance(aux_path_text, str) else Path(".") / "missing"
    aux_manifest = _read_optional_json(aux_path_text)
    check("auxiliary_split_hash", _hash_matches(aux_path_text, request.get("auxiliary_split_sha256")))
    check("native_auxiliary_split", data.get("split_sha256") == request.get("auxiliary_split_sha256"))
    check(
        "auxiliary_train_only_control_manifest",
        aux_manifest.get("chronology_basis")
        == "none; deterministic whole-component auxiliary partition, not a time split"
        and _mapping(aux_manifest.get("parent")).get("sha256")
        == request.get("primary_split_sha256")
        and _mapping(aux_manifest.get("fit")).get("valid") == []
        and _mapping(aux_manifest.get("evaluation")).get("source") == "train",
    )
    expected_fit_members = _member_pairs(_mapping(aux_manifest.get("fit")).get("train"))
    expected_eval_members = _member_pairs(_mapping(aux_manifest.get("evaluation")).get("train"))
    check(
        "native_fit_membership",
        data.get("fit_members") == expected_fit_members
        and _all_train_members(data.get("fit_members")),
    )
    check(
        "native_control_membership",
        data.get("evaluation_members") == expected_eval_members
        and _all_train_members(data.get("evaluation_members")),
    )
    check("native_control_role", data.get("evaluation_role") == "control")
    check("native_control_source", data.get("evaluation_source") == "train")
    check("native_control_mask", data.get("evaluation_mask") == "all_after_warmup")
    check("native_fit_mask", data.get("training_mask") == "all_after_warmup")
    check("no_teacher_for_label_only_anchor", data.get("teacher") is None)
    check("no_primary_valid_fit_members", _all_train_members(data.get("fit_members")))
    check("profile_did_not_evaluate", _mapping(profile.get("evaluation_annotation")).get("no_eval_during_fit") is True)
    check("profile_has_no_validation_reporting_key", "validation_reporting" not in _mapping(profile.get("evaluation_annotation")))

    check("exact_world_size", request.get("requested_world_size") == 1 and request.get("world_size") == 1 and identity.get("world_size") == 1 and runtime.get("world_size") == 1)
    check("strict_fp32_mode", precision.get("mode") == "fp32" and _mapping(config.get("model")).get("precision_mode", config.get("precision_mode")) == "fp32" and config.get("precision_mode") == "fp32")
    check(
        "strict_fp32_runtime_flags",
        runtime.get("matmul_precision") == "highest"
        and runtime.get("matmul_tf32") is False
        and runtime.get("cudnn_tf32") is False
        and runtime.get("autocast") is False
        and runtime.get("amp_scaler") is False
        and precision.get("parameter_dtype") == "float32"
        and precision.get("gradient_dtype") == "float32"
        and precision.get("optimizer_state_dtype") == "float32"
        and precision.get("cublas_emulate_single_precision") in (None, False),
    )
    check("profile_config_no_kd_or_hint", config.get("kd_weight") == 0 and config.get("hint_weight") == 0)
    check("profile_updates_match", request.get("requested_updates") == 50 and request.get("requested_profile_updates") == 50 and profile.get("updates") == 50)
    check("profile_written_no_checkpoint", profile.get("scientific_checkpoint_written") is False)
    check("profile_errors_empty", profile.get("errors") == [] and profile.get("error_count") == 0)
    losses = _mapping(profile.get("losses"))
    check("profile_losses_finite", set(losses) == {"label", "distillation", "hint", "total"} and all(_finite(v) and float(v) >= 0 for v in losses.values()))

    device_records = devices if isinstance(devices, list) else []
    observed_uuids = {item.get("uuid") for item in device_records if isinstance(item, dict)}
    check("exact_device_uuid", len(device_records) == 1 and observed_uuids == expected_uuid_set and len(expected_uuid_set) == 1)
    device_total = request.get("expected_device_total_memory_bytes")
    check("exact_device_total_memory", len(device_records) == 1 and device_records[0].get("total_memory_bytes") == device_total)
    check("device_is_b300", len(device_records) == 1 and "B300" in str(device_records[0].get("name", "")).upper())
    memory = profile.get("memory_by_rank")
    memory_rows = memory if isinstance(memory, list) else []
    check("one_rank_memory_record", len(memory_rows) == 1 and _mapping(memory_rows[0]).get("rank") == 0)
    row = _mapping(memory_rows[0]) if memory_rows else {}
    allocated, reserved, host_peak = (row.get(key) for key in ("peak_allocated_bytes", "peak_reserved_bytes", "host_peak_rss_bytes"))
    check("gpu_allocated_cap", _positive_int(allocated) and allocated <= 64 * GIB)
    check("gpu_reserved_cap", _positive_int(reserved) and reserved <= 64 * GIB and reserved >= allocated)
    check("host_rss_cap", _positive_int(host_peak) and host_peak <= 8 * GIB)
    nvml = _mapping(attempt.get("nvml_pre_profile"))
    nvml_free, nvml_total = nvml.get("free_memory_bytes"), nvml.get("total_memory_bytes")
    check("preprofile_nvml_snapshot", nvml.get("status") == "observed" and nvml.get("device_uuid") in expected_uuid_set)
    check("preprofile_total_matches", nvml_total == device_total)
    check("preprofile_headroom", _positive_int(nvml_free) and _positive_int(reserved) and nvml_free >= 32 * GIB + reserved)
    check("postpeak_headroom", _positive_int(nvml_free) and _positive_int(reserved) and nvml_free - reserved >= 32 * GIB)
    cpu_env = request.get("cpu_thread_env")
    effective_env = {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
    }
    worker_env = worker.get("effective_thread_environment")
    check(
        "cpu_thread_bound",
        cpu_env == {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
        and request.get("effective_thread_environment") == effective_env
        and worker.get("cpu_thread_env") == cpu_env
        and worker_env == effective_env,
    )

    check("source_hash_identity", identity.get("source_sha256") == request.get("source_sha256"))
    source_files = identity.get("source_files")
    check(
        "source_file_set_identity",
        isinstance(source_files, dict)
        and canonical_sha256(source_files) == request.get("source_files_sha256"),
    )
    check(
        "source_script_identity",
        isinstance(source_files, dict)
        and source_files.get("scripts/train_sequence.py") == request.get("source_script_sha256"),
    )
    check("runtime_policy_identity", isinstance(request.get("runtime_policy_sha256"), str) and len(str(request.get("runtime_policy_sha256"))) == 64)
    check("attempt_schema", attempt.get("schema") == "wnn.multistage-profile-attempt.v1")
    check("attempt_status", attempt.get("status") == "profile_complete" and attempt.get("exit_code") == 0)
    check("attempt_native_hash", attempt.get("native_profile_sha256") == profile_sha)
    check("attempt_request_plan_hash", attempt.get("trial_plan_sha256") == request.get("trial_plan_sha256"))
    check("attempt_worker_hash", attempt.get("worker_receipt_sha256") == worker_sha and _same_path(attempt.get("worker_receipt_path"), worker_path))
    check("worker_owner_receipt_hash", _hash_matches(worker_owner_path_text, request.get("worker_owner_receipt_sha256")))
    check("attempt_deadline_hash", attempt.get("campaign_deadline_sha256") == deadline_sha)
    check("campaign_deadline_hash", request.get("deadline_receipt_sha256") == deadline_sha)
    check("attempt_counter_caps", _attempt_counts_within_caps(attempt, policy))
    check("deadline_schema", deadline.get("schema") == "wnn.multistage-deadline.v1")
    wall = attempt.get("actual_wall_seconds")
    start, end = worker.get("started_at_epoch"), worker.get("finished_at_epoch")
    check("worker_success", worker.get("status") == "completed" and worker.get("exit_code") == 0 and worker.get("timed_out") is False)
    check("profile_wall_bound", _finite(wall) and 0 < wall <= 180 and _finite(start) and _finite(end) and abs(float(wall) - (float(end) - float(start))) <= 1.0)
    reserve_cutoff = deadline.get("reserve_start_epoch")
    check("campaign_cutoff", _finite(end) and _finite(reserve_cutoff) and float(end) <= float(reserve_cutoff))
    check("campaign_wall", _finite(end) and _finite(deadline.get("deadline_epoch")) and float(end) <= float(deadline["deadline_epoch"]))

    owner, child = _mapping(worker.get("owner")), _mapping(worker.get("child"))
    check("owned_process_identities", _process_identity(owner) and _process_identity(child) and child.get("pgid") == child.get("pid"))
    check(
        "detached_worker_owner_binding",
        _same_process(request.get("worker_owned_process"), owner)
        and _same_process(worker_owner, _mapping(request.get("worker_owned_process")))
        and worker_owner.get("command_sha256") == request.get("worker_owned_process_command_sha256"),
    )
    check(
        "profile_child_binding",
        _same_process(request.get("profile_child_process"), child)
        and worker.get("command_sha256") == request.get("profile_child_command_sha256"),
    )
    command = worker.get("command")
    command_ok = isinstance(command, list) and all(isinstance(value, str) for value in command)
    if command_ok:
        command_bytes = json.dumps(command, separators=(",", ":")).encode("utf-8")
        command_ok = hashlib.sha256(command_bytes).hexdigest() == worker.get("command_sha256")
        command_ok = command_ok and _command_matches_profile(command, request)
        forbidden = ("nvidia-cuda-mps-control", "set_active_thread_percentage", "--gpu-reset", "killall", "pkill")
        command_ok = command_ok and not any(token in " ".join(command).lower() for token in forbidden)
    check("owned_profile_command", command_ok)

    rank_checks, rank_refs = _validate_rank_logs(request.get("rank_log_refs"), request.get("requested_updates", 0))
    for name, passed, evidence in rank_checks:
        check(name, passed, evidence)

    request_anchor = request.get("anchor_id")
    manual_anchors = policy.get("manual_anchor_ids", [])
    is_manual_anchor = isinstance(request_anchor, str) and request_anchor in manual_anchors
    check("anchor_identity", request_anchor is None or is_manual_anchor)
    status = "rejected" if errors else "pending_anchor_review" if is_manual_anchor else "approved"
    return {
        "schema": "wnn.multistage-profile-review.v1",
        "status": status,
        "reviewer": "multistage_independent_critic",
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
        "native_profile_sha256": profile_sha,
        "identity_sha256": identity_sha,
        "trial_request_sha256": request_sha,
        "config_sha256": request.get("config_sha256"),
        "aux_split_sha256": request.get("auxiliary_split_sha256"),
        "primary_split_sha256": request.get("primary_split_sha256"),
        "profile_policy_sha256": policy_sha,
        "worker_receipt_sha256": worker_sha,
        "deadline_receipt_sha256": deadline_sha,
        "profile_attempt_receipt_sha256": attempt_sha,
        "source_sha256": request.get("source_sha256"),
        "runtime_identity_sha256": canonical_sha256(runtime),
        "rank_log_refs": rank_refs,
        "anchor_id": request_anchor,
        "checks": checks,
        "failed_checks": errors,
        "anchor_review_receipt_required": is_manual_anchor,
        "native_profile_preserved": True,
        "scientific_candidate_promotion": False,
    }


def _read_optional_json(path_text: object) -> dict[str, Any]:
    if not isinstance(path_text, str):
        return {}
    try:
        return read_json(Path(path_text))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _hash_matches(path_text: object, expected: object) -> bool:
    if not isinstance(path_text, str) or not isinstance(expected, str):
        return False
    path = Path(path_text)
    try:
        return path.is_file() and not path.is_symlink() and sha256_file(path) == expected
    except OSError:
        return False


def _contract_primary_sha(contract: dict[str, Any]) -> object:
    return _mapping(contract.get("data_policy")).get("primary_split_sha256")


def _contract_fit_counts(contract: dict[str, Any]) -> tuple[object, object, object]:
    data = _mapping(contract.get("data_policy"))
    return (
        data.get("primary_fit_train_sequences"),
        data.get("primary_fit_valid_sequences"),
        data.get("primary_eval_valid_sequences"),
    )


def _member_pairs(value: object) -> list[list[object]]:
    if not isinstance(value, list):
        return []
    return [["train", group] for group in value if type(group) is int]


def _all_train_members(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, list) and len(item) == 2 and item[0] == "train" and type(item[1]) is int
        for item in value
    )


def _process_identity(value: dict[str, Any]) -> bool:
    return (
        _positive_int(value.get("pid"))
        and _positive_int(value.get("pgid"))
        and isinstance(value.get("birth"), (str, int))
        and str(value.get("birth")) not in ("", "0")
    )


def _same_process(expected: object, observed: dict[str, Any]) -> bool:
    """Compare a request-bound PID/birth/PGID tuple exactly."""
    return isinstance(expected, dict) and all(
        expected.get(key) == observed.get(key) for key in ("pid", "birth", "pgid")
    )


def _attempt_counts_within_caps(attempt: dict[str, Any], policy: dict[str, Any]) -> bool:
    campaign = _mapping(policy.get("campaign_bounds"))
    try:
        return (
            attempt.get("max_fit_attempts") == campaign.get("max_fit_attempts")
            and attempt.get("max_profile_attempts") == campaign.get("max_profile_attempts")
            and 1 <= int(attempt["profile_attempt_count_at_completion"]) <= int(campaign["max_profile_attempts"])
            and 0 <= int(attempt["fit_attempt_count_at_completion"]) <= int(campaign["max_fit_attempts"])
            and isinstance(attempt.get("stage_fit_caps"), dict)
            and set(attempt["stage_fit_caps"]) == {"1", "2", "3", "4"}
            and all(type(value) is int and value >= 0 for value in attempt["stage_fit_caps"].values())
            and sum(attempt["stage_fit_caps"].values()) <= int(campaign["max_fit_attempts"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _command_matches_profile(command: list[str], request: dict[str, Any]) -> bool:
    if not command or "python" not in Path(command[0]).name.lower():
        return False
    for flag, expected in (
        ("--config", request.get("config_path")),
        ("--data-root", request.get("data_root")),
        ("--split-manifest", request.get("auxiliary_split_manifest")),
        ("--output", request.get("profile_output_path")),
        ("--profile-updates", str(request.get("requested_updates"))),
    ):
        try:
            actual = command[command.index(flag) + 1]
        except (ValueError, IndexError):
            return False
        if expected is None or (str(Path(actual).resolve()) != str(Path(str(expected)).resolve()) if flag != "--profile-updates" else actual != expected):
            return False
    return "--resume" not in command and "--profile-receipt" not in command


def _validate_rank_logs(
    references: object, requested_updates: object
) -> tuple[list[tuple[str, bool, object]], list[dict[str, object]]]:
    checks: list[tuple[str, bool, object]] = []
    verified: list[dict[str, object]] = []
    if not isinstance(references, list) or len(references) != 1:
        return [("rank_log_count", False, references)], verified
    reference = _mapping(references[0])
    path_text, digest = reference.get("path"), reference.get("sha256")
    if not isinstance(path_text, str) or not isinstance(digest, str):
        return [("rank_log_reference", False, reference)], verified
    path = Path(path_text)
    valid_file = path.is_file() and not path.is_symlink()
    checks.append(("rank_log_file_hash", valid_file and sha256_file(path) == digest, digest))
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError):
        rows = []
    finite_keys = ("total_loss", "label_loss", "gradient_norm_preclip", "learning_rate")
    row_ok = (
        _positive_int(requested_updates)
        and len(rows) == requested_updates
        and all(
            isinstance(row, dict)
            and row.get("rank") == 0
            and row.get("step") == index
            and all(_finite(row.get(key)) and float(row[key]) >= 0 for key in finite_keys)
            and _positive_int(row.get("valid_rows"))
            and _positive_int(row.get("real_rows"))
            and _positive_int(row.get("forwarded_rows"))
            and row.get("forwarded_rows") >= row.get("real_rows")
            and _finite(row.get("label_weight"))
            and float(row.get("label_weight")) > 0
            for index, row in enumerate(rows, 1)
        )
    )
    checks.append(("rank_log_finite_updates", row_ok, {"rows": len(rows), "requested": requested_updates}))
    if valid_file:
        verified.append({"rank": reference.get("rank"), "path": str(path.resolve()), "sha256": digest, "rows": len(rows)})
    return checks, verified


def main() -> None:
    """Review one exact profile and write a separate immutable decision receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--trial-request", type=Path, required=True)
    parser.add_argument("--attempt-receipt", type=Path, required=True)
    parser.add_argument("--worker-receipt", type=Path, required=True)
    parser.add_argument("--deadline-receipt", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.policy.name != POLICY_NAME:
        raise SystemExit(f"policy filename must be {POLICY_NAME}")
    result = review_profile(
        profile_path=args.profile,
        request_path=args.trial_request,
        attempt_path=args.attempt_receipt,
        worker_path=args.worker_receipt,
        deadline_path=args.deadline_receipt,
        policy_path=args.policy,
        contract_path=args.contract,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "failed_checks": result["failed_checks"]}))


if __name__ == "__main__":
    main()
