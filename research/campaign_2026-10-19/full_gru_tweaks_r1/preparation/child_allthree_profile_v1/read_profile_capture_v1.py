#!/usr/bin/env python3
"""Validate a saved one-shot all-three profile capture without opening payloads."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import cast

UNIT = "computer-vision-homework-fg19-child-allthree-profile-v1.service"
INVOCATION = "1d62c3fb47ac4bd5be8e8ec8fe6759cd"
BODY_SHA256 = "e03e3b4640531d8c6d3a532efbb672f1d21433d3564e590b79e6188812d414ad"
CONTRACT_SHA256 = "ab68d868261106a7548b649243f4a645ece926b6138d683c731841cfb9f0b7c5"
PROBE_SHA256 = "446e679ddae0f05225209950d928790264c8cd57fb8836b79f671d06d03be9e4"
SUPPORT_SHA256 = "958d973b17a2180c954ca7e0953ce20d7b5a3120124cd3cf964a97606604f409"
PARENT_CHECKPOINT = "6027fc1db38fbc82d7cf1bb8c48dcb89f75bcc57ff91e9a7127f803cea11ef45"
PARENT_ACCEPTANCE = "5b9ddc32f871f4dae29334b4f573d1e38efd9f8435cf2d7fde47e47f97e658b8"
PARENT_STOP = "15a1e42a90a0962c9b45b6043a7e1e152057848725391fffcd611862e5c6ffa8"
DICTIONARY_CACHE = "9cf76503b579d2a88e7342b5f727e1ad5b3945c0938f1ae62d9738442a68b659"
DICTIONARY_ACCEPTANCE = "4802815134d09bbbfeb697689a17ac82bcbbd6541e7969d5fec56867ecd3066f"
SPLIT = "c09f7240396a33ca82977115af8d6067e571c0ec25c58a3c99b438ca2d2294b7"
SOURCE_COMMIT = "68ff682977b822f38351aac0f621146f5f13b440"
CACHE_METADATA = "03d9cf1fbd20b8bb3ad7b7d0c4513293fd1d5128e816256e4bf3cdfbfa4b43e3"
CACHE_ACCEPTANCE = "88b75986538ecf4c7255705c173d5ac16946dd5ca44e9c4c69c331a381b5bec7"


def _decode_receipt(receipts: dict[str, object], name: str) -> bytes:
    """Decode and verify a receipt copied in the capture object."""
    item = cast(dict[str, object], receipts[name])
    raw = base64.b64decode(cast(str, item["base64"]), validate=True)
    if (
        type(item.get("bytes")) is not int
        or len(raw) != item["bytes"]
        or hashlib.sha256(raw).hexdigest() != item["sha256"]
    ):
        raise ValueError(f"captured {name} bytes differ from the recorded size or digest")
    if len(raw) > 1_000_000:
        raise ValueError(f"captured {name} exceeds the bounded receipt size")
    return raw


def _markers(log_raw: bytes) -> dict[str, list[object]]:
    """Parse only known profile log markers from the preserved unit.log bytes."""
    parsed: dict[str, list[object]] = {
        "CHILD_PROFILE_INITIAL_STATE": [],
        "PROFILE": [],
        "PROFILE_UPDATE_EVIDENCE": [],
    }
    for line in log_raw.decode("utf-8").splitlines():
        for marker in parsed:
            prefix = f"{marker} "
            if line.startswith(prefix):
                parsed[marker].append(json.loads(line[len(prefix) :]))
                break
    return parsed


def _exact_profile_state(record: dict[str, object]) -> bool:
    """Check the initial parent extension proof and six-batch profile identity."""
    profile = cast(dict[str, object], record.get("profile", {}))
    initialization = cast(dict[str, object], record.get("initialization", {}))
    initialization_expected = {
        "parent_input_width": 211,
        "child_input_width": 467,
        "parent_output_width": 23,
        "child_output_width": 25,
        "parent_trunk_and_prefix_equal": True,
        "new_input_columns_zero": True,
        "new_logvariance_rows_zero": True,
        "normalization_prefix_equal": True,
        "normalization_suffix_mean_zero_std_one": True,
        "finite_initial_state": True,
    }
    fingerprints = (
        initialization.get("parent_state_fingerprint"),
        initialization.get("initial_child_state_fingerprint"),
    )
    typed_initialization = all(
        (
            type(initialization.get(key)) is bool
            if type(value) is bool
            else type(initialization.get(key)) is type(value)
        )
        and initialization.get(key) == value
        for key, value in initialization_expected.items()
    )
    expected_profile = {
        "sequence_length": 20_000,
        "batch": 16,
        "chunk": 2_000,
        "warmup_batches": 2,
        "measured_batches": 4,
        "batch_count": 6,
        "updates_per_batch": 10,
        "optimizer_attempts": 60,
        "input_representation": "ema_sparse",
        "objective": "nll",
    }
    typed_profile = all(
        type(profile.get(key)) is type(value) and profile.get(key) == value
        for key, value in expected_profile.items()
    )
    return (
        record.get("schema") == "fg19-child-allthree-profile-initial-state.v1"
        and record.get("parent_run") == "fg19_parent_h256x2_f0_s1001_fit_r2_v1"
        and record.get("parent_invocation_id") == "5186b3016ff54b1fa754c10a1f87b3e4"
        and record.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT
        and record.get("parent_acceptance_sha256") == PARENT_ACCEPTANCE
        and record.get("parent_exact_stop_sha256") == PARENT_STOP
        and record.get("dictionary_cache_sha256") == DICTIONARY_CACHE
        and record.get("split_manifest_sha256") == SPLIT
        and record.get("source_commit") == SOURCE_COMMIT
        and record.get("parent_acceptance_status") == "ACCEPTED"
        and typed_initialization
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in fingerprints
        )
        and typed_profile
    )


def _valid_update_evidence(evidence: dict[str, object]) -> bool:
    """Check the canonical JSON-safe finite 60-attempt profile counters."""
    integer_values = {
        "optimizer_attempts": 60,
        "optimizer_completed_updates": 60,
        "amp_skipped_updates": 0,
        "loss_checks": 60,
        "gradient_stats_steps": 60,
        "gradient_stats_nonfinite_steps": 0,
    }
    return (
        all(
            type(evidence.get(name)) is int and evidence.get(name) == value
            for name, value in integer_values.items()
        )
        and evidence.get("grad_scaler_enabled") is False
        and evidence.get("loss_finite") is True
        and evidence.get("gradient_finite") is True
        and evidence.get("counts_consistent") is True
        and evidence.get("valid_for_throughput") is True
        and evidence.get("gradient_check_basis") == "complete pre-clip GradStats gradient norms"
    )


def read_capture(path: Path) -> dict[str, object]:
    """Validate bounded capture bytes and report whether root may review acceptance."""
    if path.stat().st_size > 6_000_000:
        raise ValueError("capture JSON exceeds the 6 MB base64 transport bound")
    capture = cast(dict[str, object], json.loads(path.read_bytes()))
    if capture.get("schema") != "fg19-child-allthree-profile-capture.v1":
        raise ValueError("capture schema differs from this exact profile observer")
    if capture.get("expected_unit") != UNIT or capture.get("expected_invocation_id") != INVOCATION:
        raise ValueError("capture unit or invocation differs from the admitted profile")
    unit = cast(dict[str, str], capture.get("unit_properties", {}))
    terminal = (
        unit.get("InvocationID") == INVOCATION
        and unit.get("ActiveState") == "active"
        and unit.get("SubState") == "exited"
        and unit.get("MainPID") == "0"
        and unit.get("ExecMainCode") == "1"
        and unit.get("ExecMainStatus") == "0"
        and unit.get("Result") == "success"
    )
    receipts = cast(dict[str, object], capture.get("receipts", {}))
    if set(receipts) != {"ADMISSION.json", "LAUNCH.json", "unit.log"}:
        raise ValueError("capture receipts do not match the exact three-file read set")
    receipt_sizes = [cast(dict[str, object], item).get("bytes") for item in receipts.values()]
    if any(type(value) is not int or value < 0 for value in receipt_sizes):
        raise ValueError("capture receipt sizes must be nonnegative integer byte counts")
    if sum(cast(list[int], receipt_sizes)) > 4_000_000:
        raise ValueError("capture receipt aggregate exceeds the 4 MB read bound")
    admission = cast(dict[str, object], json.loads(_decode_receipt(receipts, "ADMISSION.json")))
    launch = cast(dict[str, object], json.loads(_decode_receipt(receipts, "LAUNCH.json")))
    log_raw = _decode_receipt(receipts, "unit.log")
    if (
        admission.get("unit") != UNIT
        or admission.get("run_name") != "fg19_child_allthree_f0_s1001_profile_v1"
    ):
        raise ValueError("captured admission does not bind this exact unit and run")
    if (
        admission.get("source_commit") != SOURCE_COMMIT
        or admission.get("source_files_verified") != 167
        or admission.get("source_manifest_sha256")
        != "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621"
        or admission.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT
        or admission.get("parent_acceptance_sha256") != PARENT_ACCEPTANCE
        or admission.get("parent_exact_stop_sha256") != PARENT_STOP
        or admission.get("split_sha256") != SPLIT
    ):
        raise ValueError("captured admission has a different source identity")
    if (
        launch.get("unit") != UNIT
        or launch.get("schema") != "fg19-child-allthree-profile-launch.v1"
    ):
        raise ValueError("captured launch receipt has a different unit")
    staged = cast(dict[str, str], admission.get("private_stage_sha256", {}))
    expected_staged = {
        "DICTIONARY_ACCEPTANCE.json": DICTIONARY_ACCEPTANCE,
        "PARENT_EXACT_STOP.json": PARENT_STOP,
        "PARENT_TERMINAL_ACCEPTANCE.json": PARENT_ACCEPTANCE,
        "child_allthree_profile_contract_v1.py": CONTRACT_SHA256,
        "child_allthree_profile_probe_v1.py": PROBE_SHA256,
        "propensity_terminal_support.py": SUPPORT_SHA256,
    }
    if staged != expected_staged:
        raise ValueError("captured private-stage pins differ from the reviewed profile packet")
    code_pins = cast(dict[str, dict[str, object]], capture.get("staged_code_pins", {}))
    if set(code_pins) != {
        "child_allthree_profile_contract_v1.py",
        "child_allthree_profile_probe_v1.py",
        "propensity_terminal_support.py",
    } or any(item.get("matches_expected") is not True for item in code_pins.values()):
        raise ValueError("fresh private-D Python source hashes differ from executed pins")
    cache = cast(dict[str, object], admission.get("cache", {}))
    dictionary = cast(dict[str, object], admission.get("dictionary", {}))
    if (
        cache.get("metadata_sha256") != CACHE_METADATA
        or cache.get("acceptance_sha256") != CACHE_ACCEPTANCE
        or dictionary.get("sha256") != DICTIONARY_CACHE
    ):
        raise ValueError("captured admission has different dictionary or teacher-cache identities")
    submission = cast(dict[str, object], launch.get("submission", {}))
    submission_stderr = base64.b64decode(cast(str, submission.get("stderr_b64", "")), validate=True)
    command = cast(list[str], launch.get("command", []))
    if (
        f"invocation ID: {INVOCATION}".encode() not in submission_stderr
        or f"--unit={UNIT}" not in command
        or "child_allthree_profile_probe_v1.py" not in " ".join(command)
    ):
        raise ValueError("captured launch command does not bind the exact invocation and probe")
    markers = _markers(log_raw)
    initial_records = markers["CHILD_PROFILE_INITIAL_STATE"]
    timing_records = markers["PROFILE"]
    evidence_records = markers["PROFILE_UPDATE_EVIDENCE"]
    state_valid = len(initial_records) == 1 and _exact_profile_state(
        cast(dict[str, object], initial_records[0])
    )
    evidence_valid = len(evidence_records) == 1 and _valid_update_evidence(
        cast(dict[str, object], evidence_records[0])
    )
    run_inventory = cast(dict[str, object], capture.get("run_directory_stat_only", {}))
    entries = cast(list[dict[str, object]], run_inventory.get("entries", []))
    run_inventory_complete = all(
        item.get("is_symlink") is False
        and type(item.get("bytes")) is int
        and type(item.get("device")) is int
        and type(item.get("inode")) is int
        and type(item.get("mtime_ns")) is int
        for item in entries
    )
    checkpoint_names = sorted(
        cast(str, item.get("name"))
        for item in entries
        if item.get("name") in {"model.pt", "model_ep5.pt", "training_state.pt"}
    )
    query = cast(dict[str, object], capture.get("unit_query", {}))
    query_valid = (
        type(query.get("returncode")) is int
        and query.get("returncode") == 0
        and query.get("timed_out") is False
    )
    run_directory_valid = (
        run_inventory.get("exists") is True
        and run_inventory.get("symlink") is False
        and run_inventory.get("not_directory") is not True
    )
    profile_timing_seen = len(timing_records) == 1
    return {
        "schema": "fg19-child-allthree-profile-capture-reader.v1",
        "capture_path": str(path),
        "capture_consistent": query_valid
        and state_valid
        and evidence_valid
        and profile_timing_seen,
        "terminal_success": terminal,
        "initial_state_proof_valid": state_valid,
        "profile_timing_records": len(timing_records),
        "update_evidence_valid": evidence_valid,
        "checkpoint_names_from_stat_only_inventory": checkpoint_names,
        "checkpoint_absence": not checkpoint_names,
        "run_directory_valid": run_directory_valid,
        "run_inventory_complete": run_inventory_complete,
        "gpu_observation": capture.get("gpu"),
        "parent_populated_and_memory_observation": capture.get("parent_state"),
        "unit_cgroup_observation": capture.get("own_cgroup_state"),
        "acceptance_candidate_for_root_review": (
            terminal
            and state_valid
            and evidence_valid
            and profile_timing_seen
            and run_directory_valid
            and run_inventory_complete
            and not checkpoint_names
        ),
        "scope": "Read-only receipt and log check; no model, cache, dictionary, feature or data payload opened.",
    }


def main() -> None:
    """Read one already-saved capture and print its bounded validation summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    print(json.dumps(read_capture(args.capture), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
