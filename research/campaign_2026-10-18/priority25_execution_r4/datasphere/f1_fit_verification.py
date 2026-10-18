#!/usr/bin/env python3
"""Reconcile one terminal provider output with the F1 bundle manifest."""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path
from typing import Any

from controller import PROJECT_ID, TERMINAL_STATUSES, FailClosed, sha256_file
from f1_fit_admission import (
    F1_JOB_NAME,
    F1_OUTPUT_MAX_BYTES,
    F1_OUTPUT_PATH,
    F1_OUTPUT_VARIABLE,
    F1Admission,
)
from f1_fit_recipe import ARMS, FOLD1_SHA256, SOURCE_COMMIT

# Bundle manifests and fit receipts contain heterogeneous producer-generated JSON.
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class ProviderOutputPending(FailClosed):
    """The terminal job identity is sound but its expected output is not listed yet."""


def validate_provider_file(
    state: dict[str, object], provider: dict[str, object], admission: F1Admission
) -> dict[str, object]:
    """Validate exact terminal job identity and its one bounded bundle file."""
    if (
        provider.get("id") != state.get("job_id")
        or provider.get("project_id") != PROJECT_ID
        or provider.get("name") != F1_JOB_NAME
        or provider.get("status") != state.get("provider_status")
        or provider.get("status") not in TERMINAL_STATUSES
    ):
        raise FailClosed("SDK provider manifest does not match terminal F1 job identity")
    files = provider.get("output_files")
    if not isinstance(files, list):
        raise FailClosed("provider output manifest is malformed")
    errors = provider.get("output_file_errors")
    if not isinstance(errors, list):
        raise FailClosed("provider manifest lacks a valid output error list")
    if any(
        isinstance(error, dict)
        and error.get("path") == F1_OUTPUT_PATH
        and error.get("variable") == F1_OUTPUT_VARIABLE
        for error in errors
    ):
        raise FailClosed("provider reported an upload error for the F1 bundle")
    if not files:
        raise ProviderOutputPending("terminal provider manifest has not listed the F1 bundle yet")
    if len(files) != 1 or not isinstance(files[0], dict):
        raise FailClosed("provider output manifest must contain exactly the F1 bundle")
    entry = files[0]
    digest = entry.get("sha256")
    size = entry.get("size_bytes")
    if (
        entry.get("path") != F1_OUTPUT_PATH
        or entry.get("variable") != F1_OUTPUT_VARIABLE
        or type(size) is not int
        or size <= 0
        or size > min(admission.output_max_bytes, F1_OUTPUT_MAX_BYTES)
        or not isinstance(digest, str)
        or not SHA256_PATTERN.fullmatch(digest)
    ):
        raise FailClosed("provider manifest has an invalid, oversized, or unexpected F1 output")
    return entry


def verify_bundle_file(
    path: Path, provider_file: dict[str, object], admission: F1Admission
) -> tuple[bool, dict[str, Any]]:
    """Compare downloaded bytes with provider metadata, then verify every bundle file."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("download command returned without the expected regular bundle")
    if path.stat().st_size != provider_file["size_bytes"]:
        raise ValueError("downloaded bundle size differs from provider output manifest")
    if sha256_file(path) != provider_file["sha256"]:
        raise ValueError("downloaded bundle SHA-256 differs from provider output manifest")
    from f1_fit_bundle import MAX_PROVIDER_OUTPUT_BYTES, verify_bundle_archive

    if path.stat().st_size > MAX_PROVIDER_OUTPUT_BYTES:
        raise ValueError("downloaded bundle exceeds the producer verifier's size limit")
    manifest = verify_bundle_archive(path)
    complete = verify_fit_receipt(path, manifest, admission)
    return complete, manifest


def verify_fit_receipt(bundle_path: Path, manifest: dict[str, Any], admission: F1Admission) -> bool:
    """Return whether a hash-verified bundle contains the complete paired endpoint."""
    file_records = {
        item["path"]: item
        for item in manifest["files"]
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    if len(file_records) != len(manifest["files"]):
        raise FailClosed("verified bundle manifest contains malformed file records")
    receipt_name = "fit-pair-receipt.json"
    if receipt_name not in file_records:
        raise FailClosed("verified bundle has no fit-pair receipt")
    with tarfile.open(bundle_path, "r:gz") as archive:
        member = archive.extractfile(f"fit-results/{receipt_name}")
        if member is None:
            raise FailClosed("fit-pair receipt could not be read from the verified bundle")
        receipt = json.load(member)
    if not isinstance(receipt, dict):
        raise FailClosed("fit-pair receipt is not a JSON object")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict) or (
        inputs.get("profile_spec_sha256") != admission.profile_spec_sha256
        or inputs.get("fit_spec_sha256") != admission.fit_spec_sha256
        or inputs.get("code_archive_sha256") != admission.code_archive_sha256
        or inputs.get("supplemental_archive_sha256") != admission.supplemental_archive_sha256
        or inputs.get("source_commit") != SOURCE_COMMIT
        or inputs.get("fold_manifest_sha256") != FOLD1_SHA256
        or inputs.get("internal_deadline_seconds") != admission.internal_deadline_seconds
        or inputs.get("collection_reserve_seconds") != admission.collection_reserve_seconds
        or inputs.get("external_cancel_deadline_seconds_from_submit")
        != admission.external_deadline_seconds
    ):
        raise FailClosed("fit-pair receipt does not bind to the admitted F1 ancestry")
    if receipt.get("split_sha256") != FOLD1_SHA256:
        raise FailClosed("fit-pair receipt does not bind to the admitted fold-1 split")
    if (
        receipt.get("status") != "PAIR_FIT_AND_EXPORT_COMPLETE"
        or receipt.get("quality_result") is not None
    ):
        return False
    if receipt.get("collection_status") != "COMPLETE" or receipt.get("collection_errors") != []:
        return False
    arms = receipt.get("arms")
    if not isinstance(arms, dict) or set(arms) != {str(arm["name"]) for arm in ARMS}:
        return False
    required_per_arm = {
        "model_ep5.pt",
        "model.pt",
        "training_state.pt",
        "history.json",
        "provenance.json",
    }
    for arm in ARMS:
        name = str(arm["name"])
        record = arms.get(name)
        if not isinstance(record, dict) or record.get("status") != "COMPLETE":
            return False
        artifacts = record.get("fit_artifacts")
        if not isinstance(artifacts, dict) or not required_per_arm.issubset(artifacts):
            return False
        for artifact in required_per_arm:
            path = f"artifacts/runs/{name}/{artifact}"
            if not _receipt_file_matches(artifacts[artifact], file_records.get(path)):
                return False
        train_log = record.get("training_log")
        if train_log != f"{name}.train.log" or train_log not in file_records:
            return False
        prediction = record.get("prediction_export")
        if not isinstance(prediction, dict):
            return False
        if not _receipt_file_matches(
            prediction.get("prediction"), file_records.get(f"cache/pred_{name}_valid.npy")
        ):
            return False
        if not _receipt_file_matches(
            prediction.get("sidecar"), file_records.get(f"cache/pred_{name}_valid.json")
        ):
            return False
        prediction_log = prediction.get("log")
        if not isinstance(prediction_log, str) or prediction_log != f"{name}.predict.log":
            return False
        if prediction_log not in file_records:
            return False
    return True


def _receipt_file_matches(receipt_file: object, manifest_file: object) -> bool:
    """Bind one producer receipt file record to the independently checked bundle index."""
    if not isinstance(receipt_file, dict) or not isinstance(manifest_file, dict):
        return False
    size = receipt_file.get("bytes")
    digest = receipt_file.get("sha256")
    return (
        type(size) is int
        and size >= 0
        and size == manifest_file.get("bytes")
        and isinstance(digest, str)
        and SHA256_PATTERN.fullmatch(digest) is not None
        and digest == manifest_file.get("sha256")
    )
