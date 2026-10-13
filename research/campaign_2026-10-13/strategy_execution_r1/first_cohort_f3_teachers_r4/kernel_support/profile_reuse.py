"""Verify and bind the completed R2 device profile for the R4 fit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PACKET = Path("research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r4")
EVIDENCE = PACKET / "profile_reuse_r2"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_output_hashes(receipt: dict) -> dict[str, tuple[str, int]]:
    rows = receipt.get("inventory")
    if not isinstance(rows, list):
        raise ValueError("R2 terminal output inventory is missing")
    expected: dict[str, tuple[str, int]] = {}
    for row in rows:
        path = row.get("path")
        if isinstance(path, str) and "results/profile/" in path:
            expected[path] = (row["sha256"], int(row["size_bytes"]))
    if len(expected) < 8:
        raise ValueError("R2 terminal inventory lacks profile outputs")
    return expected


def _verify_archived_outputs(root: Path, receipt: dict) -> None:
    expected = _expected_output_hashes(receipt)
    required = {
        "profile_guard_receipt_r2.json": "results/profile/profile_guard_receipt.json",
        "profile_input_identity_r2.json": ".input_identity.json",
        "profile_inventory_r2.json": "/inventory.json",
        "profile_result_r2.json": "/result.json",
        "profile_spec_r2.json": "/spec.json",
        "profile_summary_r2.json": "/summary.json",
        "profile_timeout_resolution_r2.json": "/timeout_resolution.json",
        "profile_worker_r2.log": "/worker.log",
    }
    for local_name, suffix in required.items():
        matches = [
            (path, values)
            for path, values in expected.items()
            if path.endswith(suffix)
            and (suffix != "/spec.json" or ("/profile/" in path and "/runtime_specs/" not in path))
        ]
        if len(matches) != 1:
            raise ValueError(f"R2 terminal inventory has ambiguous {local_name}")
        path, (expected_hash, expected_size) = matches[0]
        local = root / EVIDENCE / local_name
        if local.stat().st_size != expected_size or _sha256(local) != expected_hash:
            raise ValueError(
                f"R2 downloaded profile output differs from terminal inventory: {path}"
            )


def _verify_recipe_identity(root: Path, profile_spec: dict, reference: dict) -> None:
    def normalized(value: object) -> object:
        if isinstance(value, str):
            return value.replace("_r2", "_r1")
        if isinstance(value, list):
            return [normalized(item) for item in value]
        if isinstance(value, dict):
            return {key: normalized(item) for key, item in value.items()}
        return value

    for field in (
        "argv",
        "artifacts",
        "environment",
        "inputs",
        "kind",
        "limits",
        "name",
        "version",
    ):
        if normalized(profile_spec.get(field)) != reference.get(field):
            raise ValueError(f"R2 profile does not match the reviewed R1 recipe field {field}")

    manifest_path = root / (
        "research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1/"
        "source_snapshot/SOURCE_MANIFEST_H390.json"
    )
    manifest = _read_json(manifest_path)
    source_entries = {}
    source_prefix = (
        "research/campaign_2026-10-13/strategy_execution_r1/first_cohort_f3_teachers_r1/"
        "source_snapshot/"
    )
    for entry in profile_spec.get("code", []):
        path = entry.get("path", "")
        if path.startswith(source_prefix):
            source_entries[path.removeprefix(source_prefix)] = entry.get("sha256")
    expected_sources = dict(manifest["sources"])
    expected_sources["SOURCE_MANIFEST_H390.json"] = _sha256(manifest_path)
    if any(expected_sources.get(path) != sha256 for path, sha256 in source_entries.items()):
        raise ValueError("R2 profile contains a source hash outside the H390 source manifest")
    reference_sources = {
        entry["path"].removeprefix(source_prefix): entry["sha256"]
        for entry in reference.get("code", [])
        if entry.get("path", "").startswith(source_prefix)
    }
    if source_entries != reference_sources:
        raise ValueError(
            "R2 profile source paths/hashes differ from the reviewed R1 profile recipe"
        )


def _verify_profile_split_log(root: Path) -> dict:
    """Bind the profile's completed data scan and actual F3 split load."""
    log_path = root / EVIDENCE / "profile_worker_r2.log"
    events = []
    for line_number, line in enumerate(log_path.read_text().splitlines(), start=1):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and isinstance(item.get("event"), str):
            events.append((line_number, item))

    def exactly_one(event_name: str) -> dict:
        matches = [item for _, item in events if item.get("event") == event_name]
        if len(matches) != 1:
            raise ValueError(f"R2 profile log must contain exactly one {event_name} event")
        return matches[0]

    split_begin = exactly_one("campaign_split_load_begin")
    if Path(split_begin.get("split_manifest", "")).name != "fold3.json":
        raise ValueError("R2 profile loaded a split other than the frozen F3 fold")
    split_complete = exactly_one("campaign_split_load_complete")
    group_counts = {
        "fit_train_groups": 10539,
        "fit_valid_groups": 1368,
        "evaluation_groups": 505,
    }
    if any(split_complete.get(name) != count for name, count in group_counts.items()):
        raise ValueError("R2 profile loaded unexpected F3 campaign split membership")

    identity = _read_json(root / EVIDENCE / "profile_input_identity_r2.json")
    expected_inputs = {}
    for row in identity.get("observations", []):
        path = row.get("path")
        expected = row.get("expected")
        if isinstance(path, str) and isinstance(expected, dict):
            expected_inputs[Path(path).name] = expected

    hash_evidence = {}
    for filename in ("train.parquet", "valid.parquet"):
        expected = expected_inputs.get(filename)
        if not isinstance(expected, dict):
            raise ValueError(f"R2 input identity lacks the expected {filename} hash")
        matches = [
            item
            for _, item in events
            if item.get("event") == "portable_sha256_complete" and item.get("file") == filename
        ]
        if len(matches) != 1:
            raise ValueError(f"R2 profile log lacks a unique completed hash for {filename}")
        item = matches[0]
        if (
            item.get("bytes_read") != expected.get("size")
            or item.get("bytes_total") != expected.get("size")
            or item.get("sha256") != expected.get("sha256")
            or not isinstance(item.get("elapsed_seconds"), (int, float))
            or item["elapsed_seconds"] <= 0
        ):
            raise ValueError(f"R2 profile hash does not match the frozen {filename} input")
        hash_evidence[filename] = {
            "bytes": item["bytes_total"],
            "sha256": item["sha256"],
            "elapsed_seconds": item["elapsed_seconds"],
            "fully_read": True,
        }

    return {
        "split_manifest": "fold3.json",
        "group_counts": group_counts,
        "split_load_elapsed_seconds": split_complete.get("elapsed_seconds"),
        "portable_hashes": hash_evidence,
        "portable_hash_seconds_total": sum(
            row["elapsed_seconds"] for row in hash_evidence.values()
        ),
    }


def validate_profile_reuse(root: Path) -> dict:
    """Fail closed unless R2's successful profile is exact and terminally bound."""
    receipt = _read_json(root / EVIDENCE / "terminal_receipt_r2.json")
    if (
        receipt.get("provider_status") != "ERROR"
        or receipt.get("pipeline_status") != "blocked_or_failed"
    ):
        raise ValueError("R2 must remain a terminally collected operational failure")
    if receipt.get("first_failure", {}).get("fit_worker_started") is not False:
        raise ValueError(
            "R2 fit worker unexpectedly started; profile reuse path must be reassessed"
        )
    _verify_archived_outputs(root, receipt)

    profile = _read_json(root / EVIDENCE / "profile_result_r2.json")
    guard = _read_json(root / EVIDENCE / "profile_guard_receipt_r2.json")
    summary = _read_json(root / EVIDENCE / "profile_summary_r2.json")
    runtime_spec = _read_json(root / EVIDENCE / "profile_spec_r2.json")
    reference = _read_json(root / EVIDENCE / "profile_reference_spec_r1.json")
    if profile.get("status") != "complete" or profile.get("returncode") != 0:
        raise ValueError("R2 profile worker did not complete successfully")
    if guard.get("status") != "passed" or guard.get("profile_batches") != 10:
        raise ValueError("R2 profile guard did not pass the 5+5 profile")
    finished = summary.get("finished", [])
    if summary.get("incident") is not None or len(finished) != 1:
        raise ValueError("R2 profile runner did not finish exactly one worker without incident")
    if finished[0].get("status") != "complete" or finished[0].get("returncode") != 0:
        raise ValueError("R2 profile runner summary is not successful")
    _verify_recipe_identity(root, runtime_spec, reference)
    if profile.get("device") != guard.get("device"):
        raise ValueError("R2 profile result and guard report different devices")
    if "T4" not in profile["device"].get("name", ""):
        raise ValueError("R2 profile was not measured on a T4")
    for field in ("peak_gpu_bytes", "peak_ram_bytes", "wall_seconds"):
        if profile.get(field) != finished[0].get(field):
            raise ValueError(f"R2 profile result differs from runner summary: {field}")
    if profile["peak_ram_bytes"] != guard.get("peak_worker_ram_bytes"):
        raise ValueError("R2 profile RAM measurement differs from its guard receipt")

    split_log_evidence = _verify_profile_split_log(root)

    result = {
        "schema": "wnn.strategy.f3-teacher-profile-reuse.r4",
        "status": "passed",
        "source_kernel_id": receipt.get("kernel_id"),
        "source_kernel_version": receipt.get("kernel_version"),
        "source_terminal_status": receipt.get("provider_status"),
        "profile_spec_r2_sha256": _sha256(root / EVIDENCE / "profile_spec_r2.json"),
        "profile_reference_r1_sha256": _sha256(root / EVIDENCE / "profile_reference_spec_r1.json"),
        "profile_worker_log_sha256": _sha256(root / EVIDENCE / "profile_worker_r2.log"),
        "profile_recipe_identity": "R2 argv, inputs, environment, limits, and every H390 source snapshot hash match the R1 resource-profile recipe after only the run-name suffix normalization.",
        "profile_device": profile["device"],
        "peak_gpu_bytes": profile["peak_gpu_bytes"],
        "peak_worker_ram_bytes": profile["peak_ram_bytes"],
        "profile_batches": guard["profile_batches"],
        "optimizer_completed_updates": guard["optimizer_completed_updates"],
        "split_log_evidence": split_log_evidence,
        "profile_host_inventory": _read_json(root / EVIDENCE / "profile_inventory_r2.json").get(
            "host"
        ),
        "scientific_result": None,
    }
    binding_path = root / PACKET / "PROFILE_REUSE_BINDING_R4.json"
    if binding_path.exists():
        binding = _read_json(binding_path)
        expected_evidence = binding.get("evidence")
        if not isinstance(expected_evidence, list):
            raise ValueError("R4 profile evidence binding is malformed")
        for row in expected_evidence:
            local = root / row["path"]
            if local.stat().st_size != row["size"] or _sha256(local) != row["sha256"]:
                raise ValueError(f"R4 frozen profile evidence changed: {row['path']}")
        if binding.get("validation") != result:
            raise ValueError("R4 profile evidence no longer matches its preparation validation")
    return result
