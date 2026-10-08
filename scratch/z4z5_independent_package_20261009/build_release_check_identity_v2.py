#!/usr/bin/env python3
"""Bind an exploratory candidate ZIP to independent lineage and delivery gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--core-audit", type=Path, required=True)
    parser.add_argument("--package-audit", type=Path, required=True)
    parser.add_argument("--abi-report", type=Path, required=True)
    parser.add_argument("--incumbent-zip", type=Path, required=True)
    parser.add_argument("--incumbent-replay", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--public-incumbent-seconds", type=int, required=True)
    parser.add_argument("--public-runtime-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    submission_dir = args.submission_dir.resolve()
    archive = submission_dir / "submission.zip"
    core_audit_path = args.core_audit.resolve()
    package_audit_path = args.package_audit.resolve()
    abi_path = args.abi_report.resolve()
    incumbent_zip = args.incumbent_zip.resolve()
    incumbent_replay_path = args.incumbent_replay.resolve()
    calibration_path = args.calibration.resolve()
    public_runtime_evidence_path = args.public_runtime_evidence.resolve()
    package_audit = read_json(package_audit_path)
    core_audit = read_json(core_audit_path)
    abi = read_json(abi_path)
    replay = read_json(submission_dir / "TIMED_REPLAY.json")
    incumbent_replay = read_json(incumbent_replay_path)
    calibration = read_json(calibration_path)
    if args.public_incumbent_seconds <= 0 or not public_runtime_evidence_path.is_file():
        raise ValueError("public incumbent runtime needs a positive duration and evidence file")
    public_runtime_evidence = read_json(public_runtime_evidence_path)
    reference_report = read_json(submission_dir / "PY311_V3_REFERENCE.json")
    causal_reset_path = submission_dir / "CAUSAL_RESET_A_B_A.json"
    causal_reset = read_json(causal_reset_path)
    callback_paths = {
        level: submission_dir / f"PY311_{level}_CALLBACK.json" for level in ("V2", "AVX2", "V3")
    }
    callbacks = {level: read_json(path) for level, path in callback_paths.items()}

    if not archive.is_file() or not incumbent_zip.is_file():
        raise FileNotFoundError("candidate or incumbent ZIP is missing")
    archive_hash = digest(archive)
    incumbent_hash = digest(incumbent_zip)
    portal_rows = public_runtime_evidence.get("new_results", []) + public_runtime_evidence.get(
        "reconfirmed_results", []
    )
    matched_runtime_rows = [row for row in portal_rows if row.get("zip_sha256") == incumbent_hash]
    if len(matched_runtime_rows) != 1:
        raise ValueError("runtime evidence must have exactly one row for the matched incumbent ZIP")
    matched_runtime_row = matched_runtime_rows[0]
    if int(matched_runtime_row.get("runtime_seconds", 0)) != args.public_incumbent_seconds:
        raise ValueError(
            "public runtime duration does not match the exact incumbent ZIP evidence row"
        )
    if package_audit.get("status") != "passed" or package_audit.get("zip_sha256") != archive_hash:
        raise ValueError("independent tensor audit does not bind the actual candidate ZIP")
    if core_audit.get("status") != "passed":
        raise ValueError("independent core derivation audit did not pass")
    if abi.get("passed") is not True or abi.get("artifact_sha256") != archive_hash:
        raise ValueError("static ABI receipt does not pass or does not bind the candidate ZIP")
    if replay.get("zip") != str(archive) or int(replay.get("rows", 0)) != 400_000:
        raise ValueError("timed replay is not the candidate's full 20-sequence replay")
    if (
        incumbent_replay.get("zip") != str(incumbent_zip)
        or int(incumbent_replay.get("rows", 0)) != 400_000
    ):
        raise ValueError("incumbent replay is not matched on the 20-sequence fixture")
    if digest(incumbent_zip) != incumbent_hash:
        raise ValueError("incumbent ZIP changed during receipt construction")
    if (
        reference_report.get("status") != "passed"
        or reference_report.get("zip_sha256") != archive_hash
    ):
        raise ValueError("Python 3.11 reference receipt does not bind the candidate ZIP")
    for level, report in callbacks.items():
        if (
            report.get("passed") is not True
            or report.get("zip_sha256") != archive_hash
            or report.get("rows") != 40_000
            or report.get("reset_prefix_rows") != 120
            or report.get("max_abs_vs_archived_native") != 0.0
        ):
            raise ValueError(f"Python 3.11 callback/reset gate failed for {level}")
    if (
        causal_reset.get("reset_bitwise") is not True
        or causal_reset.get("past_only_bitwise") is not True
        or causal_reset.get("dataset_sha256") != digest(root / "cache/valid_subset20.parquet")
    ):
        raise ValueError("full-stream A/B/A reset or past-only causal test failed")

    member = package_audit["member_declaration"][0]
    core_run_at, head_run_at = member["source"].split("|")
    core_run, core_epoch_text = core_run_at.rsplit("@ep", 1)
    head_run, head_epoch_text = head_run_at.rsplit("@ep", 1)
    core_epoch, head_epoch = int(core_epoch_text), int(head_epoch_text)
    core_checkpoint = root / "artifacts/runs" / core_run / f"model_ep{core_epoch}.pt"
    core_provenance_path = core_checkpoint.parent / "provenance.json"
    core_provenance = read_json(core_provenance_path)
    source_run = str(core_audit["source_run"])
    source_epoch = int(core_audit["epoch"])
    source_checkpoint = root / "artifacts/runs" / source_run / f"model_ep{source_epoch}.pt"
    source_provenance_path = source_checkpoint.parent / "provenance.json"
    source_provenance = read_json(source_provenance_path)
    head_checkpoint = root / "artifacts/runs" / head_run / f"model_ep{head_epoch}.pt"
    head_provenance_path = head_checkpoint.parent / "provenance.json"
    head_provenance = read_json(head_provenance_path)

    source_hash = digest(source_checkpoint)
    core_hash = digest(core_checkpoint)
    head_hash = digest(head_checkpoint)
    expected_core_map_hash = core_provenance.get("checkpoint_sha256", {}).get(core_checkpoint.name)
    if (
        core_audit.get("source_run") != source_run
        or core_audit.get("destination_run") != core_run
        or core_audit.get("epoch") != source_epoch
        or core_audit.get("source_checkpoint_sha256") != source_hash
        or core_audit.get("destination_checkpoint_sha256") != core_hash
        or core_audit.get("destination_provenance_sha256") != digest(core_provenance_path)
        or core_audit.get("source_provenance_sha256") != digest(source_provenance_path)
        or core_audit.get("source_head_rows") != 31
        or core_audit.get("core_head_rows") != 23
        or core_audit.get("non_family_tensors_bitwise_equal") is not True
        or core_audit.get("primary_output_rows_bitwise_equal") is not True
    ):
        raise ValueError("core derivation receipt does not bind source and 23-row core bytes")
    if source_provenance.get("checkpoint_sha256", {}).get(source_checkpoint.name) != source_hash:
        raise ValueError("31-row source provenance map does not bind its actual checkpoint")
    derivation_metadata = core_provenance.get("core_derivation", {})
    if (
        derivation_metadata.get("source_run") != source_run
        or derivation_metadata.get("source_epoch") != source_epoch
        or derivation_metadata.get("source_checkpoint_sha256") != source_hash
        or derivation_metadata.get("source_provenance_sha256") != digest(source_provenance_path)
    ):
        raise ValueError("core provenance derivation metadata disagrees with its audited source")
    if expected_core_map_hash == core_hash:
        core_map_binding = "actual_derived_core_checkpoint_sha256"
    elif expected_core_map_hash == source_hash:
        core_map_binding = "stale_source_checkpoint_hash_preserved_and_disclosed"
    else:
        raise ValueError("core provenance map matches neither actual derived core nor source bytes")
    if head_provenance.get("initial_checkpoint_sha256") != core_hash:
        raise ValueError("head fine-tune initial hash does not bind to the exact core checkpoint")
    if head_provenance.get("args", {}).get("init") != core_run_at:
        raise ValueError("head fine-tune args do not name the exact core initializer")
    if head_provenance.get("checkpoint_sha256", {}).get(head_checkpoint.name) != head_hash:
        raise ValueError("head fine-tune checkpoint map does not bind to its actual bytes")
    if package_audit.get("sources", [{}])[0].get("checkpoint_sha256") != core_hash:
        raise ValueError("package tensor audit core hash differs from the actual checkpoint")
    if package_audit.get("sources", [{}, {}])[1].get("checkpoint_sha256") != head_hash:
        raise ValueError("package tensor audit head hash differs from the actual checkpoint")

    with zipfile.ZipFile(archive) as packaged:
        if packaged.testzip() is not None:
            raise ValueError("candidate ZIP CRC check failed")
    candidate_us = float(replay["us_per_row"])
    incumbent_us = float(incumbent_replay["us_per_row"])
    ratio = candidate_us / incumbent_us
    public_incumbent_seconds = args.public_incumbent_seconds
    calibrated_minutes = public_incumbent_seconds * ratio / 60.0
    if calibration.get("projection_basis") is None:
        raise ValueError("baseline timing calibration lacks a documented projection basis")

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    report = {
        "schema": "wnn.exploratory-package-release-check.v1",
        "status": "technical_gates_passed_exploratory_not_quality_promoted",
        "package": {
            "directory": str(submission_dir),
            "zip": str(archive),
            "zip_sha256": archive_hash,
            "zip_bytes": archive.stat().st_size,
            "zip_limit_bytes": 20_000_000,
            "under_zip_limit": archive.stat().st_size <= 20_000_000,
        },
        "lineage": {
            "source_student_31_row_checkpoint": {
                "run_at": f"{source_run}@ep{source_epoch}",
                "checkpoint_sha256": source_hash,
                "provenance_sha256": digest(source_provenance_path),
            },
            "derived_core_23_row_checkpoint": {
                "run_at": core_run_at,
                "checkpoint_sha256": core_hash,
                "provenance_sha256": digest(core_provenance_path),
                "raw_checkpoint_map_sha256": expected_core_map_hash,
                "raw_checkpoint_map_binding": core_map_binding,
                "derivation_receipt": str(core_audit_path),
                "derivation_receipt_sha256": digest(core_audit_path),
                "retained_non_family_tensors_bitwise_equal": core_audit[
                    "non_family_tensors_bitwise_equal"
                ],
                "retained_primary_rows_bitwise_equal": core_audit[
                    "primary_output_rows_bitwise_equal"
                ],
                "row_counts": {
                    "source": core_audit["source_head_rows"],
                    "core": core_audit["core_head_rows"],
                    "dropped_family_rows": core_audit["dropped_family_rows"],
                },
            },
            "head_finetune_checkpoint": {
                "run_at": head_run_at,
                "checkpoint_sha256": head_hash,
                "provenance_sha256": digest(head_provenance_path),
                "initial_checkpoint_sha256": head_provenance["initial_checkpoint_sha256"],
                "initial_checkpoint_matches_core": head_provenance["initial_checkpoint_sha256"]
                == core_hash,
                "freeze": head_provenance.get("args", {}).get("freeze"),
                "seed": head_provenance.get("args", {}).get("seed"),
                "fit_groups": {
                    name: len(groups)
                    for name, groups in head_provenance.get("fit_row_groups", {}).items()
                },
                "full_valid_fit": bool(head_provenance.get("args", {}).get("full_valid")),
                "train_source": head_provenance.get("args", {}).get("train_source"),
                "include_heldout_valid": head_provenance.get("args", {}).get(
                    "include_heldout_valid"
                ),
                "eval_source": head_provenance.get("args", {}).get("eval_source"),
                "valid_fold": head_provenance.get("args", {}).get("valid_fold"),
                "split_manifest_sha256": head_provenance.get("split_manifest_sha256"),
            },
            "fixed_head_merge": {
                "source": member["source"],
                "outer_coefficient": member["coef"],
                "within_head_merge_coefficients": package_audit["source_head_merge_coefficients"],
                "zip_tensor_audit_sha256": digest(package_audit_path),
            },
        },
        "gates": {
            "tensor_and_provenance_chain": {
                "passed": True,
                "audit": str(package_audit_path),
                "audit_sha256": digest(package_audit_path),
                "core_provenance_checkpoint_map_binding": core_map_binding,
                "stale_core_map_is_explicit_warning": (
                    core_map_binding == "stale_source_checkpoint_hash_preserved_and_disclosed"
                ),
            },
            "static_abi": {
                "passed": True,
                "receipt": str(abi_path),
                "receipt_sha256": digest(abi_path),
            },
            "python311_callback_and_reset": {
                "passed": True,
                "python": reference_report.get("python"),
                "rows_per_callback_suite": 40_000,
                "reset_prefix_rows": 120,
                "max_abs_vs_v3_reference_by_level": {
                    level.lower(): callbacks[level]["max_abs_vs_archived_native"]
                    for level in callbacks
                },
                "reference_receipt_sha256": digest(submission_dir / "PY311_V3_REFERENCE.json"),
                "callback_receipt_sha256": {
                    level.lower(): digest(path) for level, path in callback_paths.items()
                },
            },
            "full_stream_causal_reset": {
                "passed": True,
                "reset_bitwise": causal_reset["reset_bitwise"],
                "past_only_bitwise": causal_reset["past_only_bitwise"],
                "data_sha256": causal_reset["dataset_sha256"],
                "receipt": str(causal_reset_path),
                "receipt_sha256": digest(causal_reset_path),
            },
            "timed_replay": {
                "passed_relative_to_matched_incumbent": calibrated_minutes < 60.0,
                "candidate_rows": replay["rows"],
                "candidate_us_per_row": candidate_us,
                "incumbent_rows": incumbent_replay["rows"],
                "incumbent_us_per_row": incumbent_us,
                "candidate_over_incumbent_ratio": ratio,
                "calibrated_public_runtime_minutes": calibrated_minutes,
                "public_incumbent_seconds_used": public_incumbent_seconds,
                "public_runtime_evidence": str(public_runtime_evidence_path),
                "public_runtime_evidence_sha256": digest(public_runtime_evidence_path),
                "public_runtime_evidence_match": {
                    "name": matched_runtime_row.get("name"),
                    "submission_id": matched_runtime_row.get("submission_id"),
                    "runtime_seconds": matched_runtime_row.get("runtime_seconds"),
                    "zip_sha256": matched_runtime_row.get("zip_sha256"),
                },
                "public_calibration_receipt": str(calibration_path),
                "public_calibration_receipt_sha256": digest(calibration_path),
                "incumbent_zip_sha256": incumbent_hash,
                "incumbent_local_replay_receipt": str(incumbent_replay_path),
                "incumbent_local_replay_receipt_sha256": digest(incumbent_replay_path),
                "candidate_local_replay_receipt_sha256": digest(
                    submission_dir / "TIMED_REPLAY.json"
                ),
                "scope": "qualified matched local estimate using the documented public incumbent calibration; no full public scorer run",
            },
        },
        "scientific_status": {
            "score_evaluated": False,
            "candidate_status": "exploratory_unpromoted",
            "heldout_claim": "none; source/core and head fits include validation groups",
            "public_score_claim": "none",
            "selection_exposure": "all-data validation fit; exposed groups cannot support a heldout claim",
        },
        "warnings": [],
    }
    if core_map_binding == "stale_source_checkpoint_hash_preserved_and_disclosed":
        report["warnings"].append(
            "The inherited core provenance checkpoint map contains the source checkpoint hash; "
            "this manifest binds the actual 23-row core hash through the independent derivation "
            "audit and the head-fit initializer hash."
        )
    else:
        report["warnings"].append(
            "The core identity-v2 provenance map binds the actual 23-row checkpoint; the "
            "independent derivation receipt preserves source ancestry and exact tensor parity."
        )
    report["warnings"].append(
        "The local matched timing ratio is calibrated to a historical public incumbent runtime; "
        "it is an estimate, not a public run of this candidate."
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "zip_sha256": archive_hash,
                "candidate_over_incumbent_ratio": ratio,
                "calibrated_public_runtime_minutes": calibrated_minutes,
            }
        )
    )


if __name__ == "__main__":
    main()
