"""Verify an exact final ZIP offline against its receipt and one full official sequence.

The check imports the delivered callback under Python 3.11, checks its packaged
native ABI and byte manifests, compares the W18-only callback with the receipt-
bound PyTorch W18 reference on one causal 20,000-row sequence, and measures a
single-core callback pass. It computes no task score and makes no independent
evaluation claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import utils as _callback  # noqa: E402
from scripts.package_final_tl3 import (  # noqa: E402
    BLEND,
    EXPECTED_CHECKPOINT_SHA256,
    FINAL_RECEIPT_SCHEMA,
    FINAL_RUNS,
    ROLES,
    RUNS,
    SOURCE_VALIDATION_SHA256,
    _validate_final_checkpoint,
    load_final_model_receipt,
)
from wnn.data import SEQUENCE_LENGTH, SequenceFile  # noqa: E402

FINAL_SCOPE = "approved_full_corpus_final_model_set"
MAX_ZIP_BYTES = 20_000_000
OFFICIAL_SEQUENCES = 1_970
OFFICIAL_SECONDS = 3_600.0
MEMORY_LIMIT_BYTES = 16 * (1 << 30)
QUANTIZATION_CONTRACT_SCHEMA = "wnn.w18-final-quantization-admission.v1"
NUMERIC_CEILINGS_SCHEMA = "wnn.w18-quantization-error-ceilings.v1"
FIDELITY_CONTRACT_SCHEMA = "wnn.w18-old-reference-quantization-fidelity-contract.v1"
EXPECTED_R11_NATIVE_V3_SHA256 = "a5ab8228c69173a47a4d5f9f25f1f94b2924cfe852b8f3166416d9561c947de9"
LEGACY_MALFORMED_NATIVE_SHA256_DECLARATION = (
    "a5ab8228c69173a47a4d5f9f25f1f94b2924cfe852b8f3166416d99561c947de9"
)
EXPECTED_QUANTIZATION_CONTRACT_SHA256 = (
    "ec1ce8811ceb43249aca3c1c79295c4124a6a80f6cc8320aeaadb71075ff9a3f"
)
EXPECTED_SERVING_HELPER_SHA256 = "5cc6b5852472584d010ed3f99dcadd61ecf12347ca2320c282caf62abfd8cba3"
EXPECTED_RMS_CEILINGS = (0.007595723327067938, 0.008979632848327796)
EXPECTED_P99_CEILINGS = (0.02050376683473587, 0.023459434509277344)
EXPECTED_MAX_CEILINGS = (0.04375985264778137, 0.05005139112472534)
EXPECTED_OLD_VALID_F32 = 0.6914165042737994
EXPECTED_OLD_VALID_Q8 = 0.6914021819338281
EXPECTED_OLD_VALID_DELTA = -1.4322339971339915e-5
EXPECTED_OLD_VALID_GROUPS = 1_873
EXPECTED_OLD_VALID_ROWS = 3_528_273
_SHA256 = set("0123456789abcdef")


@dataclass(frozen=True)
class DataPoint:
    """Minimal object implementing the delivered scorer callback contract."""

    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    state: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_pinned_json(
    path: Path, expected_sha256: str, label: str
) -> tuple[dict[str, object], Path]:
    """Read one regular JSON artifact and require its exact caller-pinned bytes."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError(f"{label} must be a bounded regular file")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"{label} bytes differ from their pinned SHA256")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, path.resolve()


def _read_contract_artifact(
    owner_path: Path, spec: object, label: str
) -> tuple[dict[str, object], Path, str]:
    """Resolve and read a path/SHA pair embedded in a frozen contract."""
    if not isinstance(spec, dict) or set(spec) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    relative_path, expected_sha256 = spec.get("path"), spec.get("sha256")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or not set(expected_sha256) <= _SHA256
    ):
        raise ValueError(f"{label} has a malformed path or SHA256")
    path = Path(relative_path)
    if not path.is_absolute():
        path = owner_path.parent / path
    value, resolved = _read_pinned_json(path, expected_sha256, label)
    return value, resolved, expected_sha256


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _is_positive_finite_epoch(value: object) -> bool:
    """Accept JSON integer/float epochs but reject booleans and invalid numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except OverflowError:
        return False
    return number > 0 and math.isfinite(number)


def _numeric_error_summary(
    predictions: np.ndarray,
    reference: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Summarize two-target FP32 parity without converting failures into exceptions."""
    if predictions.shape != reference.shape or predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError("numeric parity arrays must have matching (rows, 2) shapes")
    if not np.isfinite(predictions).all() or not np.isfinite(reference).all():
        raise ValueError("numeric parity arrays must be finite")
    difference = predictions.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    relative = absolute / np.maximum(np.abs(reference.astype(np.float64)), 1e-6)
    return {
        "rms": np.sqrt(np.mean(np.square(difference), axis=0)),
        "p99_abs": np.quantile(absolute, 0.99, axis=0, method="linear"),
        "max_abs_by_target": np.max(absolute, axis=0),
        "max_abs": float(absolute.max(initial=0.0)),
        "max_relative": float(relative.max(initial=0.0)),
        "allclose": bool(np.allclose(predictions, reference, atol=atol, rtol=rtol)),
    }


def _numeric_error_within_ceilings(summary: dict[str, object], ceilings: dict[str, object]) -> bool:
    """Apply frozen per-target RMS, p99 absolute, and maximum absolute bounds."""
    values = (
        (summary["rms"], ceilings["maximum_rms_by_target"]),
        (summary["p99_abs"], ceilings["maximum_p99_abs_by_target"]),
        (summary["max_abs_by_target"], ceilings["maximum_abs_by_target"]),
    )
    return all(
        isinstance(observed, np.ndarray)
        and isinstance(bound, list)
        and observed.shape == (2,)
        and np.all(observed <= np.asarray(bound, dtype=np.float64))
        for observed, bound in values
    )


def _validate_old_reference_report(
    report: dict[str, object],
    *,
    mode: str,
    precision: str,
    activation_mode: str,
    valid_sha256: str,
    native_sha256: str,
) -> tuple[float, str]:
    """Admit one same-source, same-mask historical W18-only fidelity report."""
    models = report.get("models")
    sources = report.get("sources")
    execution = report.get("execution")
    metric = report.get("exposed_valid_development_metric")
    claim = report.get("claim_policy")
    expected_checkpoint_hashes = dict(zip(ROLES, EXPECTED_CHECKPOINT_SHA256, strict=True))
    if (
        report.get("schema") != "wnn.tl3-serving-old-reference-valid-development.v1"
        or report.get("candidate") != "historical_tl3_source4_engineering_references"
        or not isinstance(models, dict)
        or not isinstance(sources, dict)
        or not isinstance(execution, dict)
        or not isinstance(metric, dict)
        or not isinstance(claim, dict)
        or claim.get("validation_status") != "exposed_development_data"
        or claim.get("independent_evaluation") is not False
        or claim.get("final_model_score_claim") != "forbidden"
        or claim.get("not_a_final_full_corpus_model_set") is not True
        or models.get("roles") != list(ROLES)
        or models.get("run_references") != list(RUNS)
        or models.get("checkpoint_sha256") != expected_checkpoint_hashes
        or models.get("requested_precision") != precision
        or models.get("serving_policy") != "w18_only"
        or models.get("active_member_indices") != [0]
        or models.get("activation_mode") != activation_mode
        or sources.get("valid_sha256") != valid_sha256
        or sources.get("source_validation_receipt_sha256") != SOURCE_VALIDATION_SHA256
        or sources.get("sequence_length") != SEQUENCE_LENGTH
        or sources.get("row_groups") != list(range(EXPECTED_OLD_VALID_GROUPS))
        or sources.get("scoring_mask") != "is_scored AND step_in_seq>=99"
        or execution.get("workers") != 32
        or execution.get("worker_cap") != 32
        or execution.get("whole_sequence_contiguous_partitions") is not True
        or execution.get("all_rows_causally_replayed") is not True
        or execution.get("global_accumulator_moments_merged_in_partition_order") is not True
        or execution.get("native_library_sha256") != native_sha256
        or metric.get("blocks") != EXPECTED_OLD_VALID_GROUPS
        or metric.get("selected_rows") != EXPECTED_OLD_VALID_ROWS
        or metric.get("effective_scored_rows") != EXPECTED_OLD_VALID_ROWS
    ):
        raise ValueError(f"{mode} old-reference report differs from the frozen fidelity contract")
    sequence_ids_sha256 = sources.get("sequence_ids_sha256")
    if (
        not isinstance(sequence_ids_sha256, str)
        or len(sequence_ids_sha256) != 64
        or not set(sequence_ids_sha256) <= _SHA256
    ):
        raise ValueError(f"{mode} report has a malformed valid sequence-ID digest")
    precisions = models.get("effective_precision_by_member")
    if not isinstance(precisions, list) or len(precisions) != 4:
        raise ValueError(f"{mode} report lacks per-member effective precision")
    if precision == "f32" and precisions != ["f32"] * 4:
        raise ValueError("FP32 fidelity report did not use four FP32 member models")
    if precision == "q8vnni" and not all(
        isinstance(name, str) and name.startswith("q8vnni") for name in precisions
    ):
        raise ValueError("Q8 fidelity report did not use dynamic Q8 kernels")
    if mode == "q8" and activation_mode != "fast":
        raise ValueError("Q8 fidelity report must use the frozen fast activation mode")
    if mode == "f32" and activation_mode != "exact":
        raise ValueError("FP32 fidelity report must use exact activations")
    value = _finite_number(metric.get("weighted_pearson"), f"{mode} weighted_pearson")
    if metric.get("wp") != value:
        raise ValueError(f"{mode} report's weighted_pearson aliases differ")
    return value, sequence_ids_sha256


def _validate_quantization_contract(
    contract_path: Path,
    expected_sha256: str,
    *,
    valid_sha256: str,
    source_validation_sha256: str,
) -> dict[str, object]:
    """Validate frozen old-reference fidelity and pre-final numeric ceilings."""
    if expected_sha256 != EXPECTED_QUANTIZATION_CONTRACT_SHA256:
        raise ValueError("quantization contract SHA256 is not the approved frozen R1 contract")
    contract, contract_path = _read_pinned_json(
        contract_path, expected_sha256, "W18 final quantization admission contract"
    )
    expected_fields = {
        "schema",
        "serving_policy",
        "precision",
        "fast_activations",
        "native_v3_sha256",
        "source_validation_sha256",
        "numeric_ceilings",
        "fidelity_contract",
        "paired_reference_reports",
        "fidelity_result",
        "created_epoch",
        "claim_policy",
    }
    if set(contract) != expected_fields:
        raise ValueError("quantization contract has missing or unsupported fields")
    if (
        contract.get("schema") != QUANTIZATION_CONTRACT_SCHEMA
        or contract.get("serving_policy") != "w18_only"
        or contract.get("precision") != "q8vnni"
        or contract.get("fast_activations") is not True
        or contract.get("native_v3_sha256") != EXPECTED_R11_NATIVE_V3_SHA256
        or contract.get("source_validation_sha256") != source_validation_sha256
        or source_validation_sha256 != SOURCE_VALIDATION_SHA256
        or contract.get("claim_policy")
        != "old-reference exposed valid fidelity only; no final model score or FP32 equivalence"
        or not _is_positive_finite_epoch(contract.get("created_epoch"))
    ):
        raise ValueError("quantization contract differs from frozen R11 serving identities")

    ceilings, ceilings_path, ceilings_sha256 = _read_contract_artifact(
        contract_path, contract.get("numeric_ceilings"), "W18 numeric error ceilings"
    )
    ceiling_fields = {
        "schema",
        "calibration_report_path",
        "calibration_report_sha256",
        "old_w18_checkpoint_sha256",
        "multiplier_fixed_before_new_weights_check",
        "maximum_rms_by_target",
        "maximum_p99_abs_by_target",
        "maximum_abs_by_target",
        "original_allclose_atol",
        "original_allclose_rtol",
        "original_allclose_failed_preserved",
        "score_computed",
        "created_epoch",
    }
    if set(ceilings) != ceiling_fields:
        raise ValueError("W18 numeric ceilings have missing or unsupported fields")
    if (
        ceilings.get("schema") != NUMERIC_CEILINGS_SCHEMA
        or ceilings.get("old_w18_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256[0]
        or ceilings.get("multiplier_fixed_before_new_weights_check") != 2
        or ceilings.get("original_allclose_atol") != 0.02
        or ceilings.get("original_allclose_rtol") != 0.02
        or ceilings.get("original_allclose_failed_preserved") is not True
        or ceilings.get("score_computed") is not False
        or not _is_positive_finite_epoch(ceilings.get("created_epoch"))
    ):
        raise ValueError("W18 numeric ceilings changed the fixed calibration policy")
    for field, expected in (
        ("maximum_rms_by_target", EXPECTED_RMS_CEILINGS),
        ("maximum_p99_abs_by_target", EXPECTED_P99_CEILINGS),
        ("maximum_abs_by_target", EXPECTED_MAX_CEILINGS),
    ):
        values = ceilings.get(field)
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(
                not math.isclose(_finite_number(value, field), bound, rel_tol=0, abs_tol=1e-15)
                for value, bound in zip(values, expected, strict=True)
            )
        ):
            raise ValueError(f"{field} differs from the frozen per-target ceilings")

    calibration_path = Path(str(ceilings["calibration_report_path"]))
    if not calibration_path.is_absolute():
        calibration_path = ceilings_path.parent / calibration_path
    calibration, calibration_path = _read_pinned_json(
        calibration_path,
        str(ceilings["calibration_report_sha256"]),
        "old W18 one-sequence quantization calibration report",
    )
    if (
        calibration.get("schema") != "wnn.w18-only-engineering-benchmark.v1"
        or calibration.get("historical_train_only_reference") is not True
        or calibration.get("serving_policy") != "w18_only"
        or calibration.get("active_member_indices") != [0]
        or calibration.get("effective_precisions") != ["q8vnni"] * 4
        or calibration.get("sequence_source") != "official-train-row-group-0"
        or calibration.get("sequence_id") != 43_601
        or calibration.get("rows") != SEQUENCE_LENGTH
        or calibration.get("warmup") != _callback.WARMUP
        or calibration.get("reset_replay_identical") is not True
        or calibration.get("allclose_at_0_02") is not False
        or calibration.get("score_computed") is not False
    ):
        raise ValueError("old W18 calibration report does not match the frozen Q8 diagnostic")
    for stat_field, ceiling_field in (
        ("rms_by_target", "maximum_rms_by_target"),
        ("p99_abs_by_target", "maximum_p99_abs_by_target"),
        ("max_abs_by_target", "maximum_abs_by_target"),
    ):
        observed = calibration.get(stat_field)
        ceilings_values = ceilings[ceiling_field]
        if (
            not isinstance(observed, list)
            or len(observed) != 2
            or any(
                not math.isclose(
                    _finite_number(bound, ceiling_field),
                    2.0 * _finite_number(value, stat_field),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                for value, bound in zip(observed, ceilings_values, strict=True)
            )
        ):
            raise ValueError(f"{ceiling_field} is not exactly twice the frozen calibration")

    fidelity, fidelity_path, fidelity_sha256 = _read_contract_artifact(
        contract_path, contract.get("fidelity_contract"), "W18 old-reference fidelity contract"
    )
    fidelity_fields = {
        "schema",
        "mean_weighted_pearson_degradation_budget",
        "evaluation",
        "old_w18_sha256",
        "source_validation_sha256",
        "native_library_sha256",
        "helper_sha256",
        "pairs",
        "serving_policy",
        "workers_per_comparison",
        "timeout_seconds",
        "created_epoch",
    }
    expected_pairs = [
        {"precision": "f32", "activation_mode": "exact"},
        {"precision": "q8vnni", "activation_mode": "fast"},
    ]
    budget = 0.0002
    declared_native_sha256 = fidelity.get("native_library_sha256")
    legacy_pin_error = None
    if declared_native_sha256 == LEGACY_MALFORMED_NATIVE_SHA256_DECLARATION:
        legacy_pin_error = {
            "kind": "known_historical_malformed_declaration",
            "declared_sha256": LEGACY_MALFORMED_NATIVE_SHA256_DECLARATION,
            "actual_sha256": EXPECTED_R11_NATIVE_V3_SHA256,
            "binding": "combined contract and both executed paired reports",
        }
    if (
        set(fidelity) != fidelity_fields
        or fidelity.get("schema") != FIDELITY_CONTRACT_SCHEMA
        or fidelity.get("mean_weighted_pearson_degradation_budget") != budget
        or fidelity.get("evaluation")
        != "old-reference exposed valid engineering fidelity; no final model score"
        or fidelity.get("old_w18_sha256") != EXPECTED_CHECKPOINT_SHA256[0]
        or fidelity.get("source_validation_sha256") != source_validation_sha256
        or declared_native_sha256
        not in (
            EXPECTED_R11_NATIVE_V3_SHA256,
            LEGACY_MALFORMED_NATIVE_SHA256_DECLARATION,
        )
        or fidelity.get("helper_sha256") != EXPECTED_SERVING_HELPER_SHA256
        or fidelity.get("pairs") != expected_pairs
        or fidelity.get("serving_policy") != "w18_only"
        or fidelity.get("workers_per_comparison") != 32
        or fidelity.get("timeout_seconds") != 1_200
        or not _is_positive_finite_epoch(fidelity.get("created_epoch"))
    ):
        raise ValueError("old-reference fidelity contract differs from its frozen policy")

    reports_spec = contract.get("paired_reference_reports")
    if not isinstance(reports_spec, dict) or set(reports_spec) != {"f32", "q8"}:
        raise ValueError("quantization contract must pin exactly the FP32 and Q8 reports")
    f32_report, f32_report_path, f32_report_sha = _read_contract_artifact(
        contract_path, reports_spec["f32"], "old W18 FP32 valid report"
    )
    q8_report, q8_report_path, q8_report_sha = _read_contract_artifact(
        contract_path, reports_spec["q8"], "old W18 Q8 valid report"
    )
    f32_value, f32_sequence_ids_sha = _validate_old_reference_report(
        f32_report,
        mode="f32",
        precision="f32",
        activation_mode="exact",
        valid_sha256=valid_sha256,
        native_sha256=EXPECTED_R11_NATIVE_V3_SHA256,
    )
    q8_value, q8_sequence_ids_sha = _validate_old_reference_report(
        q8_report,
        mode="q8",
        precision="q8vnni",
        activation_mode="fast",
        valid_sha256=valid_sha256,
        native_sha256=EXPECTED_R11_NATIVE_V3_SHA256,
    )
    if f32_sequence_ids_sha != q8_sequence_ids_sha:
        raise ValueError("paired fidelity reports use different valid sequence-ID memberships")
    fidelity_result = contract.get("fidelity_result")
    if not isinstance(fidelity_result, dict) or set(fidelity_result) != {
        "metric",
        "fp32",
        "q8",
        "delta",
        "minimum_delta",
        "passed",
    }:
        raise ValueError("quantization contract fidelity result has unsupported fields")
    expected_delta = q8_value - f32_value
    minimum_delta = -budget
    delta = _finite_number(fidelity_result.get("delta"), "old-reference fidelity delta")
    if (
        fidelity_result.get("metric") != "weighted_pearson"
        or not math.isclose(f32_value, EXPECTED_OLD_VALID_F32, rel_tol=0, abs_tol=1e-12)
        or not math.isclose(q8_value, EXPECTED_OLD_VALID_Q8, rel_tol=0, abs_tol=1e-12)
        or not math.isclose(expected_delta, EXPECTED_OLD_VALID_DELTA, rel_tol=0, abs_tol=1e-12)
        or not math.isclose(
            _finite_number(fidelity_result.get("fp32"), "old FP32 metric"),
            f32_value,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _finite_number(fidelity_result.get("q8"), "old Q8 metric"),
            q8_value,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or not math.isclose(delta, expected_delta, rel_tol=0, abs_tol=1e-12)
        or fidelity_result.get("minimum_delta") != minimum_delta
        or fidelity_result.get("passed") is not (expected_delta >= minimum_delta)
    ):
        raise ValueError(
            "combined quantization contract fidelity result disagrees with pinned reports"
        )
    return {
        "path": str(contract_path),
        "sha256": expected_sha256,
        "claim_policy": contract["claim_policy"],
        "native_v3_sha256": contract["native_v3_sha256"],
        "source_validation_sha256": contract["source_validation_sha256"],
        "numeric_ceilings_path": str(ceilings_path),
        "numeric_ceilings_sha256": ceilings_sha256,
        "calibration_report_path": str(calibration_path),
        "calibration_report_sha256": str(ceilings["calibration_report_sha256"]),
        "calibration_sequence_id": calibration["sequence_id"],
        "fidelity_contract_path": str(fidelity_path),
        "fidelity_contract_sha256": fidelity_sha256,
        "fidelity_contract_declared_native_library_sha256": declared_native_sha256,
        "legacy_declared_pin_error": legacy_pin_error,
        "reference_reports": {
            "f32": {
                "path": str(f32_report_path),
                "sha256": f32_report_sha,
                "weighted_pearson": f32_value,
            },
            "q8": {
                "path": str(q8_report_path),
                "sha256": q8_report_sha,
                "weighted_pearson": q8_value,
            },
        },
        "fidelity_delta": expected_delta,
        "minimum_delta": minimum_delta,
        "fidelity_passed": bool(expected_delta >= minimum_delta),
        "shared_sequence_ids_sha256": f32_sequence_ids_sha == q8_sequence_ids_sha,
        "ceilings": ceilings,
    }


def _safe_zip_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Reject traversal, links, duplicate names, and unexpected ZIP encodings."""
    result: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        name = info.filename
        relative = PurePosixPath(name)
        mode = info.external_attr >> 16
        if (
            not name
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or name in result
            or (mode & 0o170000) == 0o120000
            or info.flag_bits & 0x1
        ):
            raise ValueError(f"unsafe or duplicate final ZIP member: {name!r}")
        result[name] = info
    return result


def _extract_checked(
    archive: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo], root: Path
) -> None:
    """Extract only already-validated regular ZIP files without using extractall."""
    for name, info in members.items():
        if info.is_dir():
            continue
        target = root.joinpath(*PurePosixPath(name).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("xb") as destination:
            shutil.copyfileobj(source, destination)


def _check_member_digests(
    archive: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo], expected: object
) -> dict[str, str]:
    if not isinstance(expected, dict) or set(expected) != set(members):
        raise ValueError("build receipt ZIP member list differs from the delivered artifact")
    actual: dict[str, str] = {}
    for name, info in members.items():
        digest = hashlib.sha256(archive.read(info)).hexdigest()
        if expected.get(name) != digest:
            raise ValueError(f"delivered ZIP member hash differs from its build receipt: {name}")
        actual[name] = digest
    return actual


def _verify_checkpoint_receipt(
    checkpoint_paths: dict[str, Path], receipt_path: Path, receipt_sha256: str
) -> tuple[dict[str, Any], Any]:
    """Verify exact receipt bytes and each actual final checkpoint's source identity."""
    checked = load_final_model_receipt(receipt_path, receipt_sha256)
    receipt = checked["receipt"]
    membership = checked["membership"]
    if set(checkpoint_paths) != set(ROLES):
        raise ValueError("checker requires all four receipt-bound final checkpoint paths")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    members = receipt["members"]
    for role in ROLES:
        path = checkpoint_paths[role]
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"{role} checkpoint must be a regular file: {path}")
        member = members[role]
        if _sha256_file(path) != member["checkpoint_sha256"]:
            raise ValueError(f"{role} checkpoint bytes differ from the final-model receipt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"{role} checkpoint is not a mapping")
        _validate_final_checkpoint(role, checkpoint, member, membership)
    return checked, torch


def _torch_w18_reference(checkpoint_path: Path, raw: np.ndarray, torch: Any) -> np.ndarray:
    """Run the receipt-bound FP32 W18 model on the official raw112 sequence."""
    from wnn.features import event_features_torch
    from wnn.nn import RNNModel

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    args = checkpoint["args"]
    model = RNNModel(
        n_in=211,
        hidden=args["hidden"],
        layers=args["layers"],
        dropout=args["dropout"],
        n_out=state["head.2.weight"].shape[0],
        cell=args["cell"],
        enc=args["enc"],
        readout_activation=args["readout_activation"],
        numeric_map=args["numeric_map"],
        ple_intervals=args["ple_intervals"],
        head_layers=args["head_layers"],
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    raw_tensor = torch.from_numpy(np.ascontiguousarray(raw[None], dtype=np.float32))
    with torch.inference_mode():
        events = event_features_torch(raw_tensor)
        outputs = model(torch.cat((raw_tensor, events), dim=-1))[0][0, :, :2]
    return np.ascontiguousarray(outputs.cpu().numpy(), dtype=np.float32)


def _run_package_callback(model: Any, sequence_id: int, raw: np.ndarray, warmup: int) -> np.ndarray:
    predictions = np.full((len(raw), 2), np.nan, dtype=np.float32)
    for step, row in enumerate(raw):
        value = model.predict(DataPoint(sequence_id, step, step >= warmup, row))
        if step < warmup:
            if value is not None:
                raise AssertionError(f"delivered callback emitted during warm-up row {step}")
            continue
        if value is None:
            raise AssertionError(f"delivered callback omitted required output at row {step}")
        output = np.asarray(value, dtype=np.float32)
        if output.shape != (2,) or not np.isfinite(output).all():
            raise AssertionError(f"delivered callback returned invalid output at row {step}")
        predictions[step] = output
    return predictions


def _peak_rss_bytes() -> int | None:
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes. The official target is Linux.
        return peak * 1024 if sys.platform.startswith("linux") else peak
    except (ImportError, OSError, ValueError):
        return None


def _pin_one_cpu() -> int | None:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    available = sorted(os.sched_getaffinity(0))
    if not available:
        raise RuntimeError("process has no available CPU for the one-core check")
    selected = available[0]
    os.sched_setaffinity(0, {selected})
    return selected


def run(args: argparse.Namespace) -> dict[str, object]:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"exact package ABI check requires Python 3.11, found {sys.version}")
    if not sys.platform.startswith("linux"):
        raise RuntimeError("exact submission ABI and one-core check requires Linux")
    if args.sequence_index < 0:
        raise ValueError("sequence index must be nonnegative")
    if args.zip_path.is_symlink() or not args.zip_path.is_file():
        raise FileNotFoundError(f"submission ZIP must be a regular file: {args.zip_path}")
    zip_bytes = args.zip_path.stat().st_size
    if zip_bytes > MAX_ZIP_BYTES:
        raise ValueError(f"submission ZIP is {zip_bytes} bytes; limit is {MAX_ZIP_BYTES}")
    artifact_sha256 = _sha256_file(args.zip_path)

    checked_receipt, torch = _verify_checkpoint_receipt(
        {
            "w18": args.w18,
            "tiedlead_s5": args.tiedlead_s5,
            "tiedlead_s6": args.tiedlead_s6,
            "source4_lstm": args.lstm,
        },
        args.final_model_receipt,
        args.final_model_receipt_sha256,
    )
    receipt = checked_receipt["receipt"]
    membership = checked_receipt["membership"]
    if args.sequence_index != 0:
        raise ValueError("final-weight numeric parity is fixed to official train row group 0")
    quantization = _validate_quantization_contract(
        args.quantization_contract,
        args.quantization_contract_sha256,
        valid_sha256=membership.valid_sha256,
        source_validation_sha256=membership.source_validation_sha256,
    )

    build_receipt = json.loads(args.build_receipt.read_text(encoding="utf-8"))
    if build_receipt.get("artifact", {}).get("sha256") != artifact_sha256:
        raise ValueError("build receipt artifact SHA differs from the supplied ZIP")
    if build_receipt.get("artifact", {}).get("bytes") != zip_bytes:
        raise ValueError("build receipt artifact size differs from the supplied ZIP")
    if (
        build_receipt.get("serving_policy") != "w18_only"
        or build_receipt.get("active_member_indices") != [0]
        or build_receipt.get("precision") != "q8vnni"
        or build_receipt.get("fast_activations") is not True
        or build_receipt.get("candidate_blend") != BLEND
        or build_receipt.get("applied_blend") is not None
    ):
        raise ValueError("build receipt does not identify the actual W18-only callback policy")

    package_root = Path(tempfile.mkdtemp(prefix="final-tl3-package-check-"))
    try:
        with zipfile.ZipFile(args.zip_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"submission ZIP CRC failed at {bad_member}")
            zip_members = _safe_zip_members(archive)
            member_hashes = _check_member_digests(
                archive, zip_members, build_receipt.get("artifact", {}).get("members")
            )
            _extract_checked(archive, zip_members, package_root)

        manifest = json.loads((package_root / "members.json").read_text(encoding="utf-8"))
        if (
            manifest.get("schema") != "wnn.final-tl3-lstm-members.v1"
            or manifest.get("candidate_scope") != FINAL_SCOPE
            or manifest.get("serving_policy") != "w18_only"
            or manifest.get("precision") != "q8vnni"
            or manifest.get("fast_activations") is not True
            or manifest.get("active_member_indices") != [0]
            or manifest.get("candidate_blend") != BLEND
            or manifest.get("blend") is not None
            or manifest.get("independent_evaluation_sequences") != 0
            or manifest.get("independent_score_claim") != "forbidden"
            or manifest.get("final_fit") != receipt.get("final_fit")
            or manifest.get("final_model_receipt", {}).get("schema") != FINAL_RECEIPT_SCHEMA
            or manifest.get("final_model_receipt", {}).get("sha256") != checked_receipt["sha256"]
        ):
            raise ValueError("delivered ZIP metadata is not the approved W18-only final policy")
        manifest_members = manifest.get("members")
        if not isinstance(manifest_members, list) or len(manifest_members) != 4:
            raise ValueError("delivered final ZIP must retain all four receipt-bound members")
        for role, record in zip(ROLES, manifest_members, strict=True):
            expected = receipt["members"][role]
            if (
                record.get("role") != role
                or record.get("run_reference") != FINAL_RUNS[ROLES.index(role)]
                or record.get("checkpoint_sha256") != expected["checkpoint_sha256"]
            ):
                raise ValueError(f"delivered ZIP has a changed {role} receipt binding")
        if (package_root / "precision.txt").read_text(encoding="utf-8") != "q8vnni\n":
            raise ValueError("delivered precision marker differs from the serving policy")
        native_paths = [
            package_root / level / "wnn_native.abi3.so" for level in ("v2", "avx2", "v3")
        ]
        if any(not path.is_file() for path in native_paths):
            raise ValueError("delivered ZIP lacks one or more Python ABI3 Linux libraries")
        native_v3_sha256 = _sha256_file(package_root / "v3" / "wnn_native.abi3.so")
        built_native_hashes = build_receipt.get("native_libraries")
        if (
            native_v3_sha256 != EXPECTED_R11_NATIVE_V3_SHA256
            or native_v3_sha256 != quantization["native_v3_sha256"]
            or not isinstance(built_native_hashes, dict)
            or built_native_hashes.get("v3/wnn_native.abi3.so") != native_v3_sha256
        ):
            raise ValueError("delivered v3 library differs from the frozen Q8 calibration build")
        selected_cpu = _pin_one_cpu()

        source_sha = membership.train_sha256
        if (
            args.data.is_symlink()
            or not args.data.is_file()
            or _sha256_file(args.data) != source_sha
        ):
            raise ValueError("sequence data bytes differ from the receipt-bound official source")
        sequences = SequenceFile(args.data)
        if args.sequence_index >= len(sequences):
            raise ValueError("sequence index lies outside the chosen official source")
        sequence = sequences.read(args.sequence_index, targets=False, mask=False)
        raw = sequence.features
        if sequence.seq_ix != quantization["calibration_sequence_id"]:
            raise ValueError("train row group 0 differs from the frozen calibration sequence ID")
        if raw.shape != (SEQUENCE_LENGTH, 112) or not raw.flags.c_contiguous:
            raise ValueError("chosen row group is not one canonical 20,000-row raw112 sequence")
        if not np.isfinite(raw).all():
            raise ValueError("chosen official raw112 sequence contains nonfinite values")
        row_contract = pq.ParquetFile(args.data).read_row_group(
            args.sequence_index,
            columns=["seq_ix", "step_in_seq"],
            use_threads=False,
        )
        row_sequence_ids = row_contract.column("seq_ix").to_numpy()
        row_steps = row_contract.column("step_in_seq").to_numpy()
        if (
            len(row_sequence_ids) != SEQUENCE_LENGTH
            or not np.all(row_sequence_ids == sequence.seq_ix)
            or not np.array_equal(row_steps, np.arange(SEQUENCE_LENGTH))
        ):
            raise ValueError("chosen row group violates the canonical seq_ix/step order")

        module_spec = importlib.util.spec_from_file_location(
            "delivered_final_solution", package_root / "solution.py"
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError("could not load exact delivered solution.py")
        delivered = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = delivered
        module_spec.loader.exec_module(delivered)
        model = delivered.PredictionModel()
        native = model.model
        if (
            native.serving_policy != "w18_only"
            or tuple(native.active_member_indices) != (0,)
            or not native.fast_activations
            or not all(name.startswith("q8vnni") for name in native.effective_precisions)
        ):
            raise RuntimeError("delivered callback runtime differs from its declared W18-only mode")

        w18_path = args.w18.resolve(strict=True)
        reference = _torch_w18_reference(w18_path, raw, torch)
        warmup = int(_callback.WARMUP)
        started = time.perf_counter()
        first = _run_package_callback(model, sequence.seq_ix, raw, warmup)
        elapsed = time.perf_counter() - started
        scored_rows = slice(warmup, None)
        ceilings = quantization["ceilings"]
        atol = float(ceilings["original_allclose_atol"])
        rtol = float(ceilings["original_allclose_rtol"])
        numeric_summary = _numeric_error_summary(
            first[scored_rows], reference[scored_rows], atol=atol, rtol=rtol
        )
        numeric_ceilings_pass = _numeric_error_within_ceilings(numeric_summary, ceilings)
        allclose_diagnostic = bool(numeric_summary["allclose"])

        native.reset()
        replay = _run_package_callback(model, sequence.seq_ix, raw, warmup)
        replay_identical = np.array_equal(first, replay, equal_nan=True)

        prefix = min(1_000, SEQUENCE_LENGTH - 1)
        changed = raw.copy()
        changed[prefix:, 0] += np.float32(0.25)
        native.reset()
        future = _run_package_callback(model, sequence.seq_ix, changed, warmup)
        future_prefix_invariant = np.array_equal(first[:prefix], future[:prefix], equal_nan=True)

        peak_rss = _peak_rss_bytes()
        row_microseconds = elapsed * 1e6 / SEQUENCE_LENGTH
        projected_seconds = row_microseconds * OFFICIAL_SEQUENCES * SEQUENCE_LENGTH / 1e6
        runtime_pass = projected_seconds <= OFFICIAL_SECONDS
        memory_pass = peak_rss is not None and peak_rss <= MEMORY_LIMIT_BYTES
        fidelity_pass = bool(quantization["fidelity_passed"])
        eligibility_pass = bool(
            zip_bytes <= MAX_ZIP_BYTES
            and runtime_pass
            and memory_pass
            and fidelity_pass
            and numeric_ceilings_pass
            and replay_identical
            and future_prefix_invariant
        )
        return {
            "schema": "wnn.final-tl3-package-serving-check.v1",
            "artifact_sha256": artifact_sha256,
            "artifact_bytes": zip_bytes,
            "artifact_under_20mb": zip_bytes <= MAX_ZIP_BYTES,
            "build_receipt_sha256": _sha256_file(args.build_receipt),
            "final_model_receipt_sha256": checked_receipt["sha256"],
            "quantization_contract": {
                "path": quantization["path"],
                "sha256": quantization["sha256"],
                "fidelity_contract_sha256": quantization["fidelity_contract_sha256"],
                "numeric_ceilings_sha256": quantization["numeric_ceilings_sha256"],
                "calibration_report_sha256": quantization["calibration_report_sha256"],
                "reference_reports": quantization["reference_reports"],
                "claim_policy": quantization["claim_policy"],
                "old_reference_fidelity_delta": quantization["fidelity_delta"],
                "old_reference_fidelity_minimum_delta": quantization["minimum_delta"],
                "old_reference_fidelity_passed": fidelity_pass,
            },
            "python": platform.python_version(),
            "native_abi": "linux-cp37-abi3-compatible-imported-under-py311",
            "serving_policy": native.serving_policy,
            "precision_requested": manifest["precision"],
            "effective_precisions": list(native.effective_precisions),
            "fast_activations": native.fast_activations,
            "active_member_indices": list(native.active_member_indices),
            "sequence_source": "train",
            "sequence_index": args.sequence_index,
            "sequence_id": sequence.seq_ix,
            "sequence_rows": len(raw),
            "warmup_rows": warmup,
            "torch_device": "cpu",
            "original_allclose_diagnostic_only": {
                "atol": atol,
                "rtol": rtol,
                "old_calibration_failed_preserved": True,
                "new_weight_passed": allclose_diagnostic,
            },
            "max_absolute_error_vs_torch_w18": numeric_summary["max_abs"],
            "max_relative_error_vs_torch_w18": numeric_summary["max_relative"],
            "numeric_error_by_target": {
                "rms": numeric_summary["rms"].tolist(),
                "p99_abs": numeric_summary["p99_abs"].tolist(),
                "max_abs": numeric_summary["max_abs_by_target"].tolist(),
            },
            "numeric_ceilings_by_target": {
                "rms": ceilings["maximum_rms_by_target"],
                "p99_abs": ceilings["maximum_p99_abs_by_target"],
                "max_abs": ceilings["maximum_abs_by_target"],
            },
            "new_weight_numeric_ceilings_pass": numeric_ceilings_pass,
            "reset_replay_identical": replay_identical,
            "future_prefix_invariant": future_prefix_invariant,
            "timed_callback_seconds_one_cpu": elapsed,
            "timed_callback_microseconds_per_row": row_microseconds,
            "official_projected_seconds_1970_sequences": projected_seconds,
            "official_time_limit_seconds": OFFICIAL_SECONDS,
            "official_time_projection_pass": runtime_pass,
            "peak_rss_bytes": peak_rss,
            "official_memory_limit_bytes": MEMORY_LIMIT_BYTES,
            "official_memory_projection_pass": memory_pass,
            "eligibility_gates_pass": eligibility_pass,
            "pinned_cpu": selected_cpu,
            "zip_member_sha256": member_hashes,
            "score_computed": False,
            "independent_score_claim": "forbidden",
        }
    finally:
        shutil.rmtree(package_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--final-model-receipt", type=Path, required=True)
    parser.add_argument("--final-model-receipt-sha256", required=True)
    parser.add_argument("--quantization-contract", type=Path, required=True)
    parser.add_argument("--quantization-contract-sha256", required=True)
    parser.add_argument("--w18", type=Path, required=True)
    parser.add_argument("--tiedlead-s5", dest="tiedlead_s5", type=Path, required=True)
    parser.add_argument("--tiedlead-s6", dest="tiedlead_s6", type=Path, required=True)
    parser.add_argument("--lstm", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sequence-index", type=int, default=0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    if not result["eligibility_gates_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
