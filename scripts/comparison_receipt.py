"""Validate completed scientific comparisons and record durable receipts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

METRIC_ID = "global_wp_clip2_v1"
SHA256_RE = re.compile(r"[a-f0-9]{64}\Z")
MEMBER_RE = re.compile(r"([A-Za-z0-9_.-]{1,96}@ep[0-9]+)\Z")


class ComparisonReceiptError(ValueError):
    """Raised when a completed queue item cannot support a valid comparison receipt."""


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ComparisonReceiptError(f"{label} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_object(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    try:
        raw = path.read_bytes()
        return _mapping(json.loads(raw), label), raw
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ComparisonReceiptError(f"cannot read {label}: {path}: {error}") from error


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonReceiptError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComparisonReceiptError(f"{label} must be a finite number")
    return result


def _valid_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ComparisonReceiptError(f"{label} must be a lowercase SHA256")
    return value


def _resolve_path(value: object, label: str, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ComparisonReceiptError(f"comparison.{label} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _member_specs(value: object, label: str) -> list[tuple[str, float]]:
    if not isinstance(value, str) or not value:
        raise ComparisonReceiptError(f"comparison.{label} must declare RUN@epN or a blend")
    result: list[tuple[str, float]] = []
    for part in value.split(","):
        endpoint, marker, weight_text = part.partition("*")
        if not MEMBER_RE.fullmatch(endpoint):
            raise ComparisonReceiptError(f"invalid endpoint in comparison.{label}: {endpoint!r}")
        if not marker:
            weight = 1.0
        else:
            try:
                parsed_weight = float(weight_text)
            except ValueError as error:
                raise ComparisonReceiptError(
                    f"{label} blend weight must be a finite number"
                ) from error
            weight = _finite_number(parsed_weight, f"{label} blend weight")
        result.append((endpoint, weight))
    return result


def _member_identity(binding: dict[str, object], label: str) -> str:
    run_at = binding.get("run_at")
    if isinstance(run_at, str):
        return run_at
    run = binding.get("run")
    epoch = binding.get("epoch")
    if isinstance(run, str) and isinstance(epoch, int) and not isinstance(epoch, bool):
        return f"{run}@ep{epoch}"
    raise ComparisonReceiptError(f"{label} lacks a member identity")


def _validate_members(
    value: object, expected: list[tuple[str, float]], split_sha256: str, label: str
) -> list[dict[str, object]]:
    if not isinstance(value, list) or len(value) != len(expected):
        raise ComparisonReceiptError(f"{label} member bindings do not match the declaration")
    members: list[dict[str, object]] = []
    for index, (raw_binding, (expected_identity, expected_weight)) in enumerate(
        zip(value, expected, strict=True)
    ):
        binding = _mapping(raw_binding, f"{label} member {index}")
        identity = _member_identity(binding, f"{label} member {index}")
        if identity != expected_identity:
            raise ComparisonReceiptError(f"{label} member identity does not match declaration")
        _valid_hash(binding.get("checkpoint_sha256"), f"{label} checkpoint hash")
        _valid_hash(binding.get("prediction_sha256"), f"{label} prediction hash")
        if binding.get("split_manifest_sha256") != split_sha256:
            raise ComparisonReceiptError(f"{label} member split hash does not match split file")
        weight = _finite_number(binding.get("weight"), f"{label} member weight")
        if not math.isclose(weight, expected_weight, rel_tol=0.0, abs_tol=1e-12):
            raise ComparisonReceiptError(f"{label} member weight does not match declaration")
        members.append(binding)
    return members


def _validate_input_hashes(comparison: dict[str, object], base_dir: Path) -> object:
    present = [key for key in ("input_hashes", "exact_input_hashes") if key in comparison]
    if len(present) > 1:
        raise ComparisonReceiptError("declare only one of input_hashes or exact_input_hashes")
    if not present:
        return None
    declared = comparison[present[0]]
    hashes = _mapping(declared, f"comparison.{present[0]}")
    for name, value in hashes.items():
        if isinstance(value, str):
            _valid_hash(value, f"input hash {name}")
            continue
        item = _mapping(value, f"input hash {name}")
        expected = _valid_hash(item.get("sha256"), f"input hash {name}.sha256")
        if "path" in item:
            path = _resolve_path(item["path"], f"{present[0]}.{name}.path", base_dir)
            if _sha256(path) != expected:
                raise ComparisonReceiptError(f"declared input hash mismatch for {name}")
    return hashes


def _phase_timing(steps: list[dict[str, object]]) -> dict[str, object]:
    phases: list[dict[str, object]] = []
    durations: list[float] = []
    complete = True
    for index, step in enumerate(steps):
        name = step.get("name", f"step_{index}")
        status = step.get("status")
        duration_value = step.get("wall_seconds")
        duration: float | None = None
        if isinstance(duration_value, (int, float)) and not isinstance(duration_value, bool):
            observed = float(duration_value)
            if math.isfinite(observed) and observed >= 0:
                duration = observed
        if duration is None:
            complete = False
        else:
            durations.append(duration)
        phases.append({"name": name, "status": status, "wall_seconds": duration})
    return {
        "measurement": "subprocess_wall_seconds",
        "phases": phases,
        "total_wall_seconds": sum(durations) if complete else None,
    }


def _commit_receipt_and_ledger(
    receipt_path: Path,
    receipt: dict[str, object],
    ledger_path: Path,
    row: dict[str, object],
    identity: dict[str, object],
) -> bool:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ledger_path.open("a+", encoding="utf-8") as ledger:
            fcntl.flock(ledger.fileno(), fcntl.LOCK_EX)
            ledger.seek(0)
            already_present = False
            for line_number, line in enumerate(ledger, start=1):
                if not line.strip():
                    continue
                try:
                    prior = _mapping(json.loads(line), f"ledger line {line_number}")
                except json.JSONDecodeError as error:
                    raise ComparisonReceiptError(
                        f"scientific ledger has invalid JSON on line {line_number}"
                    ) from error
                if prior.get("experiment_id") != identity["experiment_id"]:
                    continue
                prior_identity = {key: prior.get(key) for key in identity}
                if prior_identity != identity:
                    raise ComparisonReceiptError(
                        f"experiment_id {identity['experiment_id']} already has a conflicting ledger row"
                    )
                already_present = True
            if already_present:
                _atomic_json_write(receipt_path, receipt)
                return False
            _atomic_json_write(receipt_path, receipt)
            ledger.seek(0, os.SEEK_END)
            ledger.write(json.dumps(row, sort_keys=True) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
            return True
    except OSError as error:
        raise ComparisonReceiptError(
            f"cannot commit comparison receipt or scientific ledger {ledger_path}: {error}"
        ) from error


def record_comparison_receipt(
    comparison_value: object,
    *,
    run_id: str,
    spec_bytes: bytes,
    steps: list[dict[str, object]],
    base_dir: Path | None = None,
    declared_spec: object | None = None,
) -> dict[str, object]:
    """Validate the grade, write its receipt, and append one idempotent ledger row.

    Args:
        comparison_value: Comparison declaration embedded in the queue spec.
        run_id: Queue run identifier associated with the exact spec bytes.
        spec_bytes: Original admitted queue spec bytes, used for its exact digest.
        steps: Successful worker step records with measured subprocess wall time.
        base_dir: Directory used to resolve relative comparison paths.
        declared_spec: Full queue spec, used to retain its declared metadata.

    Returns:
        A compact record for the worker result.

    Raises:
        ComparisonReceiptError: If the grade or artifacts do not prove the declaration.
    """
    root = Path.cwd() if base_dir is None else base_dir
    try:
        comparison = _mapping(comparison_value, "comparison")
        for key in ("experiment_id", "hypothesis_id", "candidate"):
            if not isinstance(comparison.get(key), str) or not comparison[key]:
                raise ComparisonReceiptError(f"comparison.{key} must be a non-empty string")
        experiment_id = comparison["experiment_id"]
        hypothesis_id = comparison["hypothesis_id"]
        candidate = comparison["candidate"]
        candidate_members_expected = _member_specs(candidate, "candidate")
        references_value = comparison.get("references")
        if not isinstance(references_value, list) or not references_value:
            raise ComparisonReceiptError("comparison.references must be a non-empty list")
        if any(not isinstance(reference, str) or not reference for reference in references_value):
            raise ComparisonReceiptError("comparison.references must contain non-empty strings")
        references = list(references_value)
        if len(set(references)) != len(references):
            raise ComparisonReceiptError("comparison.references contains duplicate entries")
        reference_members_expected = {
            reference: _member_specs(reference, "reference") for reference in references
        }
        expected_fold = comparison.get("expected_fold")
        if not isinstance(expected_fold, int) or isinstance(expected_fold, bool):
            raise ComparisonReceiptError("comparison.expected_fold must be an integer")
        fidelity = comparison.get("fidelity")
        if fidelity is not None:
            fidelity = _mapping(fidelity, "comparison.fidelity")
        declared_spec_mapping = (
            _mapping(declared_spec, "queue spec") if declared_spec is not None else {}
        )
        declared_metadata = {
            key: declared_spec_mapping[key]
            for key in ("fidelity", "ancestry", "exposure", "limits")
            if key in declared_spec_mapping
        }
        if fidelity is not None:
            declared_metadata["fidelity"] = fidelity
        input_hashes = _validate_input_hashes(comparison, root)

        grade_path = _resolve_path(comparison.get("grade_path"), "grade_path", root)
        receipt_path = _resolve_path(comparison.get("receipt_path"), "receipt_path", root)
        ledger_path = _resolve_path(comparison.get("ledger_path"), "ledger_path", root)
        split_path = _resolve_path(comparison.get("split_path"), "split_path", root)
        resolved_artifacts = [
            grade_path.resolve(),
            receipt_path.resolve(),
            ledger_path.resolve(),
            split_path.resolve(),
        ]
        if len(set(resolved_artifacts)) != len(resolved_artifacts):
            raise ComparisonReceiptError("grade, receipt, ledger, and split paths must be distinct")
        grade, grade_bytes = _read_object(grade_path, "grade")
        split, split_bytes = _read_object(split_path, "split manifest")
        grade_sha256 = hashlib.sha256(grade_bytes).hexdigest()
        split_sha256 = hashlib.sha256(split_bytes).hexdigest()
        if grade.get("metric_id") != METRIC_ID:
            raise ComparisonReceiptError("grade metric_id is not global_wp_clip2_v1")
        _valid_hash(grade.get("valid_targets_sha256"), "grade valid-targets hash")
        if grade.get("candidate") != candidate:
            raise ComparisonReceiptError(
                "grade candidate does not exactly match comparison.candidate"
            )
        grade_split_sha = grade.get("split_manifest_sha256", grade.get("split_sha256"))
        if grade_split_sha != split_sha256:
            raise ComparisonReceiptError("grade split SHA256 does not match the source split file")
        source_fold = split.get("fold", split.get("fold_id"))
        grade_fold = grade.get("expected_fold", grade.get("fold", grade.get("fold_id")))
        grade_split_path = grade.get("split_path")
        if grade_split_path is not None:
            declared_grade_split = _resolve_path(grade_split_path, "grade.split_path", root)
            if declared_grade_split.resolve() != split_path.resolve():
                raise ComparisonReceiptError(
                    "grade split_path does not match comparison.split_path"
                )
        if grade_fold is not None:
            if grade_fold != expected_fold:
                raise ComparisonReceiptError("grade fold does not match comparison.expected_fold")
            if grade_split_path is None:
                raise ComparisonReceiptError("grade expected_fold requires a declared split_path")
            if source_fold is not None and source_fold != expected_fold:
                raise ComparisonReceiptError("split file contradicts grade expected_fold")
        elif source_fold is None or source_fold != expected_fold:
            raise ComparisonReceiptError("split file does not prove comparison.expected_fold")
        official = _mapping(grade.get("official"), "grade.official")
        if official.get("metric_id", METRIC_ID) != METRIC_ID:
            raise ComparisonReceiptError("candidate official metric_id is mismatched")
        candidate_wp = _finite_number(official.get("wp"), "candidate official wp")
        candidate_members = _validate_members(
            grade.get("candidate_members"), candidate_members_expected, split_sha256, "candidate"
        )

        contrasts_value = grade.get("contrasts")
        if not isinstance(contrasts_value, list):
            raise ComparisonReceiptError("grade.contrasts must be a list")
        contrasts: dict[str, dict[str, object]] = {}
        for index, raw_contrast in enumerate(contrasts_value):
            contrast = _mapping(raw_contrast, f"grade.contrasts[{index}]")
            reference = contrast.get("reference")
            if not isinstance(reference, str) or reference in contrasts:
                raise ComparisonReceiptError("grade has a missing or duplicate reference identity")
            contrasts[reference] = contrast
        validated_references: list[dict[str, object]] = []
        for reference in references:
            contrast = contrasts.get(reference)
            if contrast is None:
                raise ComparisonReceiptError(f"grade is missing declared reference {reference}")
            reference_official = _mapping(
                contrast.get("reference_official"), f"reference {reference} official"
            )
            if reference_official.get("metric_id", METRIC_ID) != METRIC_ID:
                raise ComparisonReceiptError(f"reference {reference} metric_id is mismatched")
            reference_wp = _finite_number(reference_official.get("wp"), f"{reference} official wp")
            paired = _mapping(contrast.get("paired"), f"reference {reference} paired")
            if paired.get("metric_id", METRIC_ID) != METRIC_ID:
                raise ComparisonReceiptError(
                    f"reference {reference} paired metric_id is mismatched"
                )
            delta_value = paired.get("delta", paired.get("mean_diff"))
            delta = _finite_number(delta_value, f"reference {reference} paired delta")
            interval = paired.get("ci95")
            if not isinstance(interval, list) or len(interval) != 2:
                raise ComparisonReceiptError(f"reference {reference} lacks paired ci95")
            lower, upper = (
                _finite_number(value, f"reference {reference} paired ci95") for value in interval
            )
            if lower > upper:
                raise ComparisonReceiptError(f"reference {reference} paired ci95 is reversed")
            paired_candidate = _finite_number(
                paired.get("candidate_wp"), f"reference {reference} paired candidate wp"
            )
            paired_reference = _finite_number(
                paired.get("reference_wp"), f"reference {reference} paired reference wp"
            )
            expected_delta = candidate_wp - reference_wp
            if not (
                math.isclose(paired_candidate, candidate_wp, rel_tol=0.0, abs_tol=1e-8)
                and math.isclose(paired_reference, reference_wp, rel_tol=0.0, abs_tol=1e-8)
                and math.isclose(delta, expected_delta, rel_tol=1e-7, abs_tol=1e-8)
            ):
                raise ComparisonReceiptError(
                    f"reference {reference} paired values are inconsistent"
                )
            members = _validate_members(
                contrast.get("reference_members"),
                reference_members_expected[reference],
                split_sha256,
                f"reference {reference}",
            )
            validated_references.append(
                {
                    "reference": reference,
                    "wp": reference_wp,
                    "paired_delta": delta,
                    "official": reference_official,
                    "paired": paired,
                    "members": members,
                }
            )

        phase_timing = _phase_timing(steps)
        grade_metadata = {
            key: grade[key]
            for key in (
                "ancestry",
                "candidate_ancestry",
                "exposure",
                "fidelity",
                "limits",
                "evaluation_groups",
                "evaluation_components",
                "components",
                "selected_rows",
                "valid_targets_sha256",
                "uncertainty",
                "selected_groups",
            )
            if key in grade
        }
        hashes: dict[str, object] = {
            "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
            "comparison_sha256": _canonical_sha256(comparison),
            "grade_sha256": grade_sha256,
            "split_sha256": split_sha256,
        }
        receipt: dict[str, object] = {
            "schema": "wnn.scientific-comparison-receipt.v1",
            "status": "valid",
            "created_utc": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "experiment_id": experiment_id,
            "hypothesis_id": hypothesis_id,
            "comparison": comparison,
            "hashes": hashes,
            "metric_id": METRIC_ID,
            "expected_fold": expected_fold,
            "candidate": {
                "identity": candidate,
                "wp": candidate_wp,
                "official": official,
                "members": candidate_members,
            },
            "references": validated_references,
            "fidelity": fidelity,
            "declared_metadata": declared_metadata,
            "declared_input_hashes": input_hashes,
            "grade_metadata": grade_metadata,
            "phase_timing": phase_timing,
            "limits": [
                "Phase timings are subprocess wall time and do not measure GPU busy time.",
                "The receipt validates declared grade evidence and hashes; it does not rerun the grader.",
            ],
        }

        identity: dict[str, object] = {
            "experiment_id": experiment_id,
            "hypothesis_id": hypothesis_id,
            "candidate": candidate,
            "references": references,
            "spec_sha256": hashes["spec_sha256"],
            "comparison_sha256": hashes["comparison_sha256"],
            "grade_sha256": grade_sha256,
            "split_sha256": split_sha256,
        }
        row = identity | {
            "schema": "wnn.scientific-comparison-ledger.v1",
            "status": "valid",
            "receipt_path": str(receipt_path),
            "metric_id": METRIC_ID,
            "expected_fold": expected_fold,
            "candidate_wp": candidate_wp,
            "references_evidence": validated_references,
            "fidelity": fidelity,
            "declared_metadata": declared_metadata,
            "exposure": grade_metadata.get("exposure"),
            "phase_timing": phase_timing,
        }
        appended = _commit_receipt_and_ledger(receipt_path, receipt, ledger_path, row, identity)
        return {
            "status": "valid",
            "receipt_path": str(receipt_path),
            "ledger_path": str(ledger_path),
            "ledger_appended": appended,
            "receipt_sha256": _sha256(receipt_path),
        }
    except ComparisonReceiptError:
        raise
    except (OSError, UnicodeError, ValueError, TypeError, KeyError) as error:
        raise ComparisonReceiptError(f"comparison receipt failed: {error}") from error


def _atomic_json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temp_path = Path(stream.name)
    os.replace(temp_path, path)
