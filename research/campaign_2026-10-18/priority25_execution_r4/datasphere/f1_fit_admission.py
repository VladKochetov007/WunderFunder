#!/usr/bin/env python3
"""Validate finite F1 submission authority and exact input identities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from controller import (
    PROJECT_ID,
    FailClosed,
    read_single_json,
    sha256_file,
)
from f1_fit_recipe import validate_fit_spec
from f1_fit_staging import check_packet_binding, read_object

F1_JOB_NAME = "computer-vision-homework-rank01-f1-s1001-fit"
F1_OUTPUT_PATH = "fit-results.tar.gz"
F1_OUTPUT_VARIABLE = "OUT"
F1_OUTPUT_MAX_BYTES = 768 << 20
PROVIDER_GRACE_SECONDS = 60
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
UPLOADED_INPUT_FILES = (
    "f1_fit_entry.py",
    "f1_fit_bundle.py",
    "f1_fit_recipe.py",
    "f1_fit_staging.py",
    "f1_fit_outputs.py",
    "profile_staging.py",
    "profile_recipe.py",
    "requirements.txt",
    "runtime_marker.py",
)
HOST_CONTROLLER_FILES = (
    "controller.py",
    "f1_fit_admission.py",
    "f1_fit_controller.py",
    "f1_fit_verification.py",
    "f1_provider_manifest.py",
)


@dataclass(frozen=True)
class F1Admission:
    """Validated immutable identity and finite time/cost allocation."""

    sha256: str
    config_sha256: str
    profile_spec_sha256: str
    fit_spec_sha256: str
    code_archive_sha256: str
    supplemental_archive_sha256: str
    startup_reserve_seconds: int
    internal_deadline_seconds: int
    collection_reserve_seconds: int
    cancel_after_seconds: int
    external_deadline_seconds: int
    currency: str
    cost_limit_minor_units: int
    estimated_max_cost_minor_units: int
    output_max_bytes: int


def _positive_int(value: object, label: str) -> int:
    """Validate an explicit finite positive integer budget."""
    if type(value) is not int or value <= 0:
        raise FailClosed(f"admission {label} must be a finite positive integer")
    return value


def _sha256(value: object, label: str) -> str:
    """Validate a lowercase SHA-256 string."""
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise FailClosed(f"admission {label} must be a lowercase SHA-256")
    return value


def _validate_cost_budget(cost: dict[str, object]) -> tuple[str, int, int]:
    """Require a finite cost allocation beneath the sealed estimate and evidence."""
    currency = cost.get("currency")
    if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
        raise FailClosed("cost budget currency must be a three-letter uppercase code")
    limit = _positive_int(cost.get("max_cost_minor_units"), "maximum cost")
    estimate = _positive_int(cost.get("estimated_max_cost_minor_units"), "estimated maximum cost")
    if estimate > limit:
        raise FailClosed("estimated maximum cost exceeds the explicit cost allocation")
    _sha256(cost.get("balance_and_rate_evidence_sha256"), "balance/rate evidence hash")
    return currency, limit, estimate


def _validate_runtime_budget(
    runtime: dict[str, object], fit_deadlines: tuple[int, int, int]
) -> tuple[int, int, int, int, int]:
    """Require startup, fit, collection, cancellation and provider limits to fit together."""
    startup = _positive_int(runtime.get("submit_to_execution_reserve_seconds"), "startup reserve")
    internal = _positive_int(runtime.get("internal_deadline_seconds"), "internal deadline")
    collection = _positive_int(runtime.get("collection_reserve_seconds"), "collection reserve")
    cancel_after = _positive_int(runtime.get("cancel_after_seconds_from_submit"), "cancel deadline")
    external = _positive_int(runtime.get("external_deadline_seconds_from_submit"), "hard stop")
    if (internal, collection, external) != fit_deadlines:
        raise FailClosed("admission runtime does not match the sealed F1 fit specification")
    if startup + internal + collection + PROVIDER_GRACE_SECONDS > cancel_after:
        raise FailClosed("startup, fit, collection, and provider grace exceed the cancel deadline")
    if cancel_after + PROVIDER_GRACE_SECONDS > external:
        raise FailClosed("cancel plus provider graceful shutdown exceeds the hard-stop deadline")
    return startup, internal, collection, cancel_after, external


def _validate_hash_map(
    directory: Path, value: object, filenames: tuple[str, ...], label: str
) -> None:
    """Require exact, current hashes for every submitted or host-executed module."""
    if not isinstance(value, dict) or set(value) != set(filenames):
        raise FailClosed(f"admission {label} hash map is missing, extra, or malformed")
    for filename in filenames:
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise FailClosed(f"sealed {label} file is missing or not regular: {filename}")
        if sha256_file(path) != _sha256(value.get(filename), f"{label} {filename} hash"):
            raise FailClosed(f"sealed {label} file changed: {filename}")


def validate_f1_config(path: Path, expected_sha256: str) -> str:
    """Require the fixed F1 provider name, g2.1 type, and sealed config bytes."""
    if path.is_symlink() or not path.is_file():
        raise FailClosed("private F1 config is missing, non-regular, or a symlink")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise FailClosed("F1 config hash differs from the admission envelope")
    content = path.read_text(encoding="utf-8")
    names = re.findall(r"^name:\s*(\S+)\s*$", content, flags=re.MULTILINE)
    gpus = re.findall(r"^cloud-instance-type:\s*(\S+)\s*$", content, flags=re.MULTILINE)
    if names != [F1_JOB_NAME] or gpus != ["g2.1"]:
        raise FailClosed("F1 config name or GPU type differs from the fixed admission")
    if f"{F1_OUTPUT_PATH}: {F1_OUTPUT_VARIABLE}" not in content:
        raise FailClosed("F1 config does not declare the exact fit bundle output")
    return actual


def validate_admission(
    admission_path: Path,
    config_path: Path,
    profile_spec_path: Path,
    fit_spec_path: Path,
    code_archive_path: Path,
    supplemental_archive_path: Path,
) -> F1Admission:
    """Check sealed inputs and prove all startup, fit, collection, and cost bounds."""
    for path in (
        admission_path,
        config_path,
        profile_spec_path,
        fit_spec_path,
        code_archive_path,
        supplemental_archive_path,
    ):
        if path.is_symlink() or not path.is_file():
            raise FailClosed(f"admission input is missing or not a regular file: {path.name}")
    raw = read_single_json(admission_path, "F1 admission envelope")
    if raw.get("schema_version") != 1 or raw.get("state") != "SEALED_FOR_F1_FIT":
        raise FailClosed("F1 admission envelope is not sealed")
    if raw.get("project_id") != PROJECT_ID or raw.get("job_name") != F1_JOB_NAME:
        raise FailClosed("admission project or immutable F1 job name differs")
    directory = Path(__file__).parent
    _validate_hash_map(
        directory, raw.get("uploaded_input_sha256"), UPLOADED_INPUT_FILES, "uploaded input"
    )
    _validate_hash_map(
        directory, raw.get("host_controller_sha256"), HOST_CONTROLLER_FILES, "host controller"
    )
    hashes = {
        key: _sha256(raw.get(key), key)
        for key in (
            "config_sha256",
            "profile_spec_sha256",
            "fit_spec_sha256",
            "code_archive_sha256",
            "supplemental_archive_sha256",
        )
    }
    actuals = {
        "config_sha256": validate_f1_config(config_path, hashes["config_sha256"]),
        "profile_spec_sha256": sha256_file(profile_spec_path),
        "fit_spec_sha256": sha256_file(fit_spec_path),
        "code_archive_sha256": sha256_file(code_archive_path),
        "supplemental_archive_sha256": sha256_file(supplemental_archive_path),
    }
    if hashes != actuals:
        raise FailClosed("one or more F1 packet files differ from the admission hashes")
    profile = read_object(profile_spec_path, "profile specification")
    fit = read_object(fit_spec_path, "fit specification")
    check_packet_binding(profile, fit, profile_spec_path)
    try:
        fit_deadlines = validate_fit_spec(fit)
    except (TypeError, ValueError) as error:
        raise FailClosed("fit specification is not sealed or its runtime is invalid") from error
    if fit.get("code_archive_sha256") != hashes["code_archive_sha256"]:
        raise FailClosed("fit specification does not bind to the admitted source archive")
    if fit.get("supplemental_archive_sha256") != hashes["supplemental_archive_sha256"]:
        raise FailClosed("fit specification does not bind to the admitted supplemental archive")

    runtime = raw.get("runtime_budget")
    cost = raw.get("cost_budget")
    output = raw.get("output")
    if not isinstance(runtime, dict) or not isinstance(cost, dict) or not isinstance(output, dict):
        raise FailClosed("admission lacks finite runtime, cost, or output bounds")
    startup, internal, collection, cancel_after, external = _validate_runtime_budget(
        runtime, fit_deadlines
    )
    currency, cost_limit, estimated_cost = _validate_cost_budget(cost)
    output_max = _positive_int(output.get("max_bytes"), "output size limit")
    if output.get("path") != F1_OUTPUT_PATH or output.get("variable") != F1_OUTPUT_VARIABLE:
        raise FailClosed("admission output identity differs from the sealed F1 bundle")
    if output_max > F1_OUTPUT_MAX_BYTES:
        raise FailClosed("F1 bundle allowance exceeds the reserved DataSphere aggregate cap")

    return F1Admission(
        sha256=sha256_file(admission_path),
        config_sha256=hashes["config_sha256"],
        profile_spec_sha256=hashes["profile_spec_sha256"],
        fit_spec_sha256=hashes["fit_spec_sha256"],
        code_archive_sha256=hashes["code_archive_sha256"],
        supplemental_archive_sha256=hashes["supplemental_archive_sha256"],
        startup_reserve_seconds=startup,
        internal_deadline_seconds=internal,
        collection_reserve_seconds=collection,
        cancel_after_seconds=cancel_after,
        external_deadline_seconds=external,
        currency=currency,
        cost_limit_minor_units=cost_limit,
        estimated_max_cost_minor_units=estimated_cost,
        output_max_bytes=output_max,
    )
