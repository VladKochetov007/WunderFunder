"""Admit receiver checks and human-approved profiles for a Bilinear Entmax fit."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from bilinear_entmax.bindings import (
    FULL_SHAPE,
    FULL_SHAPE_ACCEPTANCE_SCHEMA,
    PENDING_FULL_SHAPE_ACCEPTANCE,
    VARIANTS,
    canonical,
    digest,
    read_object,
)
from bilinear_entmax.bindings import (
    source_profile_sha256 as canonical_source_profile_sha256,
)
from bilinear_entmax.profile_approval import validate_approval


def _number(value: object, label: str) -> float:
    """Read a finite receipt number while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def validate_checks(
    path: Path,
    source: dict[str, object],
    preprocessing_sha256: str,
    precision: str,
) -> dict[str, Any]:
    """Require one server receipt covering every declared variant and shared source."""
    checks = read_object(path)
    if (
        checks.get("schema") != "bilinear-entmax.receiver-checks.v1"
        or checks.get("status") != "passed"
        or checks.get("mechanism_contract") != "bilinear-entmax.mechanisms.v1"
        or checks.get("checked_variants") != sorted(VARIANTS)
        or checks.get("source_identity") != source
        or checks.get("preprocessing_sha256") != preprocessing_sha256
        or checks.get("precision_mode") != precision
        or checks.get("full_shape_acceptance") != PENDING_FULL_SHAPE_ACCEPTANCE
    ):
        raise ValueError("receiver check receipt differs from the requested fit identity")
    return checks


def admit_full_shape_identity(
    identity: dict[str, Any],
    provenance: dict[str, Any],
    cell: dict[str, Any],
    path: Path,
) -> None:
    """Require the fit to bind its approved, resource-checked per-cell profile."""
    profile_receipt = provenance.get("profile_receipt")
    profile_sha256 = identity.get("bilinear_entmax_profile_receipt_sha256")
    acceptance = identity.get("bilinear_entmax_full_shape_acceptance")
    acceptance_sha256 = identity.get("bilinear_entmax_full_shape_acceptance_sha256")
    if (
        not isinstance(profile_receipt, dict)
        or profile_receipt.get("sha256") != profile_sha256
        or not isinstance(profile_sha256, str)
        or len(profile_sha256) != 64
        or any(character not in "0123456789abcdef" for character in profile_sha256)
        or not isinstance(acceptance, dict)
        or acceptance.get("schema") != FULL_SHAPE_ACCEPTANCE_SCHEMA
        or acceptance.get("status") != "passed"
        or acceptance.get("required_before_fit") is not True
        or acceptance.get("human_approval_required") is not True
        or acceptance.get("human_approved") is not True
        or not isinstance(acceptance.get("reviewer"), str)
        or not acceptance["reviewer"].strip()
        or acceptance.get("profile_updates") != 50
        or acceptance.get("cell") != cell
        or acceptance.get("source_sha256") != identity.get("source_sha256")
        or acceptance.get("requested_shape") != FULL_SHAPE
        or acceptance.get("observed_shape") != FULL_SHAPE
        or acceptance_sha256 != canonical(acceptance)
    ):
        raise ValueError(f"fit lacks its source-matched approved full-shape profile: {path}")
    limits = acceptance.get("resource_limits")
    observed = acceptance.get("observed_resources")
    if not isinstance(limits, dict) or not isinstance(observed, dict):
        raise ValueError(f"full-shape acceptance omits resource measurements: {path}")
    host_limit = _number(limits.get("max_host_ram_gib"), "max_host_ram_gib")
    vram_limit = _number(limits.get("max_vram_gib_per_rank"), "max_vram_gib_per_rank")
    host_by_rank = observed.get("host_peak_rss_bytes")
    if not isinstance(host_by_rank, list) or len(host_by_rank) != 2:
        raise ValueError(f"full-shape profile must record host RSS for both ranks: {path}")
    host_rss = [_number(value, "rank host peak RSS") for value in host_by_rank]
    if any(value <= 0 for value in host_rss):
        raise ValueError(f"full-shape profile contains invalid host RSS: {path}")
    host_peak = sum(host_rss)
    reserved = observed.get("peak_reserved_by_rank_bytes")
    totals = observed.get("device_total_memory_by_rank_bytes")
    if (
        host_limit <= 0
        or vram_limit <= 0
        or not isinstance(reserved, list)
        or len(reserved) != 2
        or not isinstance(totals, list)
        or len(totals) != 2
        or host_peak <= 0
        or host_peak > host_limit * (1024**3)
    ):
        raise ValueError(f"full-shape profile exceeded or omitted its resource caps: {path}")
    for rank, (reserved_bytes, total_bytes) in enumerate(zip(reserved, totals, strict=True)):
        reserved_value = _number(reserved_bytes, f"rank_{rank}_reserved_bytes")
        total_value = _number(total_bytes, f"rank_{rank}_device_total_bytes")
        if (
            reserved_value <= 0
            or total_value <= 0
            or reserved_value > total_value
            or reserved_value > vram_limit * (1024**3)
        ):
            raise ValueError(f"rank {rank} full-shape profile exceeded its resource cap: {path}")


def validate_profile_gate(
    path: Path, source: dict[str, object], cell: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Reject a fit before model construction unless its approved full-shape profile is sound."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("full fit requires a regular approved full-shape profile receipt")
    receipt = read_object(path)
    identity = receipt.get("identity")
    acceptance = receipt.get("full_shape_acceptance")
    if not isinstance(identity, dict) or not isinstance(acceptance, dict):
        raise ValueError("full fit requires source-matched profile identity and acceptance")
    validate_approval(receipt, identity, cell)
    if (
        receipt.get("receipt_schema") != "wnn.sequence-profile.v1"
        or receipt.get("status") != "profile_complete"
        or receipt.get("approved") is not True
        or receipt.get("updates") != 50
        or receipt.get("scientific_checkpoint_written") is not False
        or receipt.get("errors") != []
        or receipt.get("error_count") != 0
        or identity.get("bilinear_entmax_cell") != cell
        or identity.get("source_sha256") != source.get("sha256")
        or identity.get("source_files") != source.get("files")
    ):
        raise ValueError("approved full-shape profile identity differs from current runtime")
    source_profile_hash = acceptance.get("source_profile_sha256")
    if source_profile_hash != canonical_source_profile_sha256(receipt):
        raise ValueError("full-shape acceptance is not bound to the canonical PR75 profile")
    profile_hash = digest(path)
    admitted_identity = dict(identity)
    admitted_identity["bilinear_entmax_profile_receipt_sha256"] = profile_hash
    admitted_identity["bilinear_entmax_full_shape_acceptance"] = acceptance
    admitted_identity["bilinear_entmax_full_shape_acceptance_sha256"] = canonical(acceptance)
    admit_full_shape_identity(
        admitted_identity,
        {"profile_receipt": {"sha256": profile_hash}},
        cell,
        path,
    )
    return profile_hash, acceptance


def admit_profile(
    path: Path, identity: dict[str, object], cell: dict[str, Any], config: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Require the human approval and canonical 50-update profile for this cell."""
    receipt = read_object(path)
    receipt_identity = receipt.get("identity")
    if not isinstance(receipt_identity, dict):
        raise ValueError("approved profile lacks its human review or canonical identity")
    validate_approval(receipt, identity, cell)
    if (
        receipt.get("updates") != 50
        or receipt.get("approved") is not True
        or receipt_identity.get("bilinear_entmax_cell") != cell
        or config.get("bilinear_entmax", {}).get("cell") != cell
    ):
        raise ValueError(
            "profile receipt is not the approved source-matched 50-update cell profile"
        )
    approval = receipt.get("approval")
    acceptance = receipt.get("full_shape_acceptance")
    if not isinstance(approval, dict) or not isinstance(acceptance, dict):
        raise ValueError("approved profile lacks human-reviewed full-shape acceptance")
    source_profile_hash = acceptance.get("source_profile_sha256")
    if (
        not isinstance(source_profile_hash, str)
        or len(source_profile_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_profile_hash)
        or source_profile_hash != canonical_source_profile_sha256(receipt)
    ):
        raise ValueError("full-shape acceptance lacks the measured profile source digest")
    memory = receipt.get("memory_by_rank")
    if (
        not isinstance(memory, list)
        or len(memory) != 2
        or any(not isinstance(item, dict) for item in memory)
    ):
        raise ValueError("approved profile omits the two-rank resource measurements")
    expected_acceptance = {
        "schema": FULL_SHAPE_ACCEPTANCE_SCHEMA,
        "status": "passed",
        "required_before_fit": True,
        "human_approval_required": True,
        "human_approved": True,
        "reviewer": approval.get("reviewer"),
        "profile_updates": 50,
        "source_profile_sha256": source_profile_hash,
        "source_sha256": identity.get("source_sha256"),
        "cell": cell,
        "requested_shape": FULL_SHAPE,
        "observed_shape": FULL_SHAPE,
        "resource_limits": {
            "max_host_ram_gib": approval.get("max_host_ram_gib"),
            "max_vram_gib_per_rank": approval.get("max_vram_gib_per_rank"),
        },
        "observed_resources": {
            "host_peak_rss_bytes": receipt.get("host_peak_rss_bytes"),
            "peak_reserved_by_rank_bytes": [item.get("peak_reserved_bytes") for item in memory]
            if isinstance(memory, list)
            else None,
            "device_total_memory_by_rank_bytes": receipt.get("device_total_memory_bytes"),
        },
    }
    if acceptance != expected_acceptance:
        raise ValueError("approved profile does not carry complete full-shape acceptance")
    from wnn.training.sequence_provenance import validate_profile_receipt

    return validate_profile_receipt(path, identity), acceptance
