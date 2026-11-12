"""Admit receiver checks and human-approved profiles for a Signal Routes fit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from signal_routes.bindings import VARIANTS, read_object
from signal_routes.profile_approval import validate_approval


def validate_checks(
    path: Path,
    source: dict[str, object],
    preprocessing_sha256: str,
    precision: str,
) -> dict[str, Any]:
    """Require one server receipt covering every declared variant and shared source."""
    checks = read_object(path)
    if (
        checks.get("schema") != "signal-routes.receiver-checks.v1"
        or checks.get("status") != "passed"
        or checks.get("mechanism_contract") != "signal-routes.mechanisms.v1"
        or checks.get("checked_variants") != sorted(VARIANTS)
        or checks.get("source_identity") != source
        or checks.get("preprocessing_sha256") != preprocessing_sha256
        or checks.get("precision_mode") != precision
    ):
        raise ValueError("receiver check receipt differs from the requested fit identity")
    return checks


def admit_profile(
    path: Path, identity: dict[str, object], cell: dict[str, Any], config: dict[str, Any]
) -> str:
    """Require the human approval and canonical 50-update profile for this cell."""
    receipt = read_object(path)
    receipt_identity = receipt.get("identity")
    if not isinstance(receipt_identity, dict):
        raise ValueError("approved profile lacks its human review or canonical identity")
    validate_approval(receipt, identity, cell)
    if (
        receipt.get("updates") != 50
        or receipt.get("approved") is not True
        or receipt_identity.get("signal_routes_cell") != cell
        or config.get("signal_routes", {}).get("cell") != cell
    ):
        raise ValueError(
            "profile receipt is not the approved source-matched 50-update cell profile"
        )
    from wnn.training.sequence_provenance import validate_profile_receipt

    return validate_profile_receipt(path, identity)
