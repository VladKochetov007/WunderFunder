"""Validate the reviewer block attached to a canonical profile receipt."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def validate_approval(
    receipt: dict[str, Any], identity: dict[str, object], cell: dict[str, Any]
) -> None:
    """Bind a human approval to the untouched profile bytes, fit identity, and cell."""
    approval = receipt.get("approval")
    receipt_identity = receipt.get("identity")
    if not isinstance(approval, dict) or not isinstance(receipt_identity, dict):
        raise ValueError("approved profile lacks its human review or canonical identity")
    fields = {
        "schema",
        "reviewer",
        "note",
        "profile_sha256",
        "source_sha256",
        "cell",
        "approved_at_utc",
        "max_host_ram_gib",
        "max_vram_gib_per_rank",
    }
    original = {key: value for key, value in receipt.items() if key != "approval"}
    original["approved"] = False
    encoded = (
        json.dumps(original, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    expected_sha256 = hashlib.sha256(encoded).hexdigest()
    for field in ("max_host_ram_gib", "max_vram_gib_per_rank"):
        value = approval.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"profile approval {field} must be finite and positive")
    if (
        set(approval) != fields
        or approval.get("schema") != "signal-routes.profile-approval.v1"
        or not isinstance(approval.get("reviewer"), str)
        or not approval["reviewer"].strip()
        or not isinstance(approval.get("note"), str)
        or not approval["note"].strip()
        or not isinstance(approval.get("approved_at_utc"), str)
        or not approval["approved_at_utc"].strip()
        or approval.get("profile_sha256") != expected_sha256
        or approval.get("source_sha256") != identity.get("source_sha256")
        or approval.get("cell") != cell
        or receipt_identity != identity
    ):
        raise ValueError("profile approval does not bind the original profile and fit identity")
