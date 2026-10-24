"""Validate the exact receiving-host Muon parity receipt contract."""

from __future__ import annotations

import math
from typing import Any


def validate_muon_receipt(value: dict[str, Any]) -> None:
    """Reject partial or inconsistent parity records before binding their hashes."""
    expected = {
        "passed": True,
        "world_size": 2,
        "explicit_updates": 2,
        "schedule_total_updates": 4,
        "precision": "strict_fp32_no_tf32_no_autocast",
        "tolerance": {"atol": 1e-5, "rtol": 1e-4, "denominator_atol": 1e-5},
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError("Muon DDP receipt has incorrect update/precision/tolerance metadata")
    cases = value.get("cases", [])
    if len(cases) != 2:
        raise ValueError("Muon DDP receipt must contain both declared cases")
    checks = {
        "gradients",
        "parameters",
        "optimizer_state",
        "scheduler",
        "semantic_groups",
        "reference_groups",
    }
    roles = {"muon_qkv", "muon_hidden", "adamw_matrix", "adamw_no_decay"}
    errors = {"denominator", "gradients", "parameters", "optimizer_state"}
    for update, (case, scenario) in enumerate(zip(cases, ("uneven", "empty_rank"), strict=True), 1):
        if case.get("scenario") != scenario or case.get("update") != update:
            raise ValueError("Muon DDP receipt has incorrect case identities")
        ranks = case.get("ranks", [])
        if case.get("passed") is not True or [row.get("rank") for row in ranks] != [0, 1]:
            raise ValueError("Muon DDP case lacks passing results from both ranks")
        for rank, row in enumerate(ranks):
            expected_rows = 8 if rank == 0 else (2 if scenario == "uneven" else 0)
            actual_checks = row.get("checks", {})
            actual_errors = row.get("errors", {})
            if (
                row.get("passed") is not True
                or row.get("valid_rows") != expected_rows
                or set(row.get("semantic_roles", [])) != roles
                or set(actual_checks) != checks
                or any(item is not True for item in actual_checks.values())
                or set(actual_errors) != errors
                or not all(math.isfinite(item) and item >= 0 for item in actual_errors.values())
            ):
                raise ValueError("Muon DDP rank lacks complete passing comparisons")
