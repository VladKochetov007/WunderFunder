#!/usr/bin/env python3
"""Prove the V6 registered-leaf parent-observation delta using source bytes only."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
CAMPAIGN = PREPARATION.parent
V5 = PREPARATION / "first_cell_evaluation_score_v5"
V6 = PACKET
RECORDS = CAMPAIGN / "records/first_cell_evaluation_v1/score_v2"
FIT_ADMISSION = CAMPAIGN / "records/child_allthree_fit_v1/ADMISSION.json"
EXPORT_BINDING = PREPARATION / "first_cell_evaluation_export_v4/EXPORT_BINDING_V4.json"
SCORE_ADMISSION = RECORDS / "ADMISSION.json"
PARENT_DIAGNOSTIC = RECORDS / "alias_parent_diagnostic_v1/STDOUT.raw.json"

STAGED_MODULES = (
    "first_cell_score_core.py",
    "first_cell_score_contract.py",
    "first_cell_score_predictions.py",
    "score_first_cell.py",
)

PINNED_METADATA = {
    "fit_admission": (12912, "5312ae5feffc67d7a33051379afabb9c134dd946f1f97315d839c7d6f10d5540"),
    "export_binding_v4": (
        15796,
        "4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd",
    ),
    "score_v2_admission": (
        69650,
        "1252b2f1d39ef255fac83c722cfc1a380705963a0b2f9dede35dcabed45ba631",
    ),
    "alias_parent_diagnostic": (
        3218,
        "e6c08c71185d34c8763ec4fc983d72a91003d3626e6dc2da6a53926c49cb8380",
    ),
}

DELTA_BLOCKS = (
    (
        b"""    parent_before = parent.stat()\n+    parent_resolved_before = parent.resolve(strict=True)\n+    if not stat.S_ISDIR(parent_before.st_mode):\n+        raise ValueError(f\"registered source parent is not a directory: {relative}\")\n+    if path.resolve(strict=True) != expected_resolved:\n+""",
        b"""    parent_key = relative.split(\"/\", 1)[0]\n+    if parent_key not in EXPECTED_PARENT_RESOLUTIONS:\n+        raise ValueError(f\"registered source has an unrecognized parent alias: {relative}\")\n+    expected_parent = Path(EXPECTED_PARENT_RESOLUTIONS[parent_key])\n+    parent_before = parent.stat()\n+    if not stat.S_ISDIR(parent_before.st_mode):\n+        raise ValueError(f\"registered source parent is not a directory: {relative}\")\n+    if (\n+        parent.resolve(strict=True) != expected_parent\n+        or path.resolve(strict=True) != expected_resolved\n+    ):\n+""",
    ),
    (
        b"""        or parent_resolved_after != parent_resolved_before\n""",
        b"""        or parent_resolved_after != expected_parent\n""",
    ),
    (
        b"""            \"resolved\": str(parent_resolved_after),\n""",
        b"""            \"resolved\": str(expected_parent),\n""",
    ),
)


def _pin(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": str(path.relative_to(CAMPAIGN)),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _load_pinned_json(path: Path, expected: tuple[int, str]) -> dict[str, object]:
    raw = path.read_bytes()
    if (len(raw), hashlib.sha256(raw).hexdigest()) != expected:
        raise AssertionError(f"pinned metadata differs: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise AssertionError(f"expected a JSON object: {path}")
    return value


def _inverse_v6_core(raw: bytes) -> bytes:
    restored = raw
    for current, prior in DELTA_BLOCKS:
        # The plus signs make the checked source delta legible in this proof table.
        current = current.replace(b"\n+", b"\n")
        prior = prior.replace(b"\n+", b"\n")
        if restored.count(current) != 1:
            raise AssertionError("V6 source delta block is absent or ambiguous")
        restored = restored.replace(current, prior, 1)
    return restored


def _ast_dump(raw: bytes) -> str:
    return ast.dump(ast.parse(raw), include_attributes=False)


def build_proof() -> dict[str, object]:
    v5_core = (V5 / "first_cell_score_core.py").read_bytes()
    v6_core = (V6 / "first_cell_score_core.py").read_bytes()
    restored = _inverse_v6_core(v6_core)
    if restored != v5_core:
        raise AssertionError("reversing the declared V6 delta did not restore exact V5 bytes")
    if _ast_dump(restored) != _ast_dump(v5_core):
        raise AssertionError("restored AST differs from the V5 core AST")

    module_pins: dict[str, dict[str, object]] = {}
    for name in STAGED_MODULES:
        v5_raw = (V5 / name).read_bytes()
        v6_raw = (V6 / name).read_bytes()
        if name != "first_cell_score_core.py" and v6_raw != v5_raw:
            raise AssertionError(f"unexpected V6 change outside the core observer: {name}")
        module_pins[name] = {
            "v5": _pin(V5 / name),
            "v6": _pin(V6 / name),
            "byte_identical": v5_raw == v6_raw,
        }

    metadata_paths = {
        "fit_admission": FIT_ADMISSION,
        "export_binding_v4": EXPORT_BINDING,
        "score_v2_admission": SCORE_ADMISSION,
        "alias_parent_diagnostic": PARENT_DIAGNOSTIC,
    }
    metadata_pins = {name: _pin(path) for name, path in metadata_paths.items()}
    metadata_values = {
        name: _load_pinned_json(path, PINNED_METADATA[name])
        for name, path in metadata_paths.items()
    }

    registered = metadata_values["export_binding_v4"]["registered_data"]["source_identities"]
    fit_sources = metadata_values["fit_admission"]["data_identity_stats_only"]
    fresh_sources = metadata_values["score_v2_admission"]["checks"][
        "registered_six_sources_stat_only"
    ]["sources_stat_only"]
    diagnostic = metadata_values["alias_parent_diagnostic"]
    if registered != fit_sources or diagnostic["parents"] != diagnostic["parents_after"]:
        raise AssertionError("registered source or parent diagnostic does not join exactly")
    if set(registered) != set(fresh_sources):
        raise AssertionError("fresh score admission does not cover the six registered leaves")
    observed_fields = {"bytes", "device", "inode", "islink", "mtime_ns", "path", "resolved"}
    for relative, identity in registered.items():
        if {key: identity[key] for key in observed_fields} != fresh_sources[relative]:
            raise AssertionError(f"fresh score admission differs for {relative}")

    parents = diagnostic["parents"]
    if parents["cache"]["resolved"] == "" or parents["datasets"]["resolved"] == "":
        raise AssertionError("bounded parent diagnostic is missing resolved paths")
    if parents["cache"]["resolved"] == parents["datasets"]["resolved"]:
        raise AssertionError("actual cache and datasets parents unexpectedly resolve identically")

    return {
        "schema": "fg19.first-cell-score-v6-registered-alias-source-proof.v1",
        "status": "PASS_LOCAL_SOURCE_PROOF_NO_SCORE_EXECUTION",
        "source_delta": {
            "core_exact_inverse_restores_v5_bytes": True,
            "core_ast_inverse_restores_v5": True,
            "only_changed_staged_module": "first_cell_score_core.py",
            "changed_behavior": (
                "Retain exact registered leaf target and full leaf stat/link identity; "
                "compare fresh parent resolution and parent device/inode across the "
                "bounded observation instead of requiring an unsupported hardcoded "
                "parent target."
            ),
            "inverse_block_count": len(DELTA_BLOCKS),
        },
        "staged_modules": module_pins,
        "actual_metadata_inputs": metadata_pins,
        "actual_metadata_joins": {
            "export_binding_registered_six_sources_equals_fit_admission": True,
            "score_v2_fresh_six_leaf_stats_equal_registered_leaf_stats": True,
            "parent_diagnostic_before_equals_after": True,
            "actual_parent_resolutions": {
                name: {
                    "path": value["path"],
                    "resolved": value["resolved"],
                    "device": value["device"],
                    "inode": value["inode"],
                }
                for name, value in parents.items()
            },
            "no_historical_parent_inode_or_resolution_baseline_claimed": True,
        },
        "boundary": {
            "no_b_access": True,
            "no_checkpoint_model_or_prediction_payload_access": True,
            "no_scorer_execution": True,
            "metadata_records_were_bounded_local_json_only": True,
            "source_code_read_for_exact_inverse_and_ast_proof": True,
        },
    }


if __name__ == "__main__":
    result = build_proof()
    output = PACKET / "SCORE_V6_REGISTERED_ALIAS_PROOF.json"
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    print(
        json.dumps(
            {"path": str(output), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    )
