"""Prove the V2 exporter packet differs from frozen V1 only at its fixed pins."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

V1_BINDING_NAME = "EXPORT_BINDING_V1.json"
V1_SOURCE_NAME = "export_first_cell_v1.py"
MANIFEST_NAME = "EXPORT_SOURCE_MANIFEST.json"
OLD_BINDING_BYTES = 15_822
OLD_BINDING_SHA256 = "fcb3ee7934a19d023f29090e527be1d151dde23ae9fae3e0e3a911cea1bd809c"
OLD_LOADER_PATH = "preparation/parent_fit_terminal_check_v3.py"
OLD_SUPPORT_PATH = "preparation/propensity_terminal_support.py"
NEW_LOADER_PATH = "parent_fit_terminal_check_v3.py"
NEW_SUPPORT_PATH = "propensity_terminal_support.py"
DIAGNOSTIC_RELATIVE_PATH = (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/records/first_cell_evaluation_v1/"
    "export_v1/preflight_outcome_v1/STDOUT.raw.json"
)
DIAGNOSTIC_BYTES = 1_509
DIAGNOSTIC_SHA256 = "40a55d9e1c2d251627e6208775c0736699e3f1aa0378cdbc2ba7b126acd126c2"


def _pin(path: Path) -> dict[str, object]:
    """Return only the byte count and digest of one source/metadata file."""
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def build_proof() -> dict[str, object]:
    """Check exact inverse transforms against the frozen V1 packet."""
    packet = Path(__file__).resolve().parent
    root = packet.parents[4]
    previous = root / "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation"
    previous /= "first_cell_evaluation_export_v1"
    binding_v1 = (previous / V1_BINDING_NAME).read_bytes()
    binding_v2 = (packet / V1_BINDING_NAME).read_bytes()
    binding_inverse = binding_v2.replace(
        NEW_LOADER_PATH.encode(), OLD_LOADER_PATH.encode(), 1
    ).replace(NEW_SUPPORT_PATH.encode(), OLD_SUPPORT_PATH.encode(), 1)
    if binding_v2.count(NEW_LOADER_PATH.encode()) != 1:
        raise ValueError("V2 loader relative path must occur exactly once")
    if binding_v2.count(NEW_SUPPORT_PATH.encode()) != 1:
        raise ValueError("V2 support relative path must occur exactly once")
    if binding_inverse != binding_v1:
        raise ValueError("binding inverse does not reproduce exact frozen V1 bytes")

    source_v1 = (previous / V1_SOURCE_NAME).read_bytes()
    source_v2 = (packet / V1_SOURCE_NAME).read_bytes()
    new_bytes_line = f"BINDING_BYTES = {len(binding_v2):,}".replace(",", "_").encode()
    old_bytes_line = f"BINDING_BYTES = {OLD_BINDING_BYTES:,}".replace(",", "_").encode()
    new_sha_line = f'BINDING_SHA256 = "{hashlib.sha256(binding_v2).hexdigest()}"'.encode()
    old_sha_line = f'BINDING_SHA256 = "{OLD_BINDING_SHA256}"'.encode()
    if source_v2.count(new_bytes_line) != 1 or source_v2.count(new_sha_line) != 1:
        raise ValueError("V2 exporter must contain exactly one updated binding pin pair")
    source_inverse = source_v2.replace(new_bytes_line, old_bytes_line, 1).replace(
        new_sha_line, old_sha_line, 1
    )
    if source_inverse != source_v1:
        raise ValueError("exporter inverse does not reproduce exact frozen V1 bytes")

    manifest_v1 = (previous / MANIFEST_NAME).read_bytes()
    manifest_v2 = (packet / MANIFEST_NAME).read_bytes()
    if manifest_v2 != manifest_v1:
        raise ValueError("167-file source manifest changed between V1 and V2")

    diagnostic = root / DIAGNOSTIC_RELATIVE_PATH
    diagnostic_pin = _pin(diagnostic)
    if diagnostic_pin != {"bytes": DIAGNOSTIC_BYTES, "sha256": DIAGNOSTIC_SHA256}:
        raise ValueError("captured CPU2 flat-support diagnostic pin differs")

    return {
        "schema": "fg19.first-cell-export-v2-inverse-byte-proof.v1",
        "status": "PASS_EXACT_TWO_BINDING_PATH_AND_TWO_PIN_CONSTANT_INVERSE",
        "scope": "Source and metadata byte comparison only; no imports, tests, B action, or payload reads.",
        "frozen_v1": {
            "binding": {
                "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
                "first_cell_evaluation_export_v1/EXPORT_BINDING_V1.json",
                **_pin(previous / V1_BINDING_NAME),
            },
            "exporter": {
                "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
                "first_cell_evaluation_export_v1/export_first_cell_v1.py",
                **_pin(previous / V1_SOURCE_NAME),
            },
        },
        "v2": {
            "binding": {
                "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
                "first_cell_evaluation_export_v2/EXPORT_BINDING_V1.json",
                **_pin(packet / V1_BINDING_NAME),
            },
            "exporter": {
                "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
                "first_cell_evaluation_export_v2/export_first_cell_v1.py",
                **_pin(packet / V1_SOURCE_NAME),
            },
        },
        "binding_only_changes": [
            {"from": OLD_LOADER_PATH, "to": NEW_LOADER_PATH},
            {"from": OLD_SUPPORT_PATH, "to": NEW_SUPPORT_PATH},
        ],
        "exporter_only_changes": {
            "binding_bytes": {"from": OLD_BINDING_BYTES, "to": len(binding_v2)},
            "binding_sha256": {
                "from": OLD_BINDING_SHA256,
                "to": hashlib.sha256(binding_v2).hexdigest(),
            },
        },
        "source_manifest": {
            "identical_to_frozen_v1": True,
            "bytes": len(manifest_v2),
            "sha256": hashlib.sha256(manifest_v2).hexdigest(),
        },
        "cpu2_flat_support_diagnostic": {
            "path": DIAGNOSTIC_RELATIVE_PATH,
            **diagnostic_pin,
        },
        "inverse_checks": {
            "two_binding_path_replacements_reproduce_exact_v1_bytes": True,
            "two_exporter_pin_constant_replacements_reproduce_exact_v1_bytes": True,
            "source_manifest_bytes_identical": True,
        },
    }


def main() -> int:
    """Write the deterministic proof receipt beside this script."""
    output = Path(__file__).resolve().with_name("EXPORT_V2_INVERSE_PROOF.json")
    raw = (json.dumps(build_proof(), sort_keys=True, indent=2) + "\n").encode()
    if output.exists():
        raise FileExistsError(f"proof receipt already exists: {output}")
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    print(f"{len(raw)} {hashlib.sha256(raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
