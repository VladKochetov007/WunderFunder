"""Render the first-cell export admission stdin body from five fixed local payloads."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[5]
DIRECTORY = Path(__file__).resolve().parent
EXPORT_DIRECTORY = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v1"
)
EXPORT_REVIEW = EXPORT_DIRECTORY / "EXPORT_SOURCE_REVIEW_V1.json"
RECORD_DIRECTORY = (
    REPOSITORY / "research/campaign_2026-10-19/full_gru_tweaks_r1/records/child_allthree_fit_v1"
)
HELPERS = DIRECTORY / "V5_STDLIB_HELPERS_EXTRACTED.py"
HELPER_PROVENANCE = DIRECTORY / "V5_STDLIB_HELPER_EXTRACTION_V1.json"
BODY = DIRECTORY / "ADMISSION_BODY_TEMPLATE_V1.py"
OUTPUT = DIRECTORY / "REMOTE_SUBMISSION_TEMPLATE.py"
OUTPUT_BINDING = DIRECTORY / "ADMISSION_SOURCE_BINDING_V1.json"

INPUT_PINS = {
    "export_first_cell_v1.py": (
        EXPORT_DIRECTORY / "export_first_cell_v1.py",
        64_315,
        "5e34aa96762c4bd9db934df6b87666f37b80bc8e1ef7e20e6a74a097d33beb00",
    ),
    "EXPORT_BINDING_V1.json": (
        EXPORT_DIRECTORY / "EXPORT_BINDING_V1.json",
        15_822,
        "fcb3ee7934a19d023f29090e527be1d151dde23ae9fae3e0e3a911cea1bd809c",
    ),
    "EXPORT_SOURCE_MANIFEST.json": (
        EXPORT_DIRECTORY / "EXPORT_SOURCE_MANIFEST.json",
        27_569,
        "a44f8a3fd3fda1efdda578c8694036dffaf72ce6999e4e9095f6298365b35621",
    ),
    "ALLTHREE_TERMINAL_ACCEPTANCE.json": (
        RECORD_DIRECTORY / "TERMINAL_ACCEPTANCE.json",
        20_208,
        "fada500ede63a175d7d055d38fbef5d38110aef481a91771ae3c5717c1e7026f",
    ),
    "ALLTHREE_EXACT_STOP.json": (
        RECORD_DIRECTORY / "exact_stop/EXACT_STOP.json",
        3_208,
        "0154b3f35c4fb849d0db7e7b60ce1f4ce2fa01ab2203c17db65877f89353f13d",
    ),
}
V5_SOURCE_PIN = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/child_allthree_terminal_verifier_v5/REMOTE_SUBMISSION_V5.py",
    "bytes": 710_848,
    "sha256": "4d555fd69954d94415c135e467eadd081aa6fd915686197e5c7e628e935b9374",
}
EXPORT_REVIEW_PIN = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_export_v1/EXPORT_SOURCE_REVIEW_V1.json",
    "bytes": 9_633,
    "sha256": "cf44af7624e81ee99e9705bf08e69b359f7048354df069cad35b26c7ec927c15",
    "schema": "fg19.first-cell-export-source-review.v1",
    "status": "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR",
}
PLACEHOLDER = 'PAYLOAD_B64: dict[str, str] = {"__UNBOUND_PAYLOADS__": "__UNBOUND_PAYLOADS__"}'


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def render() -> tuple[bytes, dict[str, object]]:
    """Verify every fixed input and render a stdin-safe body without source self-reads."""
    helpers = HELPERS.read_bytes()
    body = BODY.read_bytes()
    helper_provenance_raw = HELPER_PROVENANCE.read_bytes()
    helper_provenance = json.loads(helper_provenance_raw)
    if not isinstance(helper_provenance, dict):
        raise ValueError("helper provenance must be a JSON object")
    if helper_provenance.get("source") != V5_SOURCE_PIN:
        raise ValueError("stdlib helper extraction no longer binds the exact frozen V5 body")
    if helper_provenance.get("output") != {
        "path": str(HELPERS.relative_to(REPOSITORY)),
        "bytes": len(helpers),
        "sha256": sha256(helpers),
    }:
        raise ValueError("extracted helper bytes differ from their provenance")
    body_text = body.decode("utf-8", errors="strict")
    if body_text.count(PLACEHOLDER) != 1:
        raise ValueError("admission body must contain exactly one payload placeholder")

    export_review_raw = EXPORT_REVIEW.read_bytes()
    if (len(export_review_raw), sha256(export_review_raw)) != (
        EXPORT_REVIEW_PIN["bytes"],
        EXPORT_REVIEW_PIN["sha256"],
    ):
        raise ValueError("fixed exporter producer-consumer source review pin differs")
    export_review = json.loads(export_review_raw)
    if not isinstance(export_review, dict):
        raise ValueError("exporter source review must be a JSON object")
    if (export_review.get("schema"), export_review.get("status")) != (
        EXPORT_REVIEW_PIN["schema"],
        EXPORT_REVIEW_PIN["status"],
    ):
        raise ValueError("exporter source review is not the expected passing review")

    payloads: dict[str, bytes] = {}
    payload_pins: dict[str, dict[str, object]] = {}
    encoded_rows: list[str] = []
    for name, (path, expected_bytes, expected_sha) in INPUT_PINS.items():
        raw = path.read_bytes()
        actual = (len(raw), sha256(raw))
        if actual != (expected_bytes, expected_sha):
            raise ValueError(f"fixed payload pin differs: {name}: {actual}")
        payloads[name] = raw
        encoded = base64.b64encode(raw).decode("ascii")
        encoded_rows.append(f"    {json.dumps(name)}: {json.dumps(encoded)},")
        payload_pins[name] = {
            "path": str(path.relative_to(REPOSITORY)),
            "bytes": len(raw),
            "sha256": sha256(raw),
        }

    reviewed_producer = export_review.get("producer")
    if not isinstance(reviewed_producer, dict):
        raise ValueError("exporter review must include its reviewed producer")
    expected_reviewed_files = {
        "export_first_cell_v1.py": reviewed_producer,
        "EXPORT_BINDING_V1.json": reviewed_producer.get("binding"),
    }
    for name, reviewed_file in expected_reviewed_files.items():
        if not isinstance(reviewed_file, dict):
            raise ValueError(f"exporter review is missing reviewed input {name}")
        expected_path, expected_bytes, expected_sha = INPUT_PINS[name]
        if (
            reviewed_file.get("path") != str(expected_path.relative_to(REPOSITORY))
            or reviewed_file.get("bytes") != expected_bytes
            or reviewed_file.get("sha256") != expected_sha
        ):
            raise ValueError(f"exporter review's producer input pin differs: {name}")
    reviewed_manifest = export_review.get("scope_authority", {}).get("source_manifest")
    manifest_path, manifest_bytes, manifest_sha = INPUT_PINS["EXPORT_SOURCE_MANIFEST.json"]
    if not isinstance(reviewed_manifest, dict) or (
        reviewed_manifest.get("path") != str(manifest_path.relative_to(REPOSITORY))
        or reviewed_manifest.get("bytes") != manifest_bytes
        or reviewed_manifest.get("sha256") != manifest_sha
    ):
        raise ValueError("exporter review's source-manifest input pin differs")

    payload_map = "PAYLOAD_B64: dict[str, str] = {\n" + "\n".join(encoded_rows) + "\n}"
    rendered_body = body_text.replace(PLACEHOLDER, payload_map, 1)
    rendered = (helpers.decode("utf-8", errors="strict") + "\n\n" + rendered_body).encode("utf-8")
    tree = ast.parse(rendered.decode("utf-8", errors="strict"), filename=str(OUTPUT))
    if any(isinstance(node, ast.Name) and node.id == "__file__" for node in ast.walk(tree)):
        raise ValueError("remote stdin body must not inspect its own filename")
    if sum(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body) != 1:
        raise ValueError("remote body must have exactly one main entrypoint")
    rendered_pins = {
        "bytes": len(rendered),
        "sha256": sha256(rendered),
        "path": str(OUTPUT.relative_to(REPOSITORY)),
    }
    binding: dict[str, object] = {
        "schema": "fg19.first-cell-export-admission-source-binding.v1",
        "scope": "Local static binding of a one-shot remote admission stdin body and its five fixed staged payloads; no remote action or model/array/data payload access.",
        "helper_source": {
            "path": str(HELPERS.relative_to(REPOSITORY)),
            "bytes": len(helpers),
            "sha256": sha256(helpers),
            "provenance_path": str(HELPER_PROVENANCE.relative_to(REPOSITORY)),
            "provenance_bytes": len(helper_provenance_raw),
            "provenance_sha256": sha256(helper_provenance_raw),
            "provenance": helper_provenance,
        },
        "body_template": {
            "path": str(BODY.relative_to(REPOSITORY)),
            "bytes": len(body),
            "sha256": sha256(body),
        },
        "payloads": payload_pins,
        "producer_consumer_review": {
            **EXPORT_REVIEW_PIN,
            "producer_source": payload_pins["export_first_cell_v1.py"],
            "producer_binding": payload_pins["EXPORT_BINDING_V1.json"],
            "source_manifest": payload_pins["EXPORT_SOURCE_MANIFEST.json"],
        },
        "rendered_source": rendered_pins,
        "remote_reply_contract": ["unit", "outcome", "admission_base64", "launch_base64"],
        "remote_receipts": {
            "admission": {"path": "ADMISSION.json", "unit_field": True},
            "launch": {
                "path": "LAUNCH.json",
                "unit_field": True,
                "invocation_id": "strict lowercase 32-hex",
            },
        },
        "staging": [
            "export_first_cell_v1.py",
            "EXPORT_BINDING_V1.json",
            "EXPORT_SOURCE_MANIFEST.json",
            "accepted_receipts/ALLTHREE_TERMINAL_ACCEPTANCE.json",
            "accepted_receipts/ALLTHREE_EXACT_STOP.json",
        ],
    }
    return rendered, binding


def main() -> None:
    rendered, binding = render()
    OUTPUT.write_bytes(rendered)
    OUTPUT_BINDING.write_text(
        json.dumps(binding, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(binding["rendered_source"], sort_keys=True))


if __name__ == "__main__":
    main()
