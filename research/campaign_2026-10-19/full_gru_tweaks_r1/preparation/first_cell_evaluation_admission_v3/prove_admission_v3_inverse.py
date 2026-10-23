"""Prove V3 preserves the V2 admission body outside fixed payload constants."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[4]
V2 = ROOT / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_admission_v2"
)
V2_BODY = V2 / "REMOTE_SUBMISSION_V2.py"
V3_BODY = PACKET / "REMOTE_SUBMISSION_V3.py"
V3_BINDING = PACKET / "ADMISSION_SOURCE_BINDING_V3.json"
V3_OUTPUT = PACKET / "ADMISSION_V3_INVERSE_PROOF.json"
PAYLOAD_NAMES = (
    "export_first_cell_v1.py",
    "EXPORT_BINDING_V1.json",
    "EXPORT_SOURCE_MANIFEST.json",
    "ALLTHREE_TERMINAL_ACCEPTANCE.json",
    "ALLTHREE_EXACT_STOP.json",
)
REPLACED_ASSIGNMENTS = (
    "EXPORT_BINDING_SHA256",
    "EXPORTER_SHA256",
    "PAYLOAD_B64",
    "PAYLOAD_PINS",
)
EXPECTED_V2_BODY = {
    "bytes": 243_608,
    "sha256": "a15b2cec55657f3f630fcd18a4fe20037fe08c3186bb0f046aac5480c6e0aeab",
}
EXPECTED_EXPORT_REVIEW = {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
    "first_cell_evaluation_export_v2/EXPORT_SOURCE_REVIEW_V2.json",
    "bytes": 10_250,
    "sha256": "f736f60a6b5481e94be380a183ba840aad850958c9b863b439bc708e65e997af",
    "schema": "fg19.first-cell-export-source-review.v2",
    "status": "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR",
}


def _digest(raw: bytes) -> str:
    """Return a SHA-256 digest for exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def _assignment_spans(source: str, filename: str) -> dict[str, tuple[int, int, str]]:
    """Find one top-level assignment span for each payload-related constant."""
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source, filename=filename)
    result: dict[str, tuple[int, int, str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if not isinstance(target, ast.Name) or target.id not in REPLACED_ASSIGNMENTS:
            continue
        if node.end_lineno is None:
            raise ValueError(f"assignment has no end line: {target.id}")
        start_line = node.lineno - 1
        end_line = node.end_lineno
        result[target.id] = (start_line, end_line, "".join(lines[start_line:end_line]))
    if set(result) != set(REPLACED_ASSIGNMENTS):
        raise ValueError("V2/V3 body does not contain the exact fixed-constant set")
    return result


def _inverse_body(v2_raw: bytes, v3_raw: bytes) -> bytes:
    """Restore the V2 payload assignments and require exact source equality."""
    v2_source = v2_raw.decode("utf-8", errors="strict")
    v3_source = v3_raw.decode("utf-8", errors="strict")
    old_spans = _assignment_spans(v2_source, str(V2_BODY))
    new_spans = _assignment_spans(v3_source, str(V3_BODY))
    lines = v3_source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    replacements = []
    for name in REPLACED_ASSIGNMENTS:
        start_line, end_line, _ = new_spans[name]
        old_text = old_spans[name][2]
        replacements.append((offsets[start_line], offsets[end_line], old_text))
    restored = v3_source
    for start, end, old_text in sorted(replacements, reverse=True):
        restored = restored[:start] + old_text + restored[end:]
    return restored.encode("utf-8")


def _literal_assignment(body: bytes, name: str) -> object:
    """Extract one top-level literal assignment without executing the body."""
    tree = ast.parse(body.decode("utf-8", errors="strict"))
    matches = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if isinstance(target, ast.Name) and target.id == name and node.value is not None:
            matches.append(node.value)
    if len(matches) != 1:
        raise ValueError(f"expected one {name} assignment")
    return ast.literal_eval(matches[0])


def build_proof() -> dict[str, object]:
    """Verify exact body inverse, five payload bytes, and V2 reviewer binding."""
    v2_raw = V2_BODY.read_bytes()
    v3_raw = V3_BODY.read_bytes()
    if (len(v2_raw), _digest(v2_raw)) != (
        EXPECTED_V2_BODY["bytes"],
        EXPECTED_V2_BODY["sha256"],
    ):
        raise ValueError("frozen actual V2 body pin differs")
    if _inverse_body(v2_raw, v3_raw) != v2_raw:
        raise ValueError("reversing only V3 payload constants does not reproduce V2 body")

    binding_raw = V3_BINDING.read_bytes()
    binding = json.loads(binding_raw)
    if not isinstance(binding, dict):
        raise ValueError("V3 source binding is not a JSON object")
    payloads = binding.get("payloads")
    if not isinstance(payloads, dict) or set(payloads) != set(PAYLOAD_NAMES):
        raise ValueError("V3 source binding does not contain exactly the five payloads")
    embedded = _literal_assignment(v3_raw, "PAYLOAD_B64")
    if not isinstance(embedded, dict) or set(embedded) != set(PAYLOAD_NAMES):
        raise ValueError("V3 remote body does not embed exactly the five fixed payloads")
    embedded_pins: dict[str, dict[str, object]] = {}
    for name in PAYLOAD_NAMES:
        raw = base64.b64decode(embedded[name], validate=True)
        pin = payloads[name]
        actual = {"bytes": len(raw), "sha256": _digest(raw)}
        if actual != {"bytes": pin.get("bytes"), "sha256": pin.get("sha256")}:
            raise ValueError(f"V3 embedded payload differs from source binding: {name}")
        embedded_pins[name] = actual

    review = binding.get("producer_consumer_review")
    if not isinstance(review, dict):
        raise ValueError("V3 source binding lacks the exporter producer-consumer review")
    for key, value in EXPECTED_EXPORT_REVIEW.items():
        if review.get(key) != value:
            raise ValueError(f"V3 source binding review pin differs: {key}")
    if review.get("producer_source") != payloads["export_first_cell_v1.py"]:
        raise ValueError("V3 review producer source does not join the embedded payload")
    if review.get("producer_binding") != payloads["EXPORT_BINDING_V1.json"]:
        raise ValueError("V3 review producer binding does not join the embedded payload")
    if review.get("source_manifest") != payloads["EXPORT_SOURCE_MANIFEST.json"]:
        raise ValueError("V3 review source manifest does not join the embedded payload")

    failed_v2 = binding.get("supersedes_v2_failed_preflight")
    if not isinstance(failed_v2, dict):
        raise ValueError("V3 source binding does not preserve the failed V2 preflight")
    failed_v2_body = failed_v2.get("body")
    if (
        not isinstance(failed_v2_body, dict)
        or {key: failed_v2_body.get(key) for key in EXPECTED_V2_BODY} != EXPECTED_V2_BODY
    ):
        raise ValueError("V3 failed-V2 body pin differs")

    return {
        "schema": "fg19.first-cell-export-admission-v3-inverse-proof.v1",
        "status": "PASS_EXACT_V2_BODY_INVERSE_OUTSIDE_FIVE_PAYLOAD_CONSTANTS",
        "scope": "Local source and small fixed metadata only; no imports, tests, B action, model/data payload access, export, or scoring.",
        "v2_body": {"bytes": len(v2_raw), "sha256": _digest(v2_raw)},
        "v3_body": {"bytes": len(v3_raw), "sha256": _digest(v3_raw)},
        "inverse_assignments": list(REPLACED_ASSIGNMENTS),
        "all_other_body_bytes_reconstruct_v2": True,
        "payloads": embedded_pins,
        "producer_consumer_review": review,
        "failed_v2_preflight_preserved": failed_v2,
    }


def main() -> int:
    """Write a deterministic inverse-proof receipt beside this script."""
    raw = (json.dumps(build_proof(), sort_keys=True, indent=2) + "\n").encode()
    with V3_OUTPUT.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    print(f"{len(raw)} {_digest(raw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
