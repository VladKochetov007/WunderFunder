"""Prove V4 changes only the receipt parser relative to frozen V3."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

PACKET = Path(__file__).resolve().parent
ROOT = PACKET.parents[4]
CAMPAIGN = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1"
V3 = CAMPAIGN / "preparation/first_cell_evaluation_admission_v3/REMOTE_SUBMISSION_V3.py"
V4 = PACKET / "REMOTE_SUBMISSION_V4.py"
V4_BINDING = PACKET / "ADMISSION_SOURCE_BINDING_V4.json"
V4_OUTPUT = PACKET / "ADMISSION_V4_INVERSE_PROOF.json"
PAYLOAD_NAMES = (
    "export_first_cell_v1.py",
    "EXPORT_BINDING_V1.json",
    "EXPORT_SOURCE_MANIFEST.json",
    "ALLTHREE_TERMINAL_ACCEPTANCE.json",
    "ALLTHREE_EXACT_STOP.json",
)
EXPECTED_V3 = {
    "bytes": 243_576,
    "sha256": "d7e2ca4852da54d8d1554ff3271dd3d4f369575a6c26c7dc19990a2ca54cb509",
}


def _digest(raw: bytes) -> str:
    """Return a SHA-256 digest for exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def _function_span(source: str, filename: str, name: str) -> tuple[int, int, str]:
    """Return one top-level function's byte span and exact source."""
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source, filename=filename)
    matches = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1 or matches[0].end_lineno is None:
        raise ValueError(f"expected one top-level {name}")
    node = matches[0]
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    start = offsets[node.lineno - 1]
    end = offsets[node.end_lineno]
    return start, end, source[start:end]


def _inverse_body(v3_raw: bytes, v4_raw: bytes) -> bytes:
    """Replace V4's one changed function with the exact V3 function."""
    v3_source = v3_raw.decode("utf-8", errors="strict")
    v4_source = v4_raw.decode("utf-8", errors="strict")
    start, end, _ = _function_span(v4_source, str(V4), "check_fit_receipts")
    _, _, old_function = _function_span(v3_source, str(V3), "check_fit_receipts")
    return (v4_source[:start] + old_function + v4_source[end:]).encode("utf-8")


def _literal_assignment(body: bytes, name: str) -> object:
    """Read one top-level literal assignment without executing source."""
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
        raise ValueError(f"expected exactly one {name} assignment")
    return ast.literal_eval(matches[0])


def build_proof() -> dict[str, object]:
    """Verify V3 inverse, exact payload joins, and the preserved failed preflight."""
    v3_raw = V3.read_bytes()
    v4_raw = V4.read_bytes()
    if (len(v3_raw), _digest(v3_raw)) != (
        EXPECTED_V3["bytes"],
        EXPECTED_V3["sha256"],
    ):
        raise ValueError("frozen V3 body pin differs")
    if _inverse_body(v3_raw, v4_raw) != v3_raw:
        raise ValueError("reversing only check_fit_receipts does not reproduce V3")

    binding_raw = V4_BINDING.read_bytes()
    binding = json.loads(binding_raw)
    if not isinstance(binding, dict) or binding.get("schema") != (
        "fg19.first-cell-export-admission-source-binding.v4"
    ):
        raise ValueError("V4 binding schema differs")
    payloads = binding.get("payloads")
    if not isinstance(payloads, dict) or set(payloads) != set(PAYLOAD_NAMES):
        raise ValueError("V4 binding does not have exactly five fixed payloads")
    embedded = _literal_assignment(v4_raw, "PAYLOAD_B64")
    if not isinstance(embedded, dict) or set(embedded) != set(PAYLOAD_NAMES):
        raise ValueError("V4 body does not have exactly five embedded payloads")
    embedded_pins = {}
    for name in PAYLOAD_NAMES:
        raw = base64.b64decode(embedded[name], validate=True)
        expected = payloads[name]
        actual = {"bytes": len(raw), "sha256": _digest(raw)}
        if actual != {"bytes": expected.get("bytes"), "sha256": expected.get("sha256")}:
            raise ValueError(f"V4 embedded payload pin differs: {name}")
        embedded_pins[name] = actual

    review = binding.get("producer_consumer_review")
    if not isinstance(review, dict):
        raise ValueError("V4 binding lacks exporter producer-consumer review")
    for key, value in {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
        "first_cell_evaluation_export_v2/EXPORT_SOURCE_REVIEW_V2.json",
        "bytes": 10_250,
        "sha256": "f736f60a6b5481e94be380a183ba840aad850958c9b863b439bc708e65e997af",
        "schema": "fg19.first-cell-export-source-review.v2",
        "status": "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR",
    }.items():
        if review.get(key) != value:
            raise ValueError(f"V4 exporter review pin differs: {key}")
    for review_key, payload_key in (
        ("producer_source", "export_first_cell_v1.py"),
        ("producer_binding", "EXPORT_BINDING_V1.json"),
        ("source_manifest", "EXPORT_SOURCE_MANIFEST.json"),
    ):
        if review.get(review_key) != payloads[payload_key]:
            raise ValueError(f"V4 exporter review does not join {payload_key}")

    failed_v3 = binding.get("supersedes_v3_failed_preflight")
    if not isinstance(failed_v3, dict):
        raise ValueError("V4 binding omits preserved failed V3 metadata")
    expected_failed_body = {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
        "first_cell_evaluation_admission_v3/REMOTE_SUBMISSION_V3.py",
        "bytes": EXPECTED_V3["bytes"],
        "sha256": EXPECTED_V3["sha256"],
    }
    if failed_v3.get("body") != expected_failed_body:
        raise ValueError("V4 failed V3 body pin differs")
    body_tree = ast.parse(v4_raw.decode("utf-8", errors="strict"), filename=str(V4))
    if any(isinstance(node, ast.Name) and node.id == "__file__" for node in ast.walk(body_tree)):
        raise ValueError("remote body reads its own file path")
    if (
        sum(isinstance(node, ast.FunctionDef) and node.name == "main" for node in body_tree.body)
        != 1
    ):
        raise ValueError("remote body must have one main entrypoint")

    return {
        "schema": "fg19.first-cell-export-admission-v4-inverse-proof.v1",
        "status": "PASS_EXACT_V3_BODY_INVERSE_OUTSIDE_CHECK_FIT_RECEIPTS",
        "scope": "Local source and small fixed receipt metadata only; the changed function fixture stubs checkpoint stats. No model payload, project import, B action, export, or scoring.",
        "v3_body": {"bytes": len(v3_raw), "sha256": _digest(v3_raw)},
        "v4_body": {"bytes": len(v4_raw), "sha256": _digest(v4_raw)},
        "only_inverse_function": "check_fit_receipts",
        "all_other_body_bytes_reconstruct_v3": True,
        "payloads": embedded_pins,
        "producer_consumer_review": review,
        "failed_v3_preflight_preserved": failed_v3,
    }


def main() -> int:
    """Write a deterministic local inverse proof."""
    raw = (json.dumps(build_proof(), sort_keys=True, indent=2) + "\n").encode("utf-8")
    with V4_OUTPUT.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    print(f"{len(raw)} {_digest(raw)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
