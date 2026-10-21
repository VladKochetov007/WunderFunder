"""Prove V5 changes only V4's reviewed producer pins, unit/root, and payload bytes."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
from pathlib import Path

PACKET = Path(__file__).resolve().parent
CAMPAIGN = PACKET.parent.parent
REPOSITORY = PACKET.parents[4]
V4 = CAMPAIGN / "preparation/first_cell_evaluation_admission_v4/REMOTE_SUBMISSION_V4.py"
V5 = PACKET / "REMOTE_SUBMISSION_V5.py"
V5_BINDING = PACKET / "ADMISSION_SOURCE_BINDING_V5.json"
OUTPUT = PACKET / "ADMISSION_V5_INVERSE_PROOF.json"
V4_EXPECTED = {
    "bytes": 246_596,
    "sha256": "abe3251d9d65200b8e0c745db287e76f77e2dcb2a6524f3de7cd2f31c0cfa9ec",
}
PAYLOAD_NAMES = (
    "export_first_cell_v3.py",
    "EXPORT_BINDING_V3.json",
    "EXPORT_SOURCE_MANIFEST.json",
    "ALLTHREE_TERMINAL_ACCEPTANCE.json",
    "ALLTHREE_EXACT_STOP.json",
)


def _digest(raw: bytes) -> str:
    """Return the SHA-256 digest of exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def _assignment_span(raw: bytes, name: str) -> tuple[int, int]:
    """Locate one top-level assignment as exact UTF-8 byte offsets."""
    source = raw.decode("utf-8", errors="strict")
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    matches = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if isinstance(target, ast.Name) and target.id == name:
            matches.append(node)
    if len(matches) != 1 or matches[0].end_lineno is None:
        raise ValueError(f"expected exactly one top-level {name} assignment")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode("utf-8")))
    node = matches[0]
    return offsets[node.lineno - 1], offsets[node.end_lineno]


def _literal_assignment(raw: bytes, name: str) -> object:
    """Read a literal assignment without executing the body."""
    source = raw.decode("utf-8", errors="strict")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if isinstance(target, ast.Name) and target.id == name and node.value is not None:
            return ast.literal_eval(node.value)
    raise ValueError(f"missing literal assignment: {name}")


def _normalize_v5_to_v4(v4_raw: bytes, v5_raw: bytes) -> bytes:
    """Reverse only the frozen V3 producer-name, root, unit, and pin constants."""
    v4_start, v4_end = _assignment_span(v4_raw, "PAYLOAD_B64")
    v5_start, v5_end = _assignment_span(v5_raw, "PAYLOAD_B64")
    normalized = v5_raw[:v5_start] + v4_raw[v4_start:v4_end] + v5_raw[v5_end:]
    for newer, older in (
        (b"fg19_first_cell_f0_s1001_evaluation_v2", b"fg19_first_cell_f0_s1001_evaluation_v1"),
        (
            b"computer-vision-homework-fg19-first-cell-export-v2.service",
            b"computer-vision-homework-fg19-first-cell-export-v1.service",
        ),
        (
            b"372ec16016517baa510cc1899f140cd0978a35b83f96a4a5fa1ff4bda8b39730",
            b"6cd8c46f31214623d97e54c32bac9395608bfc918b9b44e879e44ad2548f0eb9",
        ),
        (
            b"9bfcd5110d200212bff45d2188f04e2eb094fbaf5cd78831ed03db5a130af8cb",
            b"1587a29b2269f6592a6487fc48765a8e86d4cc23aab4951f00966d970ca3dfa1",
        ),
        (b"export_first_cell_v3.py", b"export_first_cell_v1.py"),
        (b"EXPORT_BINDING_V3.json", b"EXPORT_BINDING_V1.json"),
        (b"66_080", b"64_315"),
    ):
        if newer not in normalized:
            raise ValueError(f"expected V5 identity literal is absent: {newer!r}")
        normalized = normalized.replace(newer, older)
    return normalized


def build_proof() -> dict[str, object]:
    """Verify exact inversion, five payload joins, review fields, and new root/unit."""
    v4_raw = V4.read_bytes()
    v5_raw = V5.read_bytes()
    if (len(v4_raw), _digest(v4_raw)) != (V4_EXPECTED["bytes"], V4_EXPECTED["sha256"]):
        raise ValueError("frozen V4 body pin differs")
    if _normalize_v5_to_v4(v4_raw, v5_raw) != v4_raw:
        raise ValueError("V5 changes outside the reviewed identity and payload slots")

    binding_raw = V5_BINDING.read_bytes()
    binding = json.loads(binding_raw)
    if not isinstance(binding, dict) or binding.get("schema") != (
        "fg19.first-cell-export-admission-source-binding.v5"
    ):
        raise ValueError("V5 binding schema differs")
    payloads = binding.get("payloads")
    if not isinstance(payloads, dict) or set(payloads) != set(PAYLOAD_NAMES):
        raise ValueError("V5 binding does not contain exactly the fixed five payloads")
    embedded = _literal_assignment(v5_raw, "PAYLOAD_B64")
    if not isinstance(embedded, dict) or set(embedded) != set(PAYLOAD_NAMES):
        raise ValueError("V5 body does not contain exactly the fixed five payloads")
    embedded_pins: dict[str, dict[str, object]] = {}
    for name in PAYLOAD_NAMES:
        raw = base64.b64decode(embedded[name], validate=True)
        pin = payloads[name]
        if (len(raw), _digest(raw)) != (pin.get("bytes"), pin.get("sha256")):
            raise ValueError(f"embedded payload differs from binding: {name}")
        embedded_pins[name] = {"bytes": len(raw), "sha256": _digest(raw)}

    review = binding.get("producer_consumer_review")
    if not isinstance(review, dict) or (
        review.get("path"),
        review.get("bytes"),
        review.get("sha256"),
        review.get("schema"),
        review.get("status"),
        review.get("reviewed_file_count"),
    ) != (
        "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
        "first_cell_evaluation_export_v3/EXPORT_SOURCE_REVIEW_V3.json",
        7_284,
        "ee53ce06a6119152866e8f39801e7575e59fb42bc09515d102db4c8df5d9c37a",
        "fg19.first-cell-export-source-review.v3",
        "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR",
        11,
    ):
        raise ValueError("V5 binding does not pin the exact exporter V3 producer review")
    for key, payload in (
        ("producer_source", "export_first_cell_v3.py"),
        ("producer_binding", "EXPORT_BINDING_V3.json"),
        ("source_manifest", "EXPORT_SOURCE_MANIFEST.json"),
    ):
        if review.get(key) != {
            **payloads[payload],
            "path": payloads[payload]["path"],
        }:
            raise ValueError(f"V5 review does not join its producer payload: {payload}")

    previous_failure = binding.get("supersedes_previous_actual_export_failure")
    if (
        not isinstance(previous_failure, dict)
        or previous_failure.get("root_only_not_staged_to_B") is not True
        or previous_failure.get("old_unit_must_not_be_queried_stopped_or_retried") is not True
        or previous_failure.get("unit")
        != "computer-vision-homework-fg19-first-cell-export-v1.service"
        or previous_failure.get("invocation_id") != "da6c9e1f5897478f9f3f686de58e39cc"
    ):
        raise ValueError("V5 binding lacks the root-only preserved prior failure pins")

    body_tree = ast.parse(v5_raw.decode("utf-8", errors="strict"), filename=str(V5))
    if any(isinstance(node, ast.Name) and node.id == "__file__" for node in ast.walk(body_tree)):
        raise ValueError("remote stdin body reads its own file path")
    if (
        sum(isinstance(node, ast.FunctionDef) and node.name == "main" for node in body_tree.body)
        != 1
    ):
        raise ValueError("remote body must have one main entrypoint")
    assignments = {}
    for node in body_tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if isinstance(target, ast.Name) and target.id in {
            "UNIT",
            "EXPORT_BINDING_SHA256",
            "EXPORTER_SHA256",
        }:
            assignments[target.id] = ast.literal_eval(node.value)
    if assignments != {
        "UNIT": "computer-vision-homework-fg19-first-cell-export-v2.service",
        "EXPORT_BINDING_SHA256": "372ec16016517baa510cc1899f140cd0978a35b83f96a4a5fa1ff4bda8b39730",
        "EXPORTER_SHA256": "9bfcd5110d200212bff45d2188f04e2eb094fbaf5cd78831ed03db5a130af8cb",
    }:
        raise ValueError("V5 body unit or exporter payload constants differ")

    return {
        "schema": "fg19.first-cell-export-admission-v5-inverse-proof.v1",
        "status": "PASS_EXACT_V4_BODY_INVERSE_OUTSIDE_FIXED_PAYLOAD_AND_IDENTITY_SLOTS",
        "scope": "Local source and small pinned metadata only. No remote action, source/model/data payload read, project import, or worker execution.",
        "v4_body": {
            "path": str(V4.relative_to(REPOSITORY)),
            "bytes": len(v4_raw),
            "sha256": _digest(v4_raw),
        },
        "v5_body": {
            "path": str(V5.relative_to(REPOSITORY)),
            "bytes": len(v5_raw),
            "sha256": _digest(v5_raw),
        },
        "v5_binding": {
            "path": str(V5_BINDING.relative_to(REPOSITORY)),
            "bytes": len(binding_raw),
            "sha256": _digest(binding_raw),
        },
        "payloads": embedded_pins,
        "review_contract": {
            key: review[key]
            for key in ("path", "bytes", "sha256", "schema", "status", "reviewed_file_count")
        },
        "historical_failed_export": previous_failure,
        "inverse": "Replacing only PAYLOAD_B64 with V4's exact assignment and reversing the exact exporter V3 filename/hash/size, V3 binding name/hash, export-v2 unit, and evaluation_v2 root reproduces the frozen V4 body byte-for-byte.",
        "entrypoint_count": 1,
        "self_path_read": False,
    }


def main() -> None:
    proof = build_proof()
    raw = (json.dumps(proof, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")
    with OUTPUT.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
