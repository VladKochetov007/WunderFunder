"""Prove V6 changes only the frozen V5 producer, unit/root, and payload slots."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
from pathlib import Path

PACKET = Path(__file__).resolve().parent
REPOSITORY = PACKET.parents[4]
PREVIOUS = REPOSITORY / (
    "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
    "first_cell_evaluation_admission_v5/REMOTE_SUBMISSION_V5.py"
)
V6 = PACKET / "REMOTE_SUBMISSION_V6.py"
V6_BINDING = PACKET / "ADMISSION_SOURCE_BINDING_V6.json"
OUTPUT = PACKET / "ADMISSION_V6_INVERSE_PROOF.json"
PREVIOUS_EXPECTED = {
    "bytes": 248_948,
    "sha256": "e24a6fc87e72a43056d30e1b4edf2cb12e410a37f549a858e89420e54cdb38c0",
}
PAYLOAD_NAMES = (
    "export_first_cell_v4.py",
    "EXPORT_BINDING_V4.json",
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
    tree = ast.parse(raw.decode("utf-8", errors="strict"))
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


def _normalize_v6_to_v5(previous_raw: bytes, v6_raw: bytes) -> bytes:
    """Reverse only V4 producer name/pins, new evidence root, unit, and payload bytes."""
    old_start, old_end = _assignment_span(previous_raw, "PAYLOAD_B64")
    new_start, new_end = _assignment_span(v6_raw, "PAYLOAD_B64")
    normalized = v6_raw[:new_start] + previous_raw[old_start:old_end] + v6_raw[new_end:]
    for newer, older in (
        (b"fg19_first_cell_f0_s1001_evaluation_v3", b"fg19_first_cell_f0_s1001_evaluation_v2"),
        (
            b"computer-vision-homework-fg19-first-cell-export-v3.service",
            b"computer-vision-homework-fg19-first-cell-export-v2.service",
        ),
        (
            b"4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd",
            b"372ec16016517baa510cc1899f140cd0978a35b83f96a4a5fa1ff4bda8b39730",
        ),
        (
            b"87b6d3626bdafe6353865d95b1d8116bd187cb4da46888815658db6fc192738e",
            b"9bfcd5110d200212bff45d2188f04e2eb094fbaf5cd78831ed03db5a130af8cb",
        ),
        (b"export_first_cell_v4.py", b"export_first_cell_v3.py"),
        (b"EXPORT_BINDING_V4.json", b"EXPORT_BINDING_V3.json"),
        (b"66_790", b"66_080"),
        (b"15_796", b"15_798"),
    ):
        if newer not in normalized:
            raise ValueError(f"expected V6 identity literal is absent: {newer!r}")
        normalized = normalized.replace(newer, older)
    return normalized


def build_proof() -> dict[str, object]:
    """Verify exact inversion, fixed payload joins, review fields, and root/unit."""
    previous_raw = PREVIOUS.read_bytes()
    v6_raw = V6.read_bytes()
    if (len(previous_raw), _digest(previous_raw)) != (
        PREVIOUS_EXPECTED["bytes"],
        PREVIOUS_EXPECTED["sha256"],
    ):
        raise ValueError("frozen V5 body pin differs")
    if _normalize_v6_to_v5(previous_raw, v6_raw) != previous_raw:
        raise ValueError("V6 changes outside the reviewed identity and payload slots")

    binding_raw = V6_BINDING.read_bytes()
    binding = json.loads(binding_raw)
    if not isinstance(binding, dict) or binding.get("schema") != (
        "fg19.first-cell-export-admission-source-binding.v6"
    ):
        raise ValueError("V6 binding schema differs")
    payloads = binding.get("payloads")
    if not isinstance(payloads, dict) or set(payloads) != set(PAYLOAD_NAMES):
        raise ValueError("V6 binding does not contain exactly the fixed five payloads")
    embedded = _literal_assignment(v6_raw, "PAYLOAD_B64")
    if not isinstance(embedded, dict) or set(embedded) != set(PAYLOAD_NAMES):
        raise ValueError("V6 body does not contain exactly the fixed five payloads")
    embedded_pins: dict[str, dict[str, object]] = {}
    for name in PAYLOAD_NAMES:
        raw = base64.b64decode(embedded[name], validate=True)
        pin = payloads[name]
        if (len(raw), _digest(raw)) != (pin.get("bytes"), pin.get("sha256")):
            raise ValueError(f"embedded payload differs from binding: {name}")
        embedded_pins[name] = {"bytes": len(raw), "sha256": _digest(raw)}

    review = binding.get("producer_consumer_review")
    if (
        not isinstance(review, dict)
        or (
            review.get("path"),
            review.get("schema"),
            review.get("status"),
            review.get("reviewed_file_count"),
        )
        != (
            "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/"
            "first_cell_evaluation_export_v4/EXPORT_SOURCE_REVIEW_V4.json",
            "fg19.first-cell-export-source-review.v4",
            "PASS_SOURCE_ONLY_PRODUCER_CONSUMER_PAIR",
            12,
        )
        or type(review.get("bytes")) is not int
        or review["bytes"] <= 0
    ):
        raise ValueError("V6 binding does not pin the passing exporter V4 producer review")
    for key, payload in (
        ("producer_source", "export_first_cell_v4.py"),
        ("producer_binding", "EXPORT_BINDING_V4.json"),
        ("source_manifest", "EXPORT_SOURCE_MANIFEST.json"),
    ):
        if review.get(key) != payloads[payload]:
            raise ValueError(f"V6 review does not join its producer payload: {payload}")

    first_failure = binding.get("supersedes_previous_actual_export_failure")
    latest_failure = binding.get("supersedes_latest_actual_export_failure")
    for failure, unit, invocation in (
        (
            first_failure,
            "computer-vision-homework-fg19-first-cell-export-v1.service",
            "da6c9e1f5897478f9f3f686de58e39cc",
        ),
        (
            latest_failure,
            "computer-vision-homework-fg19-first-cell-export-v2.service",
            "555122f41c884850a6bd4e847c14e82f",
        ),
    ):
        if (
            not isinstance(failure, dict)
            or failure.get("historical_only") is not True
            or failure.get("root_only_not_staged_to_B") is not True
            or failure.get("old_unit_must_not_be_queried_stopped_or_retried") is not True
            or failure.get("unit") != unit
            or failure.get("invocation_id") != invocation
        ):
            raise ValueError("V6 binding lacks a required root-only historical failure pin")

    body_tree = ast.parse(v6_raw.decode("utf-8", errors="strict"), filename=str(V6))
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
        "UNIT": "computer-vision-homework-fg19-first-cell-export-v3.service",
        "EXPORT_BINDING_SHA256": "4a8ae22204971ac07d6e5cdfd9f65a62e9f8440831035db7271e4e0da3d83abd",
        "EXPORTER_SHA256": "87b6d3626bdafe6353865d95b1d8116bd187cb4da46888815658db6fc192738e",
    }:
        raise ValueError("V6 body unit or exporter payload constants differ")

    return {
        "schema": "fg19.first-cell-export-admission-v6-inverse-proof.v1",
        "status": "PASS_EXACT_V5_BODY_INVERSE_OUTSIDE_FIXED_PAYLOAD_AND_IDENTITY_SLOTS",
        "scope": "Local source and small pinned metadata only. No remote action, source/model/data payload read, project import, or worker execution.",
        "v5_body": {
            "path": str(PREVIOUS.relative_to(REPOSITORY)),
            "bytes": len(previous_raw),
            "sha256": _digest(previous_raw),
        },
        "v6_body": {
            "path": str(V6.relative_to(REPOSITORY)),
            "bytes": len(v6_raw),
            "sha256": _digest(v6_raw),
        },
        "v6_binding": {
            "path": str(V6_BINDING.relative_to(REPOSITORY)),
            "bytes": len(binding_raw),
            "sha256": _digest(binding_raw),
        },
        "payloads": embedded_pins,
        "review_contract": {
            key: review[key]
            for key in ("path", "bytes", "sha256", "schema", "status", "reviewed_file_count")
        },
        "historical_failed_exports": [first_failure, latest_failure],
        "inverse": "Replacing PAYLOAD_B64 with V5's exact assignment and reversing only the V4 exporter filename/hash/size, V4 binding name/hash/size, export-v3 unit, and evaluation_v3 root reproduces the frozen V5 body byte-for-byte.",
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
