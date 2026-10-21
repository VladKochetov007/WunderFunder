#!/usr/bin/env python3
"""Build the V13 score-admission stdin body from the reviewed V12 source bytes."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[2]
PREPARATION = CAMPAIGN / "preparation"
PACKET = PREPARATION / "first_cell_evaluation_score_admission_v13"
V12 = PREPARATION / "first_cell_evaluation_score_admission_v12"
V6 = PREPARATION / "first_cell_evaluation_score_v6"
V5 = PREPARATION / "first_cell_evaluation_score_v5"
EXPORT_V4 = PREPARATION / "first_cell_evaluation_export_v4"
RECORDS = CAMPAIGN / "records/first_cell_evaluation_v1"

V12_BODY_PIN = (
    188838,
    "133bb309b5b3ebba238ce4f40a9e9efa60c6813a0fcd208ab90ff4bd7bba799e",
)
V12_BINDING_PIN = (
    8761,
    "f177f935578639e8ee1933750d85e9d610e27def44842dc7a3685c986e743241",
)
V12_REVIEW_PIN = (
    10264,
    "881f2682ab1fb2796969ee7e3fe650c5b5fcdd43859c07c86ad95f7ac2c83aeb",
)
EXPORT_ACCEPTANCE_PIN = (
    8833,
    "5a9a99121f80eb7fa07e4591318bea81078ca8b2db1032f150cc33afcc728d0d",
)
EXPORT_STOP_PIN = (
    1140,
    "695e06c60452e969349b64356dcd795bcbb45892328261136f105b5f8673b72f",
)
NEW_UNIT = "computer-vision-homework-fg19-first-cell-score-v2.service"
NEW_SCORE_DIR = "fg19_first_cell_f0_s1001_score_v2"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def checked_bytes(path: Path, expected: tuple[int, str]) -> bytes:
    raw = path.read_bytes()
    if (len(raw), sha(raw)) != expected:
        raise ValueError(f"pinned input changed: {path}")
    return raw


def assigned_value(tree: ast.Module, name: str) -> tuple[ast.AST, ast.AST]:
    matches: list[tuple[ast.AST, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is not None and any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            matches.append((node, value))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} assignment")
    return matches[0]


def parse_assignments(body: bytes) -> dict[str, object]:
    tree = ast.parse(body)
    values: dict[str, object] = {}
    for name in ("SCORER_FILES", "PAYLOAD_B64"):
        _, node = assigned_value(tree, name)
        value = ast.literal_eval(node)
        if not isinstance(value, dict):
            raise ValueError(f"{name} is not a literal dictionary")
        values[name] = value
    return values


def read_sources() -> dict[str, tuple[bytes, int, str]]:
    result: dict[str, tuple[bytes, int, str]] = {}
    for name in (
        "first_cell_score_contract.py",
        "first_cell_score_core.py",
        "first_cell_score_predictions.py",
        "score_first_cell.py",
    ):
        raw = (V6 / name).read_bytes()
        result[name] = (raw, len(raw), sha(raw))
    return result


def validate_v12_payloads(body: bytes) -> tuple[dict[str, object], dict[str, object]]:
    assignments = parse_assignments(body)
    scorer_files = assignments["SCORER_FILES"]
    payload_b64 = assignments["PAYLOAD_B64"]
    expected_v5 = {
        name: (V5 / name).read_bytes()
        for name in (
            "first_cell_score_contract.py",
            "first_cell_score_core.py",
            "first_cell_score_predictions.py",
            "score_first_cell.py",
        )
    }
    for name, expected in expected_v5.items():
        decoded = base64.b64decode(payload_b64[name], validate=True)
        if decoded != expected or scorer_files[name] != (len(expected), sha(expected)):
            raise ValueError(f"V12 embedded scorer differs from its V5 pin: {name}")
    receipt_sources = {
        "EXPORT_ACCEPTANCE.json": RECORDS / "export_v5/EXPORT_ACCEPTANCE.json",
        "EXPORT_EXACT_STOP.json": RECORDS / "export_v5/exact_stop/EXACT_STOP.json",
    }
    receipt_pins = {
        "EXPORT_ACCEPTANCE.json": EXPORT_ACCEPTANCE_PIN,
        "EXPORT_EXACT_STOP.json": EXPORT_STOP_PIN,
    }
    for name, path in receipt_sources.items():
        raw = checked_bytes(path, receipt_pins[name])
        if base64.b64decode(payload_b64[name], validate=True) != raw:
            raise ValueError(f"V12 embedded root export authority differs: {name}")
    if set(payload_b64) != set(scorer_files) | set(receipt_sources):
        raise ValueError("V12 exact six-payload membership differs")
    return scorer_files, payload_b64


def _replace_once(raw: bytes, before: bytes, after: bytes, label: str) -> bytes:
    count = raw.count(before)
    if count != 1:
        raise ValueError(f"expected one exact {label} source literal; found {count}")
    return raw.replace(before, after, 1)


def derive_body() -> tuple[bytes, dict[str, object]]:
    v12 = checked_bytes(V12 / "REMOTE_SUBMISSION_V12.py", V12_BODY_PIN)
    checked_bytes(V12 / "SCORE_ADMISSION_SOURCE_BINDING_V12.json", V12_BINDING_PIN)
    checked_bytes(V12 / "ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V12.json", V12_REVIEW_PIN)
    scorer_files_v12, payloads_v12 = validate_v12_payloads(v12)
    sources_v6 = read_sources()

    body = v12
    body = _replace_once(
        body,
        b'SCORE_D = SCRATCH_ROOT / "fg19_first_cell_f0_s1001_score_v1"',
        f'SCORE_D = SCRATCH_ROOT / "{NEW_SCORE_DIR}"'.encode(),
        "score directory",
    )
    body = _replace_once(
        body,
        b'UNIT = "computer-vision-homework-fg19-first-cell-score-v1.service"',
        f'UNIT = "{NEW_UNIT}"'.encode(),
        "unit identity",
    )

    old_core = scorer_files_v12["first_cell_score_core.py"]
    new_core_raw, new_core_bytes, new_core_sha = sources_v6["first_cell_score_core.py"]
    old_tuple = (
        b'    "first_cell_score_core.py": (\n'
        + f"        {old_core[0]:_},\n".encode()
        + f'        "{old_core[1]}",\n'.encode()
        + b"    ),"
    )
    new_tuple = (
        b'    "first_cell_score_core.py": (\n'
        + f"        {new_core_bytes:_},\n".encode()
        + f'        "{new_core_sha}",\n'.encode()
        + b"    ),"
    )
    body = _replace_once(body, old_tuple, new_tuple, "V6 core source pin")

    old_core_b64 = payloads_v12["first_cell_score_core.py"]
    new_core_b64 = base64.b64encode(new_core_raw).decode("ascii")
    old_payload_line = b"    'first_cell_score_core.py': '" + old_core_b64.encode("ascii") + b"',"
    new_payload_line = b"    'first_cell_score_core.py': '" + new_core_b64.encode("ascii") + b"',"
    body = _replace_once(body, old_payload_line, new_payload_line, "V6 core payload")

    if b"score-v1.service" in body or b"score_v1" in body:
        raise ValueError("V13 body retained an old score unit or private-D identity")
    if b"score-v2.service" not in body or NEW_SCORE_DIR.encode() not in body:
        raise ValueError("V13 body does not bind the new score-v2 unit and private-D")
    current_files, current_payloads = parse_assignments(body).values()
    if current_files != {
        name: (raw_bytes, raw_sha) for name, (_raw, raw_bytes, raw_sha) in sources_v6.items()
    }:
        raise ValueError("V13 scorer pin map differs from all four V6 scorer sources")
    for name, (raw, _byte_count, _digest) in sources_v6.items():
        if base64.b64decode(current_payloads[name], validate=True) != raw:
            raise ValueError(f"V13 scorer payload differs from current source: {name}")
    for name in (
        "first_cell_score_contract.py",
        "first_cell_score_predictions.py",
        "score_first_cell.py",
    ):
        if current_payloads[name] != payloads_v12[name]:
            raise ValueError(f"V13 changed an unchanged scorer payload: {name}")
    for name in ("EXPORT_ACCEPTANCE.json", "EXPORT_EXACT_STOP.json"):
        if current_payloads[name] != payloads_v12[name]:
            raise ValueError(f"V13 changed fixed export authority payload: {name}")

    restored = body
    restored = _replace_once(
        restored,
        f'SCORE_D = SCRATCH_ROOT / "{NEW_SCORE_DIR}"'.encode(),
        b'SCORE_D = SCRATCH_ROOT / "fg19_first_cell_f0_s1001_score_v1"',
        "score directory inverse",
    )
    restored = _replace_once(
        restored,
        f'UNIT = "{NEW_UNIT}"'.encode(),
        b'UNIT = "computer-vision-homework-fg19-first-cell-score-v1.service"',
        "unit identity inverse",
    )
    restored = _replace_once(restored, new_tuple, old_tuple, "V6 core pin inverse")
    restored = _replace_once(
        restored, new_payload_line, old_payload_line, "V6 core payload inverse"
    )
    if restored != v12:
        raise AssertionError("reversing the four V13 deltas did not restore exact V12 bytes")

    proof = {
        "schema": "fg19.first-cell-score-admission-v13-source-payload-proof.v1",
        "status": "PASS_V13_PAYLOAD_BINDING_AND_EXACT_INVERSE_NO_EXECUTION",
        "parent_v12_body": {
            "bytes": len(v12),
            "sha256": sha(v12),
            "path": "preparation/first_cell_evaluation_score_admission_v12/REMOTE_SUBMISSION_V12.py",
        },
        "v13_body": {"bytes": len(body), "sha256": sha(body), "path": "REMOTE_SUBMISSION_V13.py"},
        "exact_delta": {
            "changed_assignments": [
                "SCORE_D",
                "UNIT",
                "SCORER_FILES.first_cell_score_core.py",
                "PAYLOAD_B64.first_cell_score_core.py",
            ],
            "unit": NEW_UNIT,
            "score_directory": NEW_SCORE_DIR,
            "v6_core": {"bytes": new_core_bytes, "sha256": new_core_sha},
            "other_three_scorer_payloads_byte_identical_to_v12": True,
            "two_export_authority_payloads_byte_identical_to_v12": True,
            "exact_inverse_restores_v12_body": True,
        },
        "bound_payloads": {
            name: {
                "bytes": len(base64.b64decode(encoded, validate=True)),
                "sha256": sha(base64.b64decode(encoded, validate=True)),
            }
            for name, encoded in current_payloads.items()
        },
        "boundary": {
            "remote_body_executed": False,
            "b_or_ssh_action": False,
            "scoring_or_result_read": False,
            "model_cache_array_or_prediction_payload_read_local": False,
            "only_source_code_and_small_pinned_json_receipts_read_local": True,
        },
    }
    return body, proof


if __name__ == "__main__":
    body, proof = derive_body()
    body_path = PACKET / "REMOTE_SUBMISSION_V13.py"
    with body_path.open("xb") as stream:
        stream.write(body)
        stream.flush()
    proof_path = PACKET / "V13_PAYLOAD_AND_INVERSE_PROOF.json"
    proof_raw = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode()
    with proof_path.open("xb") as stream:
        stream.write(proof_raw)
        stream.flush()
    print(
        json.dumps(
            {
                "body": {"bytes": len(body), "sha256": sha(body), "path": str(body_path)},
                "proof": {
                    "bytes": len(proof_raw),
                    "sha256": sha(proof_raw),
                    "path": str(proof_path),
                },
            }
        )
    )
