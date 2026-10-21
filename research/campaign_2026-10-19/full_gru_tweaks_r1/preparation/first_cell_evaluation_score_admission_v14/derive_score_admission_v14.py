#!/usr/bin/env python3
"""Fork the frozen V13 admission body to fresh score-v3/D3 and the V7 core."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

CAMPAIGN = Path(__file__).resolve().parents[2]
PREPARATION = CAMPAIGN / "preparation"
PACKET = PREPARATION / "first_cell_evaluation_score_admission_v14"
V13 = PREPARATION / "first_cell_evaluation_score_admission_v13"
V6 = PREPARATION / "first_cell_evaluation_score_v6"
V7 = PREPARATION / "first_cell_evaluation_score_v7"
RECORDS = CAMPAIGN / "records/first_cell_evaluation_v1"

V13_BODY_PIN = (
    188_490,
    "409e2357c71799de1d486914757faf001661722f355602c62122fd32c4336ab5",
)
V13_BINDING_PIN = (
    10_316,
    "a1b70ee263a8fd03ee4042f1919319b56344d326fa51706d57315cd723c9f4a7",
)
V13_REVIEW_PIN = (
    15_305,
    "3f4b165f131ddf3ac992aaf7300b8cbd60e9a70ba8ca915546341374b60e510d",
)
V13_PROOF_PIN = (
    2_314,
    "6b5ab8c1792eeeb22a658b18e969a713d8065dc9a74d492a2840f373de71636d",
)
EXPORT_ACCEPTANCE_PIN = (
    8_833,
    "5a9a99121f80eb7fa07e4591318bea81078ca8b2db1032f150cc33afcc728d0d",
)
EXPORT_STOP_PIN = (
    1_140,
    "695e06c60452e969349b64356dcd795bcbb45892328261136f105b5f8673b72f",
)
NEW_UNIT = "computer-vision-homework-fg19-first-cell-score-v3.service"
NEW_SCORE_DIR = "fg19_first_cell_f0_s1001_score_v3"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pin(path: Path, relative: str | None = None) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": relative or str(path.relative_to(CAMPAIGN)),
        "bytes": len(raw),
        "sha256": sha(raw),
    }


def checked_bytes(path: Path, expected: tuple[int, str]) -> bytes:
    raw = path.read_bytes()
    if (len(raw), sha(raw)) != expected:
        raise ValueError(f"pinned input changed: {path}")
    return raw


def assigned_value(tree: ast.Module, name: str) -> ast.AST:
    matches: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        if value is not None and any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            matches.append(value)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name} assignment")
    return matches[0]


def assignments(body: bytes) -> tuple[dict[str, object], dict[str, object]]:
    tree = ast.parse(body)
    scorer_files = ast.literal_eval(assigned_value(tree, "SCORER_FILES"))
    payloads = ast.literal_eval(assigned_value(tree, "PAYLOAD_B64"))
    if not isinstance(scorer_files, dict) or not isinstance(payloads, dict):
        raise ValueError("scorer file or payload assignment is not a literal dictionary")
    return scorer_files, payloads


def replace_once(body: bytes, old: bytes, new: bytes, label: str) -> bytes:
    count = body.count(old)
    if count != 1:
        raise ValueError(f"expected one exact {label} literal; found {count}")
    return body.replace(old, new, 1)


def main() -> None:
    v13_body = checked_bytes(V13 / "REMOTE_SUBMISSION_V13.py", V13_BODY_PIN)
    checked_bytes(V13 / "SCORE_ADMISSION_SOURCE_BINDING_V13.json", V13_BINDING_PIN)
    checked_bytes(V13 / "ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V13.json", V13_REVIEW_PIN)
    checked_bytes(V13 / "V13_PAYLOAD_AND_INVERSE_PROOF.json", V13_PROOF_PIN)
    v13_files, v13_payloads = assignments(v13_body)

    expected_names = (
        "first_cell_score_contract.py",
        "first_cell_score_core.py",
        "first_cell_score_predictions.py",
        "score_first_cell.py",
    )
    v6_source: dict[str, bytes] = {}
    v7_source: dict[str, bytes] = {}
    v7_manifest = json.loads((V7 / "dependency_manifest.json").read_text(encoding="utf-8"))
    v7_pins = {
        item["path"]: (item["bytes"], item["sha256"]) for item in v7_manifest["scorer_files"]
    }
    for name in expected_names:
        old_raw = (V6 / name).read_bytes()
        if v13_files[name] != (len(old_raw), sha(old_raw)):
            raise ValueError(f"V13 declared source pin differs from its V6 source: {name}")
        if base64.b64decode(v13_payloads[name], validate=True) != old_raw:
            raise ValueError(f"V13 embedded scorer differs from its exact V6 source: {name}")
        v6_source[name] = old_raw
        new_raw = (V7 / name).read_bytes()
        if (len(new_raw), sha(new_raw)) != v7_pins[name]:
            raise ValueError(f"V7 scorer source differs from its V7 dependency manifest: {name}")
        v7_source[name] = new_raw

    receipt_sources = {
        "EXPORT_ACCEPTANCE.json": RECORDS / "export_v5/EXPORT_ACCEPTANCE.json",
        "EXPORT_EXACT_STOP.json": RECORDS / "export_v5/exact_stop/EXACT_STOP.json",
    }
    for name, expected in (
        ("EXPORT_ACCEPTANCE.json", EXPORT_ACCEPTANCE_PIN),
        ("EXPORT_EXACT_STOP.json", EXPORT_STOP_PIN),
    ):
        raw = checked_bytes(receipt_sources[name], expected)
        if base64.b64decode(v13_payloads[name], validate=True) != raw:
            raise ValueError(
                f"V13 embedded export authority differs from its accepted bytes: {name}"
            )

    body = v13_body
    body = replace_once(
        body,
        b'SCORE_D = SCRATCH_ROOT / "fg19_first_cell_f0_s1001_score_v2"',
        f'SCORE_D = SCRATCH_ROOT / "{NEW_SCORE_DIR}"'.encode(),
        "score directory",
    )
    body = replace_once(
        body,
        b'UNIT = "computer-vision-homework-fg19-first-cell-score-v2.service"',
        f'UNIT = "{NEW_UNIT}"'.encode(),
        "unit identity",
    )

    old_core = v13_files["first_cell_score_core.py"]
    new_core = v7_source["first_cell_score_core.py"]
    old_pin = (
        b'    "first_cell_score_core.py": (\n'
        + f"        {old_core[0]:_},\n".encode()
        + f'        "{old_core[1]}",\n'.encode()
        + b"    ),"
    )
    new_pin = (
        b'    "first_cell_score_core.py": (\n'
        + f"        {len(new_core):_},\n".encode()
        + f'        "{sha(new_core)}",\n'.encode()
        + b"    ),"
    )
    body = replace_once(body, old_pin, new_pin, "V7 core source pin")
    old_payload_line = (
        b"    'first_cell_score_core.py': '"
        + v13_payloads["first_cell_score_core.py"].encode("ascii")
        + b"',"
    )
    new_payload_line = b"    'first_cell_score_core.py': '" + base64.b64encode(new_core) + b"',"
    body = replace_once(body, old_payload_line, new_payload_line, "V7 core payload")

    current_files, current_payloads = assignments(body)
    expected_files = {name: (len(raw), sha(raw)) for name, raw in v7_source.items()}
    if current_files != expected_files:
        raise ValueError("V14 exact scorer pins do not match the four V7 files")
    if set(current_payloads) != set(expected_files) | set(receipt_sources):
        raise ValueError("V14 exact six-payload membership differs")
    for name, raw in v7_source.items():
        if base64.b64decode(current_payloads[name], validate=True) != raw:
            raise ValueError(f"V14 embedded scorer bytes differ: {name}")
    for name in expected_names:
        if name != "first_cell_score_core.py" and current_payloads[name] != v13_payloads[name]:
            raise ValueError(f"V14 changed an otherwise unchanged scorer payload: {name}")
    for name in receipt_sources:
        if current_payloads[name] != v13_payloads[name]:
            raise ValueError(f"V14 changed accepted export authority payload: {name}")

    restored = body
    restored = replace_once(
        restored,
        f'SCORE_D = SCRATCH_ROOT / "{NEW_SCORE_DIR}"'.encode(),
        b'SCORE_D = SCRATCH_ROOT / "fg19_first_cell_f0_s1001_score_v2"',
        "score-directory inverse",
    )
    restored = replace_once(
        restored,
        f'UNIT = "{NEW_UNIT}"'.encode(),
        b'UNIT = "computer-vision-homework-fg19-first-cell-score-v2.service"',
        "unit inverse",
    )
    restored = replace_once(restored, new_pin, old_pin, "V7 core pin inverse")
    restored = replace_once(restored, new_payload_line, old_payload_line, "V7 core payload inverse")
    if restored != v13_body:
        raise AssertionError("reversing V14's four assignments did not restore exact V13 bytes")

    proof = {
        "schema": "fg19.first-cell-score-admission-v14-source-payload-proof.v1",
        "status": "PASS_V14_V7_PAYLOAD_BINDING_AND_EXACT_INVERSE_NO_EXECUTION",
        "parent_v13_body": pin(
            V13 / "REMOTE_SUBMISSION_V13.py",
            "preparation/first_cell_evaluation_score_admission_v13/REMOTE_SUBMISSION_V13.py",
        ),
        "v14_body": {"path": "REMOTE_SUBMISSION_V14.py", "bytes": len(body), "sha256": sha(body)},
        "exact_delta": {
            "changed_assignments": [
                "SCORE_D",
                "UNIT",
                "SCORER_FILES.first_cell_score_core.py",
                "PAYLOAD_B64.first_cell_score_core.py",
            ],
            "new_unit": NEW_UNIT,
            "new_private_directory": NEW_SCORE_DIR,
            "new_core": {"bytes": len(new_core), "sha256": sha(new_core)},
            "other_three_scorer_payloads_byte_identical_to_v13": True,
            "both_accepted_export_authority_payloads_byte_identical_to_v13": True,
            "exact_inverse_restores_v13_body": True,
        },
        "bound_payloads": {
            name: {
                "bytes": len(base64.b64decode(encoded, validate=True)),
                "sha256": sha(base64.b64decode(encoded, validate=True)),
            }
            for name, encoded in current_payloads.items()
        },
        "v7_source_binding": {
            "dependency_manifest": pin(
                V7 / "dependency_manifest.json",
                "preparation/first_cell_evaluation_score_v7/dependency_manifest.json",
            ),
            "source_review": {
                "path": "preparation/first_cell_evaluation_score_v7/SCORE_SOURCE_REVIEW_V7.json",
                "status": "PENDING_INDEPENDENT_V7_SOURCE_REVIEW",
                "bytes": None,
                "sha256": None,
            },
        },
        "boundary": {
            "remote_body_executed": False,
            "b_or_ssh_action": False,
            "model_cache_array_or_prediction_payload_read_locally": False,
            "scoring_or_result_read": False,
        },
    }
    body_path = PACKET / "REMOTE_SUBMISSION_V14.py"
    proof_path = PACKET / "V14_PAYLOAD_AND_INVERSE_PROOF.json"
    if body_path.exists() or proof_path.exists():
        raise FileExistsError("V14 body/proof output already exists; preserve frozen bytes")
    with body_path.open("xb") as stream:
        stream.write(body)
        stream.flush()
    proof_raw = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with proof_path.open("xb") as stream:
        stream.write(proof_raw)
        stream.flush()
    print(
        json.dumps(
            {
                "body": {"path": str(body_path), "bytes": len(body), "sha256": sha(body)},
                "proof": pin(proof_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
