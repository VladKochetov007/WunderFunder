"""Prove caller V7 binds the reviewed score-v3 source chain without execution."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
C = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1"
PREP = C / "preparation/first_cell_evaluation_score_acceptance_v1"
V6 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V6.py"
V7 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V7.py"
PROOF = PREP / "ROOT_CALLER_V7_NAMESPACE_PROOF_V2.json"
V6_PIN = (31_837, "f669e345738b4159f5585d24724afbd375c4bcaea29076022d60c076f06e534f")
EXPECTED_V7 = {
    "RECORD": "C / 'records/first_cell_evaluation_v1/score_v4'",
    "HELPER": "C / 'preparation/first_cell_evaluation_score_acceptance_v3/score_result_acceptance_v3.py'",
    "HELPER_PIN": "(46448, '72f25104d5a338e55d3d7667df79882cd699d0798c7d8fc19638fea392ad0e8c')",
    "HELPER_REVIEW": "C / 'preparation/first_cell_evaluation_score_acceptance_v3/HELPER_SOURCE_REVIEW_V3.json'",
    "HELPER_REVIEW_PIN": "(8451, '9ad6a92a06e1c481666f11446fbfc9eeb741e31769f0c016370f44442dcc9fed')",
    "HELPER_REVIEW_SCHEMA": "'fg19.first-cell-score-acceptance-helper-source-review.v3'",
    "HELPER_REVIEW_STATUS": "'PASS_SOURCE_ONLY_SCORE_ACCEPTANCE_HELPER_V3'",
    "READER_TEMPLATE": "C / 'preparation/first_cell_evaluation_score_capture_v6/SCORE_RESULT_CAPTURE_TEMPLATE_V6.py'",
    "READER_TEMPLATE_PIN": "(26354, '049e2e9245356a4ac5f39596348e186d7af9a1e04aeb747255b7f62843324de0')",
    "READER_REVIEW": "C / 'preparation/first_cell_evaluation_score_capture_v6/INDEPENDENT_SOURCE_REVIEW_V6.json'",
    "READER_REVIEW_PIN": "(9525, '844fb16d236a6778179f55edfe37a553809097691b19060869309c0c07bbeeec')",
    "READER_REVIEW_SCHEMA": "'fg19.first-cell-score-result-capture-source-review.v6'",
    "READER_REVIEW_STATUS": "'PASS_SOURCE_ONLY_READER_V6_HARD_BOUNDED_UNBOUND'",
    "ROOT_CAPTURE_DRIVER": "C / 'preparation/first_cell_evaluation_score_capture_v2/ROOT_CAPTURE_SCORE_STATUS_V5.py'",
    "ROOT_CAPTURE_DRIVER_PIN": "(8895, '7367fa5000d43f4ca66ea8fae06d250cec54dde0f8ef58fa5d5287cad869747c')",
    "ROOT_CAPTURE_DRIVER_REVIEW": "C / 'preparation/first_cell_evaluation_score_capture_v2/ROOT_SCORE_STATUS_DRIVER_SOURCE_REVIEW_V5.json'",
    "ROOT_CAPTURE_DRIVER_REVIEW_PIN": "(6932, 'fad67e3759f7c115842766d782eecf9d67088b60b97063761a62356638d21181')",
    "ROOT_CAPTURE_DRIVER_REVIEW_SCHEMA": "'fg19.first-cell-score-root-status-driver-source-review.v5'",
    "ROOT_CAPTURE_DRIVER_REVIEW_STATUS": "'PASS_SOURCE_ONLY_ROOT_STATUS_DRIVER_V5_REBASED_TO_CURRENT_READER_V6'",
    "SCORE_ADMISSION_BODY": "C / 'preparation/first_cell_evaluation_score_admission_v14/REMOTE_SUBMISSION_V14.py'",
    "SCORE_ADMISSION_REVIEW": "C / 'preparation/first_cell_evaluation_score_admission_v14/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V14.json'",
    "SCORE_ADMISSION_REVIEW_SCHEMA": "'fg19.first-cell-score-admission-actual-source-review.v14'",
    "SCORE_ADMISSION_REVIEW_STATUS": "'PASS_SOURCE_ONLY_ACTUAL_SCORE_BODY_V14'",
    "SCORE_UNIT": "'computer-vision-homework-fg19-first-cell-score-v3.service'",
}
NEW_PIN_ASSIGNMENTS = {
    "SCORE_ADMISSION_BODY_PIN": "(193262, 'fea0734c5485fa984e0bb961db58dafc01950c1308a3893dd400d15d7c332670')",
    "SCORE_ADMISSION_REVIEW_PIN": "(18342, '4cd8c964d4bfb59032e9e6b79cfd63777417d0f96d0031ae9f2cd01f21227a82')",
}
STRING_REBASE = {
    "reader review does not identify the exact V6 template": "reader review does not identify the exact V4 template",
    "captured REMOTE.py is not the exact five-literal V6 reader binding": "captured REMOTE.py is not the exact five-literal V4 reader binding",
    "reviewed V6 bounded score reader template pin differs": "reviewed V4 bounded score reader template pin differs",
}
FIXED_DEPENDENCIES = {
    "helper": (
        C / "preparation/first_cell_evaluation_score_acceptance_v3/score_result_acceptance_v3.py",
        46_448,
        "72f25104d5a338e55d3d7667df79882cd699d0798c7d8fc19638fea392ad0e8c",
    ),
    "helper_review": (
        C / "preparation/first_cell_evaluation_score_acceptance_v3/HELPER_SOURCE_REVIEW_V3.json",
        8_451,
        "9ad6a92a06e1c481666f11446fbfc9eeb741e31769f0c016370f44442dcc9fed",
    ),
    "reader": (
        C / "preparation/first_cell_evaluation_score_capture_v6/SCORE_RESULT_CAPTURE_TEMPLATE_V6.py",
        26_354,
        "049e2e9245356a4ac5f39596348e186d7af9a1e04aeb747255b7f62843324de0",
    ),
    "reader_review": (
        C / "preparation/first_cell_evaluation_score_capture_v6/INDEPENDENT_SOURCE_REVIEW_V6.json",
        9_525,
        "844fb16d236a6778179f55edfe37a553809097691b19060869309c0c07bbeeec",
    ),
    "status_driver": (
        C / "preparation/first_cell_evaluation_score_capture_v2/ROOT_CAPTURE_SCORE_STATUS_V5.py",
        8_895,
        "7367fa5000d43f4ca66ea8fae06d250cec54dde0f8ef58fa5d5287cad869747c",
    ),
    "status_driver_review": (
        C / "preparation/first_cell_evaluation_score_capture_v2/ROOT_SCORE_STATUS_DRIVER_SOURCE_REVIEW_V5.json",
        6_932,
        "fad67e3759f7c115842766d782eecf9d67088b60b97063761a62356638d21181",
    ),
    "score_admission_body": (
        C / "preparation/first_cell_evaluation_score_admission_v14/REMOTE_SUBMISSION_V14.py",
        193_262,
        "fea0734c5485fa984e0bb961db58dafc01950c1308a3893dd400d15d7c332670",
    ),
    "score_admission_review": (
        C / "preparation/first_cell_evaluation_score_admission_v14/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V14.json",
        18_342,
        "4cd8c964d4bfb59032e9e6b79cfd63777417d0f96d0031ae9f2cd01f21227a82",
    ),
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def assignments(tree: ast.Module) -> dict[str, ast.Assign]:
    return {
        node.targets[0].id: node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }


def main() -> None:
    old_raw, new_raw = V6.read_bytes(), V7.read_bytes()
    if (len(old_raw), sha(old_raw)) != V6_PIN:
        raise ValueError("frozen caller V6 pin differs")
    old_tree = ast.parse(old_raw.decode("utf-8", errors="strict"))
    new_tree = ast.parse(new_raw.decode("utf-8", errors="strict"))
    old_assigns, new_assigns = assignments(old_tree), assignments(new_tree)
    for name, expected in EXPECTED_V7.items():
        if name not in old_assigns or name not in new_assigns:
            raise ValueError(f"caller assignment missing: {name}")
        if ast.unparse(new_assigns[name].value) != expected:
            raise ValueError(f"V7 source binding differs: {name}")
        new_assigns[name].value = copy.deepcopy(old_assigns[name].value)
    for name, expected in NEW_PIN_ASSIGNMENTS.items():
        if name not in new_assigns or ast.unparse(new_assigns[name].value) != expected:
            raise ValueError(f"V14 source pin differs: {name}")
        new_tree.body.remove(new_assigns[name])

    for expected_new, expected_old in STRING_REBASE.items():
        matches = [
            node for node in ast.walk(new_tree)
            if isinstance(node, ast.Constant) and node.value == expected_new
        ]
        if len(matches) != 1:
            raise ValueError(f"reader error-message rebase differs: {expected_new}")
        matches[0].value = expected_old

    main_functions = [
        node for node in new_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    ]
    if len(main_functions) != 1:
        raise ValueError("caller main function inventory differs")
    guard_source = """if (args.score_body_bytes, args.score_body_sha256) != SCORE_ADMISSION_BODY_PIN or (
    args.score_review_bytes, args.score_review_sha256
) != SCORE_ADMISSION_REVIEW_PIN:
    _fail("score-admission body/review CLI pins differ from frozen V14 source pins")
"""
    guard = ast.parse(guard_source).body[0]
    guard_matches = [
        node for node in main_functions[0].body
        if isinstance(node, ast.If)
        and ast.dump(node, include_attributes=False) == ast.dump(guard, include_attributes=False)
    ]
    if len(guard_matches) != 1:
        raise ValueError("exact fixed V14 source-pin guard is missing or duplicated")
    main_functions[0].body.remove(guard_matches[0])

    dependency_proofs = {}
    for name, (path, byte_count, digest) in FIXED_DEPENDENCIES.items():
        raw = path.read_bytes()
        if (len(raw), sha(raw)) != (byte_count, digest):
            raise ValueError(f"fixed source dependency pin differs: {name}")
        dependency_proofs[name] = {
            "path": str(path.relative_to(ROOT)),
            "bytes": byte_count,
            "sha256": digest,
        }
    for path, expected_schema, expected_status in (
        (
            FIXED_DEPENDENCIES["helper_review"][0],
            "fg19.first-cell-score-acceptance-helper-source-review.v3",
            "PASS_SOURCE_ONLY_SCORE_ACCEPTANCE_HELPER_V3",
        ),
        (
            FIXED_DEPENDENCIES["reader_review"][0],
            "fg19.first-cell-score-result-capture-source-review.v6",
            "PASS_SOURCE_ONLY_READER_V6_HARD_BOUNDED_UNBOUND",
        ),
        (
            FIXED_DEPENDENCIES["status_driver_review"][0],
            "fg19.first-cell-score-root-status-driver-source-review.v5",
            "PASS_SOURCE_ONLY_ROOT_STATUS_DRIVER_V5_REBASED_TO_CURRENT_READER_V6",
        ),
        (
            FIXED_DEPENDENCIES["score_admission_review"][0],
            "fg19.first-cell-score-admission-actual-source-review.v14",
            "PASS_SOURCE_ONLY_ACTUAL_SCORE_BODY_V14",
        ),
    ):
        review = json.loads(path.read_bytes())
        if review.get("schema") != expected_schema or review.get("status") != expected_status:
            raise ValueError(f"fixed source review schema/status differs: {path.name}")
    if ast.dump(old_tree, include_attributes=False) != ast.dump(
        new_tree, include_attributes=False
    ):
        raise ValueError("caller V7 differs outside declared source-chain bindings")

    proof = {
        "schema": "fg19.first-cell-score-root-caller-v7-namespace-proof.v2",
        "status": "PASS_SCORE_V3_CALLER_BINDS_CURRENT_REVIEWED_SOURCE_CHAIN",
        "predecessor": {
            "path": str(V6.relative_to(ROOT)),
            "bytes": len(old_raw),
            "sha256": V6_PIN[1],
        },
        "candidate": {
            "path": str(V7.relative_to(ROOT)),
            "bytes": len(new_raw),
            "sha256": sha(new_raw),
        },
        "fixed_source_chain": dependency_proofs,
        "namespace_bindings": list(EXPECTED_V7),
        "additional_exact_changes": [
            "two fixed V14 source pins",
            "one V14 CLI source-pin guard",
            "three reader-version error messages",
        ],
        "all_other_ast_identical": True,
        "caller_or_reader_executed": False,
        "tests_or_unit_queries_performed": False,
        "b_or_ssh_attempted": False,
        "actual_score_or_quality_claimed": False,
    }
    raw = (json.dumps(proof, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with PROOF.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(PREP, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(f"{len(raw)} {sha(raw)} {PROOF}")


if __name__ == "__main__":
    main()
