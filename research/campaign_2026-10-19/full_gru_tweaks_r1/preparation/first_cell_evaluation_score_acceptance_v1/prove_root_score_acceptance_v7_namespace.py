"""Prove the unbound V7 caller is a score-v3 namespace/review-chain fork."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
PREP = (
    ROOT
    / "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_acceptance_v1"
)
V6 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V6.py"
V7 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V7.py"
PROOF = PREP / "ROOT_CALLER_V7_NAMESPACE_PROOF.json"
V6_PIN = (31_837, "f669e345738b4159f5585d24724afbd375c4bcaea29076022d60c076f06e534f")

EXPECTED = {
    "RECORD": (
        "C / 'records/first_cell_evaluation_v1/score_v3'",
        "C / 'records/first_cell_evaluation_v1/score_v4'",
    ),
    "HELPER": (
        "C / 'preparation/first_cell_evaluation_score_acceptance_v2/score_result_acceptance_v2.py'",
        "C / 'preparation/first_cell_evaluation_score_acceptance_v3/score_result_acceptance_v3.py'",
    ),
    "HELPER_PIN": (
        "(46448, '1b8a6e899dcd2c250eecf096bd1797aa77466d32eedd598b063f48249490f446')",
        "(-1, 'UNBOUND_SCORE_ACCEPTANCE_HELPER_V3_PIN')",
    ),
    "HELPER_REVIEW": (
        "C / 'preparation/first_cell_evaluation_score_acceptance_v2/HELPER_SOURCE_REVIEW_V2.json'",
        "C / 'preparation/first_cell_evaluation_score_acceptance_v3/HELPER_SOURCE_REVIEW_V3.json'",
    ),
    "HELPER_REVIEW_PIN": (
        "(10059, 'c4d0c968868c4c0977d961bd8e0dfe48c9d368e46224e03edbdd87c8d335223c')",
        "(0, 'UNBOUND_SCORE_ACCEPTANCE_HELPER_V3_REVIEW_PIN')",
    ),
    "HELPER_REVIEW_SCHEMA": (
        "'fg19.first-cell-score-acceptance-helper-source-review.v2'",
        "'UNBOUND_SCORE_ACCEPTANCE_HELPER_V3_REVIEW_SCHEMA'",
    ),
    "HELPER_REVIEW_STATUS": (
        "'PASS_SOURCE_ONLY_SCORE_ACCEPTANCE_HELPER_V2'",
        "'UNBOUND_SCORE_ACCEPTANCE_HELPER_V3_REVIEW_STATUS'",
    ),
    "READER_TEMPLATE": (
        "C / 'preparation/first_cell_evaluation_score_capture_v5/SCORE_RESULT_CAPTURE_TEMPLATE_V5.py'",
        "C / 'preparation/first_cell_evaluation_score_capture_v6/SCORE_RESULT_CAPTURE_TEMPLATE_V6.py'",
    ),
    "READER_TEMPLATE_PIN": (
        "(26354, '8d3c37f89376a23db1a548001989c11dd7eb5b83720061c4b229e59f75d1f69c')",
        "(-1, 'UNBOUND_SCORE_RESULT_CAPTURE_V6_PIN')",
    ),
    "READER_REVIEW": (
        "C / 'preparation/first_cell_evaluation_score_capture_v5/INDEPENDENT_SOURCE_REVIEW_V5.json'",
        "C / 'preparation/first_cell_evaluation_score_capture_v6/INDEPENDENT_SOURCE_REVIEW_V6.json'",
    ),
    "READER_REVIEW_PIN": (
        "(9206, '6be421f63040c941368af2c08670b22ce1699434a8b264c4f100dd253e229b95')",
        "(-1, 'UNBOUND_SCORE_RESULT_CAPTURE_V6_REVIEW_PIN')",
    ),
    "READER_REVIEW_SCHEMA": (
        "'fg19.first-cell-score-result-capture-source-review.v5'",
        "'UNBOUND_SCORE_RESULT_CAPTURE_V6_REVIEW_SCHEMA'",
    ),
    "READER_REVIEW_STATUS": (
        "'PASS_SOURCE_ONLY_READER_V5_HARD_BOUNDED_UNBOUND'",
        "'UNBOUND_SCORE_RESULT_CAPTURE_V6_REVIEW_STATUS'",
    ),
    "ROOT_CAPTURE_DRIVER": (
        "C / 'preparation/first_cell_evaluation_score_capture_v2/ROOT_CAPTURE_SCORE_STATUS_V3.py'",
        "C / 'preparation/first_cell_evaluation_score_capture_v2/ROOT_CAPTURE_SCORE_STATUS_V4.py'",
    ),
    "ROOT_CAPTURE_DRIVER_PIN": (
        "(8895, 'c540db668e371355dd11851f828e8d34b394037fe106d97bcd9477e30db0f03d')",
        "(-1, 'UNBOUND_ROOT_SCORE_STATUS_DRIVER_V4_PIN')",
    ),
    "ROOT_CAPTURE_DRIVER_REVIEW": (
        "C / 'preparation/first_cell_evaluation_score_capture_v2/ROOT_SCORE_STATUS_DRIVER_SOURCE_REVIEW_V3.json'",
        "C / 'preparation/first_cell_evaluation_score_capture_v2/ROOT_SCORE_STATUS_DRIVER_SOURCE_REVIEW_V4.json'",
    ),
    "ROOT_CAPTURE_DRIVER_REVIEW_PIN": (
        "(6131, '7fa00af508374a366c0e94b58526c3c2e0a173fe3a5c53af03b9b793c752cdad')",
        "(-1, 'UNBOUND_ROOT_SCORE_STATUS_DRIVER_V4_REVIEW_PIN')",
    ),
    "ROOT_CAPTURE_DRIVER_REVIEW_SCHEMA": (
        "'fg19.first-cell-score-root-status-driver-source-review.v3'",
        "'UNBOUND_ROOT_SCORE_STATUS_DRIVER_V4_REVIEW_SCHEMA'",
    ),
    "ROOT_CAPTURE_DRIVER_REVIEW_STATUS": (
        "'PASS_SOURCE_ONLY_ROOT_STATUS_DRIVER_V3_REBASED_TO_REVIEWED_V5'",
        "'UNBOUND_ROOT_SCORE_STATUS_DRIVER_V4_REVIEW_STATUS'",
    ),
    "SCORE_ADMISSION_BODY": (
        "C / 'preparation/first_cell_evaluation_score_admission_v13/REMOTE_SUBMISSION_V13.py'",
        "C / 'preparation/first_cell_evaluation_score_admission_v14/REMOTE_SUBMISSION_V14.py'",
    ),
    "SCORE_ADMISSION_REVIEW": (
        "C / 'preparation/first_cell_evaluation_score_admission_v13/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V13.json'",
        "C / 'preparation/first_cell_evaluation_score_admission_v14/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V14.json'",
    ),
    "SCORE_ADMISSION_REVIEW_SCHEMA": (
        "'fg19.first-cell-score-admission-actual-source-review.v13'",
        "'fg19.first-cell-score-admission-actual-source-review.v14'",
    ),
    "SCORE_ADMISSION_REVIEW_STATUS": (
        "'PASS_SOURCE_ONLY_ACTUAL_SCORE_BODY_V13'",
        "'PASS_SOURCE_ONLY_ACTUAL_SCORE_BODY_V14'",
    ),
    "SCORE_UNIT": (
        "'computer-vision-homework-fg19-first-cell-score-v2.service'",
        "'computer-vision-homework-fg19-first-cell-score-v3.service'",
    ),
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def bindings(tree: ast.Module) -> dict[str, ast.Assign]:
    result: dict[str, ast.Assign] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in EXPECTED
        ):
            name = node.targets[0].id
            if name in result:
                raise ValueError(f"duplicate namespace assignment: {name}")
            result[name] = node
    if set(result) != set(EXPECTED):
        raise ValueError("caller namespace assignment inventory differs")
    return result


def main() -> None:
    old_raw, new_raw = V6.read_bytes(), V7.read_bytes()
    if (len(old_raw), sha(old_raw)) != V6_PIN:
        raise ValueError("frozen caller V6 pin differs")
    old_tree = ast.parse(old_raw.decode("utf-8", errors="strict"))
    new_tree = ast.parse(new_raw.decode("utf-8", errors="strict"))
    old_bindings, new_bindings = bindings(old_tree), bindings(new_tree)
    for name, (old_value, new_value) in EXPECTED.items():
        if ast.unparse(old_bindings[name].value) != old_value:
            raise ValueError(f"frozen V6 namespace differs: {name}")
        if ast.unparse(new_bindings[name].value) != new_value:
            raise ValueError(f"V7 namespace differs: {name}")
        new_bindings[name].value = copy.deepcopy(old_bindings[name].value)
    if ast.dump(old_tree, include_attributes=False) != ast.dump(
        new_tree, include_attributes=False
    ):
        raise ValueError("caller V7 differs outside approved namespace bindings")
    proof = {
        "schema": "fg19.first-cell-score-root-caller-v7-namespace-proof.v1",
        "status": "PASS_UNIT_V3_SCORE_V4_RECORD_FORK_REVIEWS_UNBOUND",
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
        "changed_assignments": list(EXPECTED),
        "all_other_ast_identical": True,
        "v7_core_and_reviews_bound": False,
        "caller_imported_or_executed": False,
        "b_or_ssh_attempted": False,
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
