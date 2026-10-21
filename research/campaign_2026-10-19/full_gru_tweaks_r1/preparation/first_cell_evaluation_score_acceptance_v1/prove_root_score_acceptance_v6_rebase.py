"""Prove V6 is only the approved score-v3/review-chain literal rebase."""

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
V5 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V5.py"
V6 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V6.py"
PROOF = PREP / "ROOT_CALLER_V6_LITERAL_REBASE_PROOF.json"
V5_PIN = (31_750, "35ddb4711f267eaf05f1df10c92acf613ab33d1868f57c43fd59abcf98ea2858")

EXPECTED = {
    "RECORD": (
        "C / 'records/first_cell_evaluation_v1/score_v2'",
        "C / 'records/first_cell_evaluation_v1/score_v3'",
    ),
    "HELPER": (
        "PREP / 'score_result_acceptance.py'",
        "C / 'preparation/first_cell_evaluation_score_acceptance_v2/score_result_acceptance_v2.py'",
    ),
    "HELPER_PIN": (
        "(45795, '5987d527382f0ce33e72b4c17ae020d03485df3057b398ae285940ead29d528b')",
        "(46448, '1b8a6e899dcd2c250eecf096bd1797aa77466d32eedd598b063f48249490f446')",
    ),
    "HELPER_REVIEW": (
        "PREP / 'SOURCE_REVIEW_V1.json'",
        "C / 'preparation/first_cell_evaluation_score_acceptance_v2/HELPER_SOURCE_REVIEW_V2.json'",
    ),
    "HELPER_REVIEW_PIN": (
        "(5695, '31dd246873448a214f772be5793b13cff12b7fcb1d79d4cdc64009fcaf8a56f0')",
        "(10059, 'c4d0c968868c4c0977d961bd8e0dfe48c9d368e46224e03edbdd87c8d335223c')",
    ),
    "HELPER_REVIEW_SCHEMA": (
        "'fg19.first-cell-score-result-acceptance-source-review.v1'",
        "'fg19.first-cell-score-acceptance-helper-source-review.v2'",
    ),
    "HELPER_REVIEW_STATUS": (
        "'PASS_SOURCE_ONLY_ROOT_RESULT_ACCEPTANCE_V1'",
        "'PASS_SOURCE_ONLY_SCORE_ACCEPTANCE_HELPER_V2'",
    ),
    "READER_TEMPLATE": (
        "C / 'preparation/first_cell_evaluation_score_capture_v4/SCORE_RESULT_CAPTURE_TEMPLATE_V4.py'",
        "C / 'preparation/first_cell_evaluation_score_capture_v5/SCORE_RESULT_CAPTURE_TEMPLATE_V5.py'",
    ),
    "READER_TEMPLATE_PIN": (
        "(26354, '88968f6dd73d4b489c4fc57308906c5e390137a49389ab991408408f353b9be6')",
        "(26354, '8d3c37f89376a23db1a548001989c11dd7eb5b83720061c4b229e59f75d1f69c')",
    ),
    "READER_REVIEW": (
        "C / 'preparation/first_cell_evaluation_score_capture_v4/INDEPENDENT_SOURCE_REVIEW_V4.json'",
        "C / 'preparation/first_cell_evaluation_score_capture_v5/INDEPENDENT_SOURCE_REVIEW_V5.json'",
    ),
    "READER_REVIEW_PIN": (
        "(5655, 'b05b06a6e56888348e30ebe0ded0272a5e9f3ca709037feb975b938b4d0bb7d0')",
        "(9206, '6be421f63040c941368af2c08670b22ce1699434a8b264c4f100dd253e229b95')",
    ),
    "READER_REVIEW_SCHEMA": (
        "'fg19.first-cell-score-result-capture-source-review.v4'",
        "'fg19.first-cell-score-result-capture-source-review.v5'",
    ),
    "READER_REVIEW_STATUS": (
        "'PASS_SOURCE_ONLY_READER_V4_HARD_BOUNDED_UNBOUND'",
        "'PASS_SOURCE_ONLY_READER_V5_HARD_BOUNDED_UNBOUND'",
    ),
    "ROOT_CAPTURE_DRIVER": (
        "C / 'preparation/first_cell_evaluation_score_capture_v2/ROOT_CAPTURE_SCORE_STATUS_V2.py'",
        "C / 'preparation/first_cell_evaluation_score_capture_v2/ROOT_CAPTURE_SCORE_STATUS_V3.py'",
    ),
    "ROOT_CAPTURE_DRIVER_PIN": (
        "(8895, 'e7b479e8237e6ca1f23417ce7a685c3e08fc4bb772fcfc33775634e680f8748c')",
        "(8895, 'c540db668e371355dd11851f828e8d34b394037fe106d97bcd9477e30db0f03d')",
    ),
    "ROOT_CAPTURE_DRIVER_REVIEW": (
        "C / 'preparation/first_cell_evaluation_score_capture_v2/ROOT_SCORE_STATUS_DRIVER_SOURCE_REVIEW_V2.json'",
        "C / 'preparation/first_cell_evaluation_score_capture_v2/ROOT_SCORE_STATUS_DRIVER_SOURCE_REVIEW_V3.json'",
    ),
    "ROOT_CAPTURE_DRIVER_REVIEW_PIN": (
        "(5662, 'ae7d68b74b61423d4ef15f93ca9e1c30581287229fd03d9028c2a80ee54b264e')",
        "(6131, '7fa00af508374a366c0e94b58526c3c2e0a173fe3a5c53af03b9b793c752cdad')",
    ),
    "ROOT_CAPTURE_DRIVER_REVIEW_SCHEMA": (
        "'fg19.first-cell-score-root-status-driver-source-review.v2'",
        "'fg19.first-cell-score-root-status-driver-source-review.v3'",
    ),
    "ROOT_CAPTURE_DRIVER_REVIEW_STATUS": (
        "'PASS_SOURCE_ONLY_SCORE_STATUS_DRIVER_V2_PINNED_READER_V4_CALLER_PATH_CONSTRAINED'",
        "'PASS_SOURCE_ONLY_ROOT_STATUS_DRIVER_V3_REBASED_TO_REVIEWED_V5'",
    ),
    "SCORE_ADMISSION_BODY": (
        "C / 'preparation/first_cell_evaluation_score_admission_v12/REMOTE_SUBMISSION_V12.py'",
        "C / 'preparation/first_cell_evaluation_score_admission_v13/REMOTE_SUBMISSION_V13.py'",
    ),
    "SCORE_ADMISSION_REVIEW": (
        "C / 'preparation/first_cell_evaluation_score_admission_v12/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V12.json'",
        "C / 'preparation/first_cell_evaluation_score_admission_v13/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V13.json'",
    ),
    "SCORE_ADMISSION_REVIEW_SCHEMA": (
        "'fg19.first-cell-score-admission-actual-source-review.v12'",
        "'fg19.first-cell-score-admission-actual-source-review.v13'",
    ),
    "SCORE_ADMISSION_REVIEW_STATUS": (
        "'PASS_SOURCE_ONLY_ACTUAL_SCORE_BODY_V12'",
        "'PASS_SOURCE_ONLY_ACTUAL_SCORE_BODY_V13'",
    ),
    "SCORE_UNIT": (
        "'computer-vision-homework-fg19-first-cell-score-v1.service'",
        "'computer-vision-homework-fg19-first-cell-score-v2.service'",
    ),
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _assignments(tree: ast.Module) -> dict[str, ast.Assign]:
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
                raise ValueError(f"duplicate source binding: {name}")
            result[name] = node
    if set(result) != set(EXPECTED):
        raise ValueError("caller source-binding inventory differs")
    return result


def main() -> None:
    old_raw, new_raw = V5.read_bytes(), V6.read_bytes()
    if (len(old_raw), sha(old_raw)) != V5_PIN:
        raise ValueError("frozen V5 caller pin differs")
    old_tree = ast.parse(old_raw.decode("utf-8", errors="strict"))
    new_tree = ast.parse(new_raw.decode("utf-8", errors="strict"))
    old_assignments = _assignments(old_tree)
    new_assignments = _assignments(new_tree)
    for name, (old_value, new_value) in EXPECTED.items():
        old_node = old_assignments[name].value
        new_node = new_assignments[name].value
        if ast.unparse(old_node) != old_value or ast.unparse(new_node) != new_value:
            raise ValueError(f"approved literal differs: {name}")
        new_assignments[name].value = copy.deepcopy(old_node)
    if ast.dump(old_tree, include_attributes=False) != ast.dump(
        new_tree, include_attributes=False
    ):
        raise ValueError("V6 caller AST differs beyond approved literal rebases")

    proof = {
        "schema": "fg19.first-cell-score-root-caller-v6-literal-rebase.v1",
        "status": "PASS_EXACT_SCORE_V3_REVIEW_CHAIN_REBASE",
        "source": {
            "v5": {
                "path": str(V5.relative_to(ROOT)),
                "bytes": len(old_raw),
                "sha256": V5_PIN[1],
            },
            "v6": {
                "path": str(V6.relative_to(ROOT)),
                "bytes": len(new_raw),
                "sha256": sha(new_raw),
            },
        },
        "changed_assignments": list(EXPECTED),
        "all_other_ast_identical": True,
        "caller_imported_or_executed": False,
        "b_or_ssh_attempted": False,
        "historical_versions_preserved": True,
    }
    proof_raw = (json.dumps(proof, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with PROOF.open("xb") as stream:
        stream.write(proof_raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(PREP, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(f"{len(proof_raw)} {sha(proof_raw)} {PROOF}")


if __name__ == "__main__":
    main()
