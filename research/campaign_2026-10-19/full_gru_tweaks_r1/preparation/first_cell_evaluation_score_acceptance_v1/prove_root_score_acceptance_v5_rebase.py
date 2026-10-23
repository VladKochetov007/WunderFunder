"""Prove V5 is only the approved ROOT-record and admission-source rebase."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
PREP = (
    ROOT
    / "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_acceptance_v1"
)
V4 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V4.py"
V5 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V5.py"
PROOF = PREP / "ROOT_CALLER_V5_LITERAL_REBASE_PROOF.json"
V4_PIN = (31_750, "074cb822cbcb16a5b25ec28cbeb7f8dbd4d2395340c125ed8ee298b2a0584e17")
EXPECTED = {
    "RECORD": (
        "records/first_cell_evaluation_v1/score_v1",
        "records/first_cell_evaluation_v1/score_v2",
    ),
    "SCORE_ADMISSION_BODY": (
        "preparation/first_cell_evaluation_score_admission_v11/REMOTE_SUBMISSION_V11.py",
        "preparation/first_cell_evaluation_score_admission_v12/REMOTE_SUBMISSION_V12.py",
    ),
    "SCORE_ADMISSION_REVIEW": (
        "preparation/first_cell_evaluation_score_admission_v11/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V11.json",
        "preparation/first_cell_evaluation_score_admission_v12/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V12.json",
    ),
    "SCORE_ADMISSION_REVIEW_SCHEMA": (
        "fg19.first-cell-score-admission-actual-source-review.v11",
        "fg19.first-cell-score-admission-actual-source-review.v12",
    ),
    "SCORE_ADMISSION_REVIEW_STATUS": (
        "PASS_SOURCE_ONLY_ACTUAL_SCORE_BODY_V11",
        "PASS_SOURCE_ONLY_ACTUAL_SCORE_BODY_V12",
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
                raise ValueError(f"duplicate top-level source binding: {name}")
            result[name] = node
    if set(result) != set(EXPECTED):
        raise ValueError("caller source-binding inventory differs")
    return result


def _literal(node: ast.expr) -> ast.Constant:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and isinstance(node.left, ast.Name)
        and node.left.id == "C"
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, str)
    ):
        return node.right
    raise ValueError("expected a string literal or fixed campaign path")


def main() -> None:
    old_raw, new_raw = V4.read_bytes(), V5.read_bytes()
    if (len(old_raw), sha(old_raw)) != V4_PIN:
        raise ValueError("frozen V4 caller pin differs")
    old_tree = ast.parse(old_raw.decode("utf-8", errors="strict"))
    new_tree = ast.parse(new_raw.decode("utf-8", errors="strict"))
    old_assignments = _assignments(old_tree)
    new_assignments = _assignments(new_tree)
    changed: list[str] = []
    for name, (old_value, new_value) in EXPECTED.items():
        old_literal = _literal(old_assignments[name].value)
        new_literal = _literal(new_assignments[name].value)
        if old_literal.value != old_value or new_literal.value != new_value:
            raise ValueError(f"caller rebase literal differs from plan: {name}")
        if old_value != new_value:
            changed.append(name)
            new_literal.value = old_value
    if set(changed) != set(EXPECTED):
        raise ValueError("V5 did not change exactly the five approved literals")
    if ast.dump(old_tree, include_attributes=False) != ast.dump(new_tree, include_attributes=False):
        raise ValueError("V5 caller AST differs beyond the five approved literals")

    proof = {
        "schema": "fg19.first-cell-score-root-caller-v5-literal-rebase.v1",
        "status": "PASS_EXACT_SCORE_RECORD_AND_V12_SOURCE_REBASE",
        "source": {
            "v4": {
                "path": str(V4.relative_to(ROOT)),
                "bytes": len(old_raw),
                "sha256": V4_PIN[1],
            },
            "v5": {
                "path": str(V5.relative_to(ROOT)),
                "bytes": len(new_raw),
                "sha256": sha(new_raw),
            },
        },
        "changed_assignments": changed,
        "literal_replacements": {
            name: {"from": before, "to": after} for name, (before, after) in EXPECTED.items()
        },
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
