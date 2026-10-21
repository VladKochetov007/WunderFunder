"""Prove the V3 caller changes only its V10-to-V11 source-review bindings."""

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
V2 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V2.py"
V3 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V3.py"
PROOF = PREP / "ROOT_CALLER_V3_LITERAL_REBASE_PROOF.json"
V2_PIN = (
    31_177,
    "eba267c9a626d9384b5006ce3d285defa1650e805cf4f03dda099d134a890eeb",
)
EXPECTED_REPLACEMENTS = {
    "SCORE_ADMISSION_BODY": (
        'C / "preparation/first_cell_evaluation_score_admission_v10/REMOTE_SUBMISSION_V10.py"',
        'C / "preparation/first_cell_evaluation_score_admission_v11/REMOTE_SUBMISSION_V11.py"',
    ),
    "SCORE_ADMISSION_REVIEW": (
        'C\n    / "preparation/first_cell_evaluation_score_admission_v10/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V10.json"',
        'C\n    / "preparation/first_cell_evaluation_score_admission_v11/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V11.json"',
    ),
    "SCORE_ADMISSION_REVIEW_SCHEMA": (
        "fg19.first-cell-score-admission-actual-source-review.v10",
        "fg19.first-cell-score-admission-actual-source-review.v11",
    ),
    "SCORE_ADMISSION_REVIEW_STATUS": (
        "PASS_SOURCE_ONLY_ACTUAL_SCORE_BODY_V10",
        "PASS_SOURCE_ONLY_ACTUAL_SCORE_BODY_V11",
    ),
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _top_assignments(tree: ast.Module) -> dict[str, ast.Assign]:
    result: dict[str, ast.Assign] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name in EXPECTED_REPLACEMENTS:
                if name in result:
                    raise ValueError(f"duplicate top-level assignment: {name}")
                result[name] = node
    return result


def _string_literal(node: ast.expr) -> ast.Constant:
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
    raise ValueError("expected a string literal or campaign-root path expression")


def main() -> None:
    old_raw, new_raw = V2.read_bytes(), V3.read_bytes()
    if (len(old_raw), sha(old_raw)) != V2_PIN:
        raise ValueError("frozen V2 caller pin differs")
    old_text = old_raw.decode("utf-8", errors="strict")
    new_text = new_raw.decode("utf-8", errors="strict")
    for name, (old_literal, new_literal) in EXPECTED_REPLACEMENTS.items():
        if old_literal not in old_text or new_literal not in new_text:
            raise ValueError(f"expected source literal is absent: {name}")
    old_tree, new_tree = ast.parse(old_text), ast.parse(new_text)
    old_assignments, new_assignments = _top_assignments(old_tree), _top_assignments(new_tree)
    if set(old_assignments) != set(EXPECTED_REPLACEMENTS) or set(new_assignments) != set(
        EXPECTED_REPLACEMENTS
    ):
        raise ValueError("V2/V3 source-review assignment inventory differs")
    changed = []
    for name, (old_literal, new_literal) in EXPECTED_REPLACEMENTS.items():
        old_value = _string_literal(old_assignments[name].value)
        new_value = _string_literal(new_assignments[name].value)
        old_expected = None
        if name == "SCORE_ADMISSION_BODY":
            old_expected = (
                "preparation/first_cell_evaluation_score_admission_v10/REMOTE_SUBMISSION_V10.py"
            )
            new_expected = (
                "preparation/first_cell_evaluation_score_admission_v11/REMOTE_SUBMISSION_V11.py"
            )
        elif name == "SCORE_ADMISSION_REVIEW":
            old_expected = "preparation/first_cell_evaluation_score_admission_v10/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V10.json"
            new_expected = "preparation/first_cell_evaluation_score_admission_v11/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V11.json"
        else:
            old_expected = old_literal
            new_expected = new_literal
        if old_value.value != old_expected or new_value.value != new_expected:
            raise ValueError(f"V2/V3 literal values differ from the planned rebase: {name}")
        if old_value.value != new_value.value:
            changed.append(name)
            _string_literal(new_assignments[name].value).value = old_value.value
    if set(changed) != set(EXPECTED_REPLACEMENTS):
        raise ValueError("V3 caller did not change exactly the four intended literals")
    if ast.dump(old_tree, include_attributes=False) != ast.dump(new_tree, include_attributes=False):
        raise ValueError("V2/V3 caller AST differs beyond the four source binding literals")
    script_raw = Path(__file__).read_bytes()
    proof = {
        "schema": "fg19.first-cell-score-root-caller-v3-literal-rebase.v1",
        "status": "PASS_EXACT_FOUR_SOURCE_REVIEW_LITERAL_REBASE",
        "source": {
            "v2": {
                "path": str(V2.relative_to(ROOT)),
                "bytes": len(old_raw),
                "sha256": sha(old_raw),
            },
            "v3": {
                "path": str(V3.relative_to(ROOT)),
                "bytes": len(new_raw),
                "sha256": sha(new_raw),
            },
        },
        "proof_script": {
            "path": str(Path(__file__).relative_to(ROOT)),
            "bytes": len(script_raw),
            "sha256": sha(script_raw),
        },
        "changed_assignments": changed,
        "replacements": {
            name: {"from": old, "to": new} for name, (old, new) in EXPECTED_REPLACEMENTS.items()
        },
        "all_other_ast_identical": True,
        "caller_imported_or_executed": False,
        "b_or_ssh_attempted": False,
        "tests_run": False,
    }
    raw = (json.dumps(proof, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with PROOF.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(PREP, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(f"{len(raw)} {sha(raw)} {PROOF}")


if __name__ == "__main__":
    main()
