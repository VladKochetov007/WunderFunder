"""Prove the two unbound unit-v3 templates are narrow AST forks."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
C = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1"
SPECS = (
    {
        "name": "acceptance_helper",
        "base": C / "preparation/first_cell_evaluation_score_acceptance_v2/score_result_acceptance_v2.py",
        "base_pin": (
            46_448,
            "1b8a6e899dcd2c250eecf096bd1797aa77466d32eedd598b063f48249490f446",
        ),
        "candidate": C / "preparation/first_cell_evaluation_score_acceptance_v3/score_result_acceptance_v3.py",
        "changed": ("SCORE_UNIT", "EXPECTED_SCORE_CODE_PINS"),
        "expected_unit": "computer-vision-homework-fg19-first-cell-score-v3.service",
    },
    {
        "name": "bounded_reader",
        "base": C / "preparation/first_cell_evaluation_score_capture_v5/SCORE_RESULT_CAPTURE_TEMPLATE_V5.py",
        "base_pin": (
            26_354,
            "8d3c37f89376a23db1a548001989c11dd7eb5b83720061c4b229e59f75d1f69c",
        ),
        "candidate": C / "preparation/first_cell_evaluation_score_capture_v6/SCORE_RESULT_CAPTURE_TEMPLATE_V6.py",
        "changed": ("SCORE_D", "SCORE_UNIT", "SCORER_CODE_PINS"),
        "expected_unit": "computer-vision-homework-fg19-first-cell-score-v3.service",
    },
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def assignment_map(tree: ast.Module) -> dict[str, ast.Assign]:
    result: dict[str, ast.Assign] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            result[node.targets[0].id] = node
    return result


def prove(spec: dict[str, object]) -> dict[str, object]:
    base_path = spec["base"]
    candidate_path = spec["candidate"]
    assert isinstance(base_path, Path) and isinstance(candidate_path, Path)
    base_raw, candidate_raw = base_path.read_bytes(), candidate_path.read_bytes()
    if (len(base_raw), sha(base_raw)) != spec["base_pin"]:
        raise ValueError(f"frozen predecessor pin differs: {spec['name']}")
    base_tree = ast.parse(base_raw.decode("utf-8", errors="strict"))
    candidate_tree = ast.parse(candidate_raw.decode("utf-8", errors="strict"))
    base_assignments = assignment_map(base_tree)
    candidate_assignments = assignment_map(candidate_tree)
    changed = tuple(spec["changed"])
    for name in changed:
        if name not in base_assignments or name not in candidate_assignments:
            raise ValueError(f"missing namespace binding {name}")
    if ast.literal_eval(candidate_assignments["SCORE_UNIT"].value) != spec["expected_unit"]:
        raise ValueError(f"wrong unit namespace: {spec['name']}")
    if "SCORE_D" in changed:
        score_path = candidate_assignments["SCORE_D"].value
        if not (
            isinstance(score_path, ast.BinOp)
            and isinstance(score_path.left, ast.Name)
            and score_path.left.id == "SCRATCH_ROOT"
            and isinstance(score_path.op, ast.Div)
            and isinstance(score_path.right, ast.Constant)
            and score_path.right.value == "fg19_first_cell_f0_s1001_score_v3"
        ):
            raise ValueError("reader scratch namespace differs")
    pins = ast.literal_eval(candidate_assignments["EXPECTED_SCORE_CODE_PINS" if spec["name"] == "acceptance_helper" else "SCORER_CODE_PINS"].value)
    if pins["first_cell_score_core.py"] != (
        30_545,
        "c2164bf3df334e2ddbfb354d7fd1e4b02562430a6225b675ca0b1d86c82b772e",
    ):
        raise ValueError("the V7 scorer core pin differs")
    for name in changed:
        candidate_assignments[name].value = copy.deepcopy(base_assignments[name].value)
    if ast.dump(base_tree, include_attributes=False) != ast.dump(
        candidate_tree, include_attributes=False
    ):
        raise ValueError(f"template differs outside approved namespace assignments: {spec['name']}")
    return {
        "name": spec["name"],
        "base": {
            "path": str(base_path.relative_to(ROOT)),
            "bytes": len(base_raw),
            "sha256": sha(base_raw),
        },
        "candidate": {
            "path": str(candidate_path.relative_to(ROOT)),
            "bytes": len(candidate_raw),
            "sha256": sha(candidate_raw),
        },
        "changed_assignments": list(changed),
        "unit": spec["expected_unit"],
        "v7_core_pin": {
            "bytes": 30_545,
            "sha256": "c2164bf3df334e2ddbfb354d7fd1e4b02562430a6225b675ca0b1d86c82b772e",
        },
        "all_other_ast_identical": True,
    }


def main() -> None:
    rows = [prove(spec) for spec in SPECS]
    proof = {
        "schema": "fg19.first-cell-score-v3-template-namespace-proof.v2",
        "status": "PASS_NARROW_UNIT_D3_FORKS_PINNED_V7_CORE",
        "forks": rows,
        "templates_imported_or_executed": False,
        "tests_or_b_actions_attempted": False,
    }
    output = C / "preparation/first_cell_evaluation_score_capture_v6/NAMESPACE_FORK_PROOF_V2.json"
    raw = (json.dumps(proof, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(f"{len(raw)} {sha(raw)} {output}")


if __name__ == "__main__":
    main()
