"""Prove that V7's score body differs from V6 only in source-inventory functions."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
V6 = (
    PREPARATION / "first_cell_evaluation_score_admission_v6" / "SCORE_ADMISSION_BODY_TEMPLATE_V6.py"
)
V7 = PACKET / "SCORE_ADMISSION_BODY_TEMPLATE_V7.py"
OUTPUT = PACKET / "V7_INVERSE_PROOF.json"
V6_PIN = (70_643, "aecec19975a498174a859228499b2e12a5a24965b0ead94f0b8f78941fdf4e84")
ALLOWED_FUNCTIONS = {"inventory_exact_source_files", "check_source_tree"}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def outside_inventory(raw: bytes, filename: Path) -> bytes:
    tree = ast.parse(raw.decode("utf-8", errors="strict"), filename=str(filename))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name in ALLOWED_FUNCTIONS)
    ]
    return ast.dump(tree, include_attributes=False).encode("utf-8")


def main() -> None:
    v6_raw = V6.read_bytes()
    v7_raw = V7.read_bytes()
    if (len(v6_raw), sha256(v6_raw)) != V6_PIN:
        raise ValueError("frozen V6 body pin differs")
    v6_ast = outside_inventory(v6_raw, V6)
    v7_ast = outside_inventory(v7_raw, V7)
    if v6_ast != v7_ast:
        raise ValueError("V7 changed body AST outside the inventory functions")
    proof = {
        "schema": "fg19.first-cell-score-admission-v7-inverse-proof.v1",
        "status": "PASS_V6_AST_IDENTICAL_OUTSIDE_SOURCE_INVENTORY_FUNCTIONS",
        "baseline": {
            "path": "first_cell_evaluation_score_admission_v6/SCORE_ADMISSION_BODY_TEMPLATE_V6.py",
            "bytes": len(v6_raw),
            "sha256": sha256(v6_raw),
        },
        "candidate": {
            "path": "first_cell_evaluation_score_admission_v7/SCORE_ADMISSION_BODY_TEMPLATE_V7.py",
            "bytes": len(v7_raw),
            "sha256": sha256(v7_raw),
        },
        "removed_functions_for_comparison": sorted(ALLOWED_FUNCTIONS),
        "remaining_module_ast_sha256": sha256(v7_ast),
        "actual_export_or_runtime_action": False,
    }
    OUTPUT.write_text(json.dumps(proof, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    main()
