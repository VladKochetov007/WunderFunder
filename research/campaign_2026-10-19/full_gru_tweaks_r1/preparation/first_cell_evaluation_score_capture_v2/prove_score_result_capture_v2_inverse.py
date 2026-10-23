"""Prove the reader V2 change is limited to root, scorer pins, and exact unit Id."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PACKET = Path(__file__).resolve().parent
PREPARATION = PACKET.parent
BASE = "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation"
V1 = PREPARATION / "first_cell_evaluation_score_capture_v1" / "SCORE_RESULT_CAPTURE_TEMPLATE_V1.py"
V2 = PACKET / "SCORE_RESULT_CAPTURE_TEMPLATE_V2.py"
SCORE_V3 = PREPARATION / "first_cell_evaluation_score_v3"
EXPECTED_SCORER_PINS = {
    "first_cell_score_contract.py": (
        19_485,
        "7784c91713dbfd7d11ec3cfd276eb8a62607289067999f4ecd5c03a3f9a18038",
    ),
    "first_cell_score_core.py": (
        27_226,
        "ffe8a6e7aad38b131a78783504ef55c761d4e8037c13bac279f20c5e55f6249e",
    ),
    "first_cell_score_predictions.py": (
        11_056,
        "88c4ffafaa05a02f40825122e425d310ef4376276ecdfb6b0223aacd6f37ec7d",
    ),
    "score_first_cell.py": (
        10_630,
        "b2546281deb54f944577d4db19a8bd7d903cc0f5b755bdb21b2b962696f56dd6",
    ),
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _assignments(tree: ast.Module) -> dict[str, ast.Assign]:
    result: dict[str, ast.Assign] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            result[node.targets[0].id] = node
    return result


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _normalized_tree(tree: ast.Module, *, remove_v2_guard: bool) -> ast.Module:
    assignments = _assignments(tree)
    for name in ("EXPORT_D", "EXPORT_UNIT", "SCORER_CODE_PINS", "UNIT_PROPERTIES"):
        assignments[name].value = ast.Constant(value=f"<{name}>")
    if remove_v2_guard:
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.FunctionDef) and node.name == "_require_exact_unit_identity"
            )
        ]
        main = _function(tree, "main")
        unit_index = next(
            index
            for index, node in enumerate(main.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "unit" for target in node.targets)
        )
        guard = main.body.pop(unit_index + 1)
        expected_call = ast.parse("_require_exact_unit_identity(unit)").body[0]
        if _dump(guard) != _dump(expected_call):
            raise AssertionError("V2 unit-ID guard is not the exact narrow insertion")
    return tree


def prove() -> dict[str, object]:
    old_tree, new_tree = _parse(V1), _parse(V2)
    old_assignments, new_assignments = _assignments(old_tree), _assignments(new_tree)
    old_export = ast.unparse(old_assignments["EXPORT_D"].value)
    new_export = ast.unparse(new_assignments["EXPORT_D"].value)
    if "evaluation_v1" not in old_export or "evaluation_v2" not in new_export:
        raise AssertionError("the output-root version transition is not exact")
    if ast.unparse(old_assignments["SCORE_D"].value) != ast.unparse(
        new_assignments["SCORE_D"].value
    ):
        raise AssertionError("score private directory changed unexpectedly")
    if ast.unparse(old_assignments["SCORE_UNIT"].value) != ast.unparse(
        new_assignments["SCORE_UNIT"].value
    ):
        raise AssertionError("score unit changed unexpectedly")

    old_properties = ast.literal_eval(old_assignments["UNIT_PROPERTIES"].value)
    new_properties = ast.literal_eval(new_assignments["UNIT_PROPERTIES"].value)
    if new_properties != ("Id", *old_properties):
        raise AssertionError("V2 unit properties are not the exact V1 list plus Id")

    observed_pins = ast.literal_eval(new_assignments["SCORER_CODE_PINS"].value)
    for name, expected in EXPECTED_SCORER_PINS.items():
        raw = (SCORE_V3 / name).read_bytes()
        actual = (len(raw), hashlib.sha256(raw).hexdigest())
        if actual != expected or observed_pins[name] != expected:
            raise AssertionError(f"scorer pin does not match V3 source: {name}")

    expected_helper = ast.parse(
        "def _require_exact_unit_identity(unit: dict[str, str]) -> None:\n"
        '    """Reject a systemd response whose Id is not the requested score unit."""\n'
        '    if unit.get("Id") != SCORE_UNIT:\n'
        '        raise ValueError("systemd returned a different score unit Id")\n'
    ).body[0]
    if _dump(_function(new_tree, "_require_exact_unit_identity")) != _dump(expected_helper):
        raise AssertionError("V2 unit identity helper differs from the exact guard")
    if _dump(_normalized_tree(old_tree, remove_v2_guard=False)) != _dump(
        _normalized_tree(new_tree, remove_v2_guard=True)
    ):
        raise AssertionError("V2 contains changes beyond the pinned root, pins, Id, and guard")

    template_raw = V2.read_bytes()
    proof: dict[str, object] = {
        "schema": "fg19.first-cell-score-result-capture-inverse-proof.v2",
        "status": "PASS_EXACT_V1_TO_V2_UNIT_ID_AND_EXPORT_ROOT_DELTA",
        "scope": {
            "B_or_SSH_accessed": False,
            "unit_queried": False,
            "reader_executed": False,
            "result_or_prediction_payload_read": False,
            "quality_claim": False,
        },
        "prior_template": {
            "path": f"{BASE}/first_cell_evaluation_score_capture_v1/SCORE_RESULT_CAPTURE_TEMPLATE_V1.py",
            "bytes": V1.stat().st_size,
            "sha256": hashlib.sha256(V1.read_bytes()).hexdigest(),
            "review_status": "REJECT_SOURCE_ONLY_NO_SCORE_RESULT_CAPTURE",
        },
        "candidate_template": {
            "path": f"{BASE}/first_cell_evaluation_score_capture_v2/SCORE_RESULT_CAPTURE_TEMPLATE_V2.py",
            "bytes": len(template_raw),
            "sha256": hashlib.sha256(template_raw).hexdigest(),
        },
        "exact_changes": [
            "EXPORT_D evidence root changes from evaluation_v1 to evaluation_v2",
            "EXPORT_UNIT changes from the failed export-v1 unit to the frozen export-v2 unit",
            "SCORER_CODE_PINS bind the frozen scorer V3 files",
            "UNIT_PROPERTIES adds only Id",
            "one helper requires Id == SCORE_UNIT immediately after the exact systemctl show",
        ],
        "verified": {
            "score_private_directory_and_unit_unchanged": True,
            "export_unit_matches_frozen_export_v2": True,
            "four_scorer_pins_match_score_v3_sources": True,
            "V2_properties_equal_Id_plus_V1_properties": True,
            "one_exact_unit_id_guard_precedes_result_capture": True,
            "all_other_operational_AST_matches_V1": True,
        },
        "scorer_code_pins": {
            name: {"bytes": size, "sha256": digest}
            for name, (size, digest) in EXPECTED_SCORER_PINS.items()
        },
        "limitations": [
            "This inverse proof compares local source ASTs and reads only code files.",
            "It does not bind actual score admission or launch receipts, query systemd, access the result, or make a metric claim.",
        ],
    }
    return proof


if __name__ == "__main__":
    value = prove()
    path = PACKET / "SCORE_RESULT_CAPTURE_V2_INVERSE_PROOF.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": value["status"], "proof": path.name}, sort_keys=True))
