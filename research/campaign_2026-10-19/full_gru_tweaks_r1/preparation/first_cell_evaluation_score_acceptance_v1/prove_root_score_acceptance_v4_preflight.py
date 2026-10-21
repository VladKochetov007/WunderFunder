"""Prove the V4 caller differs from frozen V3 only at the output preflight."""

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
V3 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V3.py"
V4 = PREP / "ROOT_ACCEPT_SCORE_RESULT_V4.py"
PROOF = PREP / "ROOT_CALLER_V4_PREFLIGHT_PROOF.json"
V3_PIN = (31_177, "50851aac67f25d81340143d3f845d837bc773db2ae642f7f8f53cc08af29376d")

OLD_DOCSTRING = (
    '"""Apply the reviewed stdlib score-result validator to one exact ROOT capture."""\n'
)
NEW_DOCSTRING = (
    '"""Apply the reviewed stdlib score-result validator to one exact ROOT capture.\n'
    "\n"
    "This is a one-shot receipt writer. If any write fails, preserve every output\n"
    "already present and inspect the exact files, source, and raw inputs read-only;\n"
    "never rerun acceptance or blindly retry this caller.\n"
    '"""\n'
)
OLD_PREFLIGHT = (
    '    if (RECORD / "SCORE_ACCEPTANCE.json").exists() or (\n'
    '        RECORD / "ROOT_ACCEPTANCE_BINDING.json"\n'
    "    ).exists():\n"
)
NEW_PREFLIGHT = (
    "    if (\n"
    '        (RECORD / "SCORE_ACCEPTANCE.json").exists()\n'
    '        or (RECORD / "SCORE_ACCEPTANCE.json").is_symlink()\n'
    '        or (RECORD / "ROOT_ACCEPTANCE_BINDING.json").exists()\n'
    '        or (RECORD / "ROOT_ACCEPTANCE_BINDING.json").is_symlink()\n'
    "    ):\n"
)
OLD_WRITE_CONTEXT = (
    "    acceptance_raw = module.write_root_acceptance_exclusive(\n"
    '        RECORD / "SCORE_ACCEPTANCE.json", acceptance\n'
    "    )\n"
    "\n"
)
NEW_WRITE_CONTEXT = OLD_WRITE_CONTEXT + (
    "    # These two create-only receipts cannot be made atomic together. If the\n"
    "    # binding write fails after acceptance is durable, preserve both paths and\n"
    "    # use a separately reviewed binding-only recovery if one is needed.\n"
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _call_path_and_method(node: ast.expr) -> tuple[str, str] | None:
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return None
    if not isinstance(node.func, ast.Attribute) or node.func.attr not in {"exists", "is_symlink"}:
        return None
    path_expr = node.func.value
    if (
        not isinstance(path_expr, ast.BinOp)
        or not isinstance(path_expr.op, ast.Div)
        or not isinstance(path_expr.left, ast.Name)
        or path_expr.left.id != "RECORD"
        or not isinstance(path_expr.right, ast.Constant)
        or not isinstance(path_expr.right.value, str)
    ):
        return None
    return path_expr.right.value, node.func.attr


def _find_preflight(tree: ast.Module) -> ast.If:
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    matches = []
    for node in ast.walk(main):
        if not isinstance(node, ast.If):
            continue
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "_fail"
            and child.args
            and isinstance(child.args[0], ast.Constant)
            and child.args[0].value
            == "ROOT score acceptance already exists; never repeat this action"
            for statement in node.body
            for child in ast.walk(statement)
        ):
            matches.append(node)
    if len(matches) != 1:
        raise ValueError("caller must have one recognizable create-only output preflight")
    return matches[0]


def main() -> None:
    old_raw, new_raw = V3.read_bytes(), V4.read_bytes()
    if (len(old_raw), sha(old_raw)) != V3_PIN:
        raise ValueError("frozen V3 caller pin differs")
    old_text, new_text = old_raw.decode("utf-8"), new_raw.decode("utf-8")
    for old, new, label in (
        (OLD_DOCSTRING, NEW_DOCSTRING, "one-shot policy docstring"),
        (OLD_PREFLIGHT, NEW_PREFLIGHT, "symlink-aware output preflight"),
        (OLD_WRITE_CONTEXT, NEW_WRITE_CONTEXT, "partial-write recovery policy"),
    ):
        if old_text.count(old) != 1:
            raise ValueError(f"V3 source does not contain one expected {label}")
        old_text = old_text.replace(old, new, 1)
    if old_text != new_text:
        raise ValueError("V4 source contains changes beyond the reviewed narrow patch")

    tree = ast.parse(new_text)
    preflight = _find_preflight(tree)
    if not isinstance(preflight.test, ast.BoolOp) or not isinstance(preflight.test.op, ast.Or):
        raise ValueError("output preflight is not an OR guard")
    calls = [_call_path_and_method(value) for value in preflight.test.values]
    expected_calls = [
        ("SCORE_ACCEPTANCE.json", "exists"),
        ("SCORE_ACCEPTANCE.json", "is_symlink"),
        ("ROOT_ACCEPTANCE_BINDING.json", "exists"),
        ("ROOT_ACCEPTANCE_BINDING.json", "is_symlink"),
    ]
    if calls != expected_calls:
        raise ValueError("preflight does not cover both receipt names and dangling symlinks")

    proof = {
        "schema": "fg19.first-cell-score-root-caller-v4-preflight-proof.v1",
        "status": "PASS_EXACT_SYMLINK_AWARE_PREFLIGHT_AND_ONE_SHOT_POLICY_PATCH",
        "base": {
            "path": str(V3.relative_to(ROOT)),
            "bytes": len(V3.read_bytes()),
            "sha256": V3_PIN[1],
        },
        "candidate": {
            "path": str(V4.relative_to(ROOT)),
            "bytes": len(new_raw),
            "sha256": sha(new_raw),
        },
        "only_source_edits": [
            "one-shot policy docstring",
            "symlink-aware preflight for both create-only receipt paths",
            "comment documenting partial-write recovery policy",
        ],
        "preflight_calls": [{"path": path, "method": method} for path, method in expected_calls],
        "output_mutation_or_helper_execution": False,
        "b_or_ssh_attempted": False,
        "historical_v3_bytes_preserved": True,
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
