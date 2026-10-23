"""Prove status-driver V4 only targets the unit-v3 reader namespace."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
C = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1"
PREP = C / "preparation/first_cell_evaluation_score_capture_v2"
V3 = PREP / "ROOT_CAPTURE_SCORE_STATUS_V3.py"
V4 = PREP / "ROOT_CAPTURE_SCORE_STATUS_V4.py"
PROOF = PREP / "ROOT_SCORE_STATUS_DRIVER_V4_NAMESPACE_PROOF.json"
V3_PIN = (8_895, "c540db668e371355dd11851f828e8d34b394037fe106d97bcd9477e30db0f03d")
EXPECTED = {
    "TEMPLATE": (
        "C / 'preparation/first_cell_evaluation_score_capture_v5/SCORE_RESULT_CAPTURE_TEMPLATE_V5.py'",
        "C / 'preparation/first_cell_evaluation_score_capture_v6/SCORE_RESULT_CAPTURE_TEMPLATE_V6.py'",
    ),
    "TEMPLATE_SHA": (
        "'8d3c37f89376a23db1a548001989c11dd7eb5b83720061c4b229e59f75d1f69c'",
        "'UNBOUND_SCORE_CAPTURE_TEMPLATE_V6_SHA256'",
    ),
    "REVIEW": (
        "TEMPLATE.parent / 'INDEPENDENT_SOURCE_REVIEW_V5.json'",
        "TEMPLATE.parent / 'INDEPENDENT_SOURCE_REVIEW_V6.json'",
    ),
    "REVIEW_SHA": (
        "'6be421f63040c941368af2c08670b22ce1699434a8b264c4f100dd253e229b95'",
        "'UNBOUND_SCORE_CAPTURE_REVIEW_V6_SHA256'",
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
        and node.targets[0].id in EXPECTED
    }


def main() -> None:
    old_raw, new_raw = V3.read_bytes(), V4.read_bytes()
    if (len(old_raw), sha(old_raw)) != V3_PIN:
        raise ValueError("frozen status-driver V3 pin differs")
    old_tree = ast.parse(old_raw.decode("utf-8", errors="strict"))
    new_tree = ast.parse(new_raw.decode("utf-8", errors="strict"))
    old_assignments, new_assignments = assignments(old_tree), assignments(new_tree)
    if set(old_assignments) != set(EXPECTED) or set(new_assignments) != set(EXPECTED):
        raise ValueError("status-driver namespace assignment inventory differs")
    for name, (old_value, new_value) in EXPECTED.items():
        if ast.unparse(old_assignments[name].value) != old_value:
            raise ValueError(f"frozen predecessor binding differs: {name}")
        if ast.unparse(new_assignments[name].value) != new_value:
            raise ValueError(f"candidate binding differs: {name}")
        new_assignments[name].value = copy.deepcopy(old_assignments[name].value)
    if ast.dump(old_tree, include_attributes=False) != ast.dump(
        new_tree, include_attributes=False
    ):
        raise ValueError("status driver differs beyond the four namespace assignments")
    proof = {
        "schema": "fg19.first-cell-score-status-driver-v4-namespace-proof.v1",
        "status": "PASS_V6_READER_PATH_FORK_REVIEW_UNBOUND",
        "predecessor": {
            "path": str(V3.relative_to(ROOT)),
            "bytes": len(old_raw),
            "sha256": V3_PIN[1],
        },
        "candidate": {
            "path": str(V4.relative_to(ROOT)),
            "bytes": len(new_raw),
            "sha256": sha(new_raw),
        },
        "changed_assignments": list(EXPECTED),
        "all_other_ast_identical": True,
        "reader_review_and_candidate_hash": "UNBOUND",
        "driver_executed": False,
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
