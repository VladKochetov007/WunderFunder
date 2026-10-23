#!/usr/bin/env python3
"""Fork the frozen root admission driver only for the fresh score-v2 unit."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PACKET = Path(__file__).resolve().parent
CAMPAIGN = PACKET.parents[1]
V2_DRIVER = (
    CAMPAIGN
    / "preparation/first_cell_evaluation_score_capture_v1/ROOT_CAPTURE_SCORE_ADMISSION_V2.py"
)
V2_DRIVER_PIN = (
    6728,
    "965e6352cd56f5f940f501fb5baa0c83699ce7841919795327d0c140d92ee0e4",
)
OLD_UNIT = "computer-vision-homework-fg19-first-cell-score-v1.service"
NEW_UNIT = "computer-vision-homework-fg19-first-cell-score-v2.service"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def derive() -> tuple[bytes, dict[str, object]]:
    parent = V2_DRIVER.read_bytes()
    if (len(parent), sha(parent)) != V2_DRIVER_PIN:
        raise ValueError("frozen root score-admission V2 driver pin differs")
    old = f'UNIT = "{OLD_UNIT}"'.encode()
    new = f'UNIT = "{NEW_UNIT}"'.encode()
    if parent.count(old) != 1:
        raise ValueError("expected one exact frozen V2 unit literal")
    child = parent.replace(old, new, 1)
    if child.replace(new, old, 1) != parent:
        raise AssertionError("one-literal inverse did not restore exact V2 driver bytes")
    tree_parent = ast.parse(parent)
    tree_child = ast.parse(child)
    parent_assignments = [
        node
        for node in tree_parent.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "UNIT" for target in node.targets)
    ]
    child_assignments = [
        node
        for node in tree_child.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "UNIT" for target in node.targets)
    ]
    if len(parent_assignments) != 1 or len(child_assignments) != 1:
        raise AssertionError("root driver UNIT assignment AST is absent or ambiguous")
    proof = {
        "schema": "fg19.first-cell-score-root-admission-driver-v3-source-proof.v1",
        "status": "PASS_ROOT_CAPTURE_DRIVER_UNIT_FORK_NO_EXECUTION",
        "parent": {
            "path": "preparation/first_cell_evaluation_score_capture_v1/ROOT_CAPTURE_SCORE_ADMISSION_V2.py",
            "bytes": len(parent),
            "sha256": sha(parent),
        },
        "child": {
            "path": "preparation/first_cell_evaluation_score_admission_v13/ROOT_CAPTURE_SCORE_ADMISSION_V3.py",
            "bytes": len(child),
            "sha256": sha(child),
        },
        "only_source_delta": {"assignment": "UNIT", "old": OLD_UNIT, "new": NEW_UNIT},
        "exact_inverse_restores_parent_bytes": True,
        "ast_reviewed_only": True,
        "driver_executed": False,
        "ssh_or_b_action": False,
    }
    return child, proof


if __name__ == "__main__":
    child, proof = derive()
    body_path = PACKET / "ROOT_CAPTURE_SCORE_ADMISSION_V3.py"
    with body_path.open("xb") as stream:
        stream.write(child)
        stream.flush()
    proof_path = PACKET / "ROOT_CAPTURE_SCORE_ADMISSION_V3_PROOF.json"
    proof_raw = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode()
    with proof_path.open("xb") as stream:
        stream.write(proof_raw)
        stream.flush()
    print(
        json.dumps(
            {
                "driver": {"path": str(body_path), "bytes": len(child), "sha256": sha(child)},
                "proof": {
                    "path": str(proof_path),
                    "bytes": len(proof_raw),
                    "sha256": sha(proof_raw),
                },
            }
        )
    )
