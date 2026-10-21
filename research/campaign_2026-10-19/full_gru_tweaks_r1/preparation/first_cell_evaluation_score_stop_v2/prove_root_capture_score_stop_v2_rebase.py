"""Prove the score-stop ROOT capture driver changes only its local record path."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
V1 = (
    ROOT
    / "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_stop_v1/ROOT_CAPTURE_SCORE_STOP_V1.py"
)
PREP = (
    ROOT
    / "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_stop_v2"
)
V2 = PREP / "ROOT_CAPTURE_SCORE_STOP_V2.py"
PROOF = PREP / "ROOT_CAPTURE_SCORE_STOP_V2_REBASE_PROOF.json"
V1_PIN = (12_159, "ad36cac11a2dc67cde3fe61463742537430e97683f63530cb7ce8fe30e75c39d")
OLD_RECORD = "records/first_cell_evaluation_v1/score_v1/exact_stop"
NEW_RECORD = "records/first_cell_evaluation_v1/score_v2/exact_stop"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _record_assignment(tree: ast.Module) -> ast.Assign:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "D"
    ]
    if len(matches) != 1:
        raise ValueError("expected one top-level ROOT stop output path")
    node = matches[0]
    value = node.value
    if (
        not isinstance(value, ast.BinOp)
        or not isinstance(value.op, ast.Div)
        or not isinstance(value.left, ast.Name)
        or value.left.id != "REPOSITORY"
        or not isinstance(value.right, ast.Constant)
        or not isinstance(value.right.value, str)
    ):
        raise ValueError("ROOT stop output path is not a fixed repository-relative string")
    return node


def main() -> None:
    old_raw, new_raw = V1.read_bytes(), V2.read_bytes()
    if (len(old_raw), sha(old_raw)) != V1_PIN:
        raise ValueError("frozen V1 ROOT capture driver pin differs")
    old_tree = ast.parse(old_raw.decode("utf-8", errors="strict"))
    new_tree = ast.parse(new_raw.decode("utf-8", errors="strict"))
    old_path = _record_assignment(old_tree).value.right
    new_path = _record_assignment(new_tree).value.right
    if old_path.value != OLD_RECORD or new_path.value != NEW_RECORD:
        raise ValueError("ROOT stop record path differs from the authorized V1-to-V2 rebase")
    new_path.value = OLD_RECORD
    if ast.dump(old_tree, include_attributes=False) != ast.dump(new_tree, include_attributes=False):
        raise ValueError("V2 root capture driver differs beyond the one record-path literal")

    proof = {
        "schema": "fg19.first-cell-score-stop-root-driver-v2-rebase.v1",
        "status": "PASS_EXACT_SCORE_RECORD_PATH_REBASE",
        "source": {
            "v1": {"path": str(V1.relative_to(ROOT)), "bytes": len(old_raw), "sha256": V1_PIN[1]},
            "v2": {
                "path": str(V2.relative_to(ROOT)),
                "bytes": len(new_raw),
                "sha256": sha(new_raw),
            },
        },
        "changed_assignment": "D",
        "record_path_from": OLD_RECORD,
        "record_path_to": NEW_RECORD,
        "all_other_ast_identical": True,
        "score_systemd_unit_unchanged": "computer-vision-homework-fg19-first-cell-score-v1.service",
        "driver_imported_or_executed": False,
        "b_or_ssh_attempted": False,
        "historical_v1_preserved": True,
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
