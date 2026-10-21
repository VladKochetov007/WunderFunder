"""Prove status-driver V5 only rebinds the current V6 reader review pin."""

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
OLD = PREP / "ROOT_CAPTURE_SCORE_STATUS_V4.py"
NEW = PREP / "ROOT_CAPTURE_SCORE_STATUS_V5.py"
REVIEW = C / "preparation/first_cell_evaluation_score_capture_v6/INDEPENDENT_SOURCE_REVIEW_V6.json"
PROOF = PREP / "ROOT_SCORE_STATUS_DRIVER_V5_REBASE_PROOF_V1.json"
OLD_PIN = (8_895, "5741407e72454b3f45d92426d21a70a00303bad30b3de9ebc4be870171c8382a")
REVIEW_PIN = (9_525, "844fb16d236a6778179f55edfe37a553809097691b19060869309c0c07bbeeec")
OLD_REVIEW_SHA = "5e067c1d9972be3f00f462a075084b5ea315e3708bdb18669a0cf52be1a2d8ed"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def review_assignment(tree: ast.Module) -> ast.Assign:
    found = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "REVIEW_SHA"
    ]
    if len(found) != 1:
        raise ValueError("expected exactly one REVIEW_SHA assignment")
    return found[0]


def main() -> None:
    old_raw, new_raw, review_raw = OLD.read_bytes(), NEW.read_bytes(), REVIEW.read_bytes()
    if (len(old_raw), sha(old_raw)) != OLD_PIN:
        raise ValueError("frozen V4 status-driver pin differs")
    if (len(review_raw), sha(review_raw)) != REVIEW_PIN:
        raise ValueError("current V6 reader review pin differs")
    review = json.loads(review_raw)
    if review.get("schema") != "fg19.first-cell-score-result-capture-source-review.v6":
        raise ValueError("reader review schema differs")
    if review.get("status") != "PASS_SOURCE_ONLY_READER_V6_HARD_BOUNDED_UNBOUND":
        raise ValueError("reader review status differs")
    old_tree = ast.parse(old_raw.decode("utf-8", errors="strict"))
    new_tree = ast.parse(new_raw.decode("utf-8", errors="strict"))
    old_assignment, new_assignment = review_assignment(old_tree), review_assignment(new_tree)
    if ast.unparse(old_assignment.value) != repr(OLD_REVIEW_SHA):
        raise ValueError("V4 predecessor review pin differs")
    if ast.unparse(new_assignment.value) != repr(REVIEW_PIN[1]):
        raise ValueError("V5 candidate does not pin current reader review")
    new_assignment.value = copy.deepcopy(old_assignment.value)
    if ast.dump(old_tree, include_attributes=False) != ast.dump(new_tree, include_attributes=False):
        raise ValueError("status driver differs beyond REVIEW_SHA")
    proof = {
        "schema": "fg19.first-cell-score-status-driver-v5-rebase-proof.v1",
        "status": "PASS_CURRENT_V6_READER_REVIEW_PINNED",
        "predecessor": {"path": str(OLD.relative_to(ROOT)), "bytes": len(old_raw), "sha256": OLD_PIN[1]},
        "candidate": {"path": str(NEW.relative_to(ROOT)), "bytes": len(new_raw), "sha256": sha(new_raw)},
        "reader_review": {"path": str(REVIEW.relative_to(ROOT)), "bytes": len(review_raw), "sha256": REVIEW_PIN[1]},
        "changed_assignment_only": "REVIEW_SHA",
        "all_other_ast_identical": True,
        "driver_or_reader_executed": False,
        "tests_or_unit_queries_performed": False,
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
