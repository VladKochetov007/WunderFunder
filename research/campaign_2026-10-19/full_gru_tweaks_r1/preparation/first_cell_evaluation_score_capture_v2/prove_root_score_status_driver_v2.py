"""Prove the V2 root score-status driver's exact reviewed-reader rebase."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

PACKET = Path(__file__).parent
PREPARATION = PACKET.parent
C = PREPARATION.parent
V1_DRIVER = PACKET / "ROOT_CAPTURE_SCORE_STATUS_V1.py"
V2_DRIVER = PACKET / "ROOT_CAPTURE_SCORE_STATUS_V2.py"
V4_TEMPLATE = (
    C / "preparation/first_cell_evaluation_score_capture_v4/SCORE_RESULT_CAPTURE_TEMPLATE_V4.py"
)
V4_REVIEW = V4_TEMPLATE.parent / "INDEPENDENT_SOURCE_REVIEW_V4.json"
PROOF_PATH = PACKET / "ROOT_SCORE_STATUS_DRIVER_PROOF_V2.json"
EXPECTED_V1_DRIVER_SHA256 = "cf357eabe5d563fc98899fae6ee1131bffd80c7d412f2b104e9c612afb950ba1"
EXPECTED_V4_TEMPLATE_SHA256 = "88968f6dd73d4b489c4fc57308906c5e390137a49389ab991408408f353b9be6"
EXPECTED_V4_REVIEW_SHA256 = "b05b06a6e56888348e30ebe0ded0272a5e9f3ca709037feb975b938b4d0bb7d0"
REBASABLE = {"TEMPLATE", "TEMPLATE_SHA", "REVIEW", "REVIEW_SHA"}


def _sha(raw: bytes) -> str:
    """Return the SHA-256 digest of source bytes."""
    return hashlib.sha256(raw).hexdigest()


def _tree(path: Path) -> ast.Module:
    """Parse a pinned local source file without importing it."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _normalize_driver(path: Path) -> ast.Module:
    """Replace only the four reader/review assignments in a driver AST."""
    tree = _tree(path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in REBASABLE:
                node.value = ast.Constant(value=f"<{target.id}>")
    return tree


def _top_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    values: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                values[target.id] = node.value
    return values


def _file_pin(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"path": path.relative_to(C).as_posix(), "bytes": len(raw), "sha256": _sha(raw)}


def main() -> None:
    """Validate pinned source joins and write one local inverse-AST record."""
    if PROOF_PATH.exists():
        raise FileExistsError(f"proof output already exists: {PROOF_PATH}")
    v1_raw = V1_DRIVER.read_bytes()
    v4_raw, review_raw = V4_TEMPLATE.read_bytes(), V4_REVIEW.read_bytes()
    if _sha(v1_raw) != EXPECTED_V1_DRIVER_SHA256:
        raise ValueError("frozen V1 root score-status driver pin differs")
    if _sha(v4_raw) != EXPECTED_V4_TEMPLATE_SHA256:
        raise ValueError("reviewed V4 score reader pin differs")
    if _sha(review_raw) != EXPECTED_V4_REVIEW_SHA256:
        raise ValueError("independent V4 score-reader review pin differs")
    candidate_tree = _tree(V2_DRIVER)
    values = _top_assignments(candidate_tree)
    template_expression = values["TEMPLATE"]
    if (
        not isinstance(template_expression, ast.BinOp)
        or not isinstance(template_expression.left, ast.Name)
        or template_expression.left.id != "C"
        or not isinstance(template_expression.right, ast.Constant)
        or template_expression.right.value
        != "preparation/first_cell_evaluation_score_capture_v4/SCORE_RESULT_CAPTURE_TEMPLATE_V4.py"
    ):
        raise ValueError("V2 driver does not select the V4 reader template")
    review_expression = values["REVIEW"]
    if (
        not isinstance(review_expression, ast.BinOp)
        or ast.unparse(review_expression.left) != "TEMPLATE.parent"
        or not isinstance(review_expression.right, ast.Constant)
        or review_expression.right.value != "INDEPENDENT_SOURCE_REVIEW_V4.json"
    ):
        raise ValueError("V2 driver does not select the V4 reader review")
    if ast.literal_eval(values["TEMPLATE_SHA"]) != EXPECTED_V4_TEMPLATE_SHA256:
        raise ValueError("V2 driver template digest differs")
    if ast.literal_eval(values["REVIEW_SHA"]) != EXPECTED_V4_REVIEW_SHA256:
        raise ValueError("V2 driver review digest differs")
    review = json.loads(review_raw)
    if review.get("status") != "PASS_SOURCE_ONLY_READER_V4_HARD_BOUNDED_UNBOUND":
        raise ValueError("V4 reader review is not the expected independent PASS")
    candidate = review.get("candidate")
    if not isinstance(candidate, dict) or (
        candidate.get("bytes"),
        candidate.get("sha256"),
    ) != (len(v4_raw), _sha(v4_raw)):
        raise ValueError("V4 independent review does not pin its exact candidate")
    if ast.dump(_normalize_driver(V1_DRIVER), include_attributes=False) != ast.dump(
        _normalize_driver(V2_DRIVER), include_attributes=False
    ):
        raise ValueError("root status driver differs beyond the four reader/review assignments")
    reader_values = _top_assignments(_tree(V4_TEMPLATE))
    if any(
        ast.literal_eval(reader_values[name]) is not None
        for name in ("SCORE_ADMISSION_BYTES", "SCORE_LAUNCH_BYTES")
    ) or "UNBOUND" not in ast.literal_eval(reader_values["SCORE_INVOCATION_ID"]):
        raise ValueError("actual score admission/launch literals were prematurely bound")
    record = {
        "schema": "fg19.first-cell-score-root-status-driver-pair.v2",
        "status": "LOCAL_V2_REBASED_TO_REVIEWED_SCORE_CAPTURE_V4_NOT_EXECUTED",
        "driver": _file_pin(V2_DRIVER),
        "predecessor": _file_pin(V1_DRIVER),
        "reader": _file_pin(V4_TEMPLATE),
        "independent_reader_review": {
            **_file_pin(V4_REVIEW),
            "status": review["status"],
        },
        "rebase_assignments": ["TEMPLATE", "TEMPLATE_SHA", "REVIEW", "REVIEW_SHA"],
        "all_other_driver_ast_identical": True,
        "actual_score_admission_and_launch_literals_unbound": True,
        "scope": {
            "driver_imported_or_executed": False,
            "actual_score_receipts_read": False,
            "B_or_SSH_attempted": False,
            "unit_queried": False,
            "result_or_payload_read": False,
            "quality_claimed": False,
        },
    }
    encoded = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with PROOF_PATH.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(PACKET, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
