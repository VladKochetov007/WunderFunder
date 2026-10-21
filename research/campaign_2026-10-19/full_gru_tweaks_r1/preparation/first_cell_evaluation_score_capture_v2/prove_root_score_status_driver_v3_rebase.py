"""Prove the V3 ROOT status driver binds only the reviewed reader V5 source."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREPARATION = HERE.parent
C = PREPARATION.parent
ROOT = C.parents[1]
V2_DRIVER = HERE / "ROOT_CAPTURE_SCORE_STATUS_V2.py"
V3_DRIVER = HERE / "ROOT_CAPTURE_SCORE_STATUS_V3.py"
V5_TEMPLATE = (
    C / "preparation/first_cell_evaluation_score_capture_v5/SCORE_RESULT_CAPTURE_TEMPLATE_V5.py"
)
V5_REVIEW = V5_TEMPLATE.parent / "INDEPENDENT_SOURCE_REVIEW_V5.json"
PROOF_PATH = HERE / "ROOT_SCORE_STATUS_DRIVER_PROOF_V3.json"
V2_DRIVER_PIN = (
    8_895,
    "e7b479e8237e6ca1f23417ce7a685c3e08fc4bb772fcfc33775634e680f8748c",
)
V5_TEMPLATE_PIN = (
    26_354,
    "8d3c37f89376a23db1a548001989c11dd7eb5b83720061c4b229e59f75d1f69c",
)
V5_REVIEW_PIN = (
    9_206,
    "6be421f63040c941368af2c08670b22ce1699434a8b264c4f100dd253e229b95",
)
REBASABLE = {"TEMPLATE", "TEMPLATE_SHA", "REVIEW", "REVIEW_SHA"}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_stable(path: Path, cap: int, expected: tuple[int, str] | None = None) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
        raise ValueError(f"source is not a bounded regular file: {path}")
    if expected is not None and before.st_size != expected[0]:
        raise ValueError(f"source byte count differs: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise ValueError(f"source changed before read: {path}")
        raw = os.read(descriptor, cap + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or len(raw) > cap:
        raise ValueError(f"source changed size during read: {path}")
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
        raise ValueError(f"source changed during read: {path}")
    if expected is not None and _sha(raw) != expected[1]:
        raise ValueError(f"source SHA-256 differs: {path}")
    return raw


def _tree(raw: bytes, path: Path) -> ast.Module:
    return ast.parse(raw.decode("utf-8", errors="strict"), filename=str(path))


def _normalize_driver(tree: ast.Module) -> ast.Module:
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in REBASABLE:
                node.value = ast.Constant(value=f"<{target.id}>")
    return tree


def _top_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    result: dict[str, ast.expr] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            result[node.targets[0].id] = node.value
    return result


v2_raw = _read_stable(V2_DRIVER, 1_000_000, V2_DRIVER_PIN)
v3_raw = _read_stable(V3_DRIVER, 1_000_000)
v5_raw = _read_stable(V5_TEMPLATE, 1_000_000, V5_TEMPLATE_PIN)
review_raw = _read_stable(V5_REVIEW, 128_000, V5_REVIEW_PIN)
v2_tree = _tree(v2_raw, V2_DRIVER)
v3_tree = _tree(v3_raw, V3_DRIVER)
if ast.dump(_normalize_driver(v2_tree), include_attributes=False) != ast.dump(
    _normalize_driver(v3_tree), include_attributes=False
):
    raise ValueError("V3 root status driver differs beyond its four V5 reader/review pins")

v3_values = _top_assignments(_tree(v3_raw, V3_DRIVER))
if ast.literal_eval(v3_values["TEMPLATE_SHA"]) != V5_TEMPLATE_PIN[1]:
    raise ValueError("V3 driver template SHA differs")
if ast.literal_eval(v3_values["REVIEW_SHA"]) != V5_REVIEW_PIN[1]:
    raise ValueError("V3 driver review SHA differs")
template_expression = v3_values["TEMPLATE"]
if not (
    isinstance(template_expression, ast.BinOp)
    and isinstance(template_expression.left, ast.Name)
    and template_expression.left.id == "C"
    and isinstance(template_expression.op, ast.Div)
    and isinstance(template_expression.right, ast.Constant)
    and template_expression.right.value
    == "preparation/first_cell_evaluation_score_capture_v5/SCORE_RESULT_CAPTURE_TEMPLATE_V5.py"
):
    raise ValueError("V3 driver does not select the exact reader V5 template")
review_expression = v3_values["REVIEW"]
if not (
    isinstance(review_expression, ast.BinOp)
    and isinstance(review_expression.left, ast.Attribute)
    and isinstance(review_expression.left.value, ast.Name)
    and review_expression.left.value.id == "TEMPLATE"
    and review_expression.left.attr == "parent"
    and isinstance(review_expression.op, ast.Div)
    and isinstance(review_expression.right, ast.Constant)
    and review_expression.right.value == "INDEPENDENT_SOURCE_REVIEW_V5.json"
):
    raise ValueError("V3 driver does not select the exact reader V5 review")

review = json.loads(review_raw)
if (
    review.get("schema") != "fg19.first-cell-score-result-capture-source-review.v5"
    or review.get("status") != "PASS_SOURCE_ONLY_READER_V5_HARD_BOUNDED_UNBOUND"
):
    raise ValueError("reader V5 independent review does not have its required status")
if review.get("candidate") != {
    "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_capture_v5/SCORE_RESULT_CAPTURE_TEMPLATE_V5.py",
    "bytes": V5_TEMPLATE_PIN[0],
    "sha256": V5_TEMPLATE_PIN[1],
}:
    raise ValueError("reader V5 review does not pin the exact reader template")
reader_values = _top_assignments(_tree(v5_raw, V5_TEMPLATE))
if any(
    ast.literal_eval(reader_values[name]) is not None
    for name in ("SCORE_ADMISSION_BYTES", "SCORE_LAUNCH_BYTES")
) or any(
    "UNBOUND" not in ast.literal_eval(reader_values[name])
    for name in ("SCORE_ADMISSION_SHA256", "SCORE_LAUNCH_SHA256", "SCORE_INVOCATION_ID")
):
    raise ValueError("V5 reader actual score receipt literals were bound prematurely")

record = {
    "schema": "fg19.first-cell-score-root-status-driver-pair.v3",
    "status": "LOCAL_V3_REBASED_TO_REVIEWED_SCORE_CAPTURE_V5_NOT_EXECUTED",
    "created_utc": datetime.now(UTC).isoformat(),
    "driver": {
        "path": str(V3_DRIVER.relative_to(C)),
        "bytes": len(v3_raw),
        "sha256": _sha(v3_raw),
    },
    "predecessor": {
        "path": str(V2_DRIVER.relative_to(C)),
        "bytes": len(v2_raw),
        "sha256": _sha(v2_raw),
    },
    "reader": {
        "path": str(V5_TEMPLATE.relative_to(C)),
        "bytes": len(v5_raw),
        "sha256": _sha(v5_raw),
    },
    "independent_reader_review": {
        "path": str(V5_REVIEW.relative_to(C)),
        "bytes": len(review_raw),
        "sha256": _sha(review_raw),
        "schema": review["schema"],
        "status": review["status"],
    },
    "rebase_assignments": sorted(REBASABLE),
    "all_other_driver_ast_identical": True,
    "actual_score_admission_and_launch_literals_unbound": True,
    "scope": {
        "driver_imported_or_executed": False,
        "reader_main_called": False,
        "actual_score_receipts_read": False,
        "B_or_SSH_attempted": False,
        "unit_queried": False,
        "result_or_payload_read": False,
        "quality_claimed": False,
    },
}
encoded = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
descriptor = os.open(
    PROOF_PATH,
    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
    0o644,
)
try:
    position = 0
    while position < len(encoded):
        position += os.write(descriptor, encoded[position:])
    os.fsync(descriptor)
finally:
    os.close(descriptor)
directory_fd = os.open(HERE, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
print(json.dumps(record, sort_keys=True))
