"""Write a source-only proof of the V4-to-V5 score-capture literal rebase."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
V4 = BASE / "preparation/first_cell_evaluation_score_capture_v4/SCORE_RESULT_CAPTURE_TEMPLATE_V4.py"
V5 = HERE / "SCORE_RESULT_CAPTURE_TEMPLATE_V5.py"
CORE = BASE / "preparation/first_cell_evaluation_score_v6/first_cell_score_core.py"
V4_PIN = (
    26_354,
    "88968f6dd73d4b489c4fc57308906c5e390137a49389ab991408408f353b9be6",
)
CORE_PIN = (
    26_965,
    "961f566eaeec0cd7e9ab48788666d03da9b0482810a11f16a019b083cfc65005",
)
CHANGED = {"SCORE_D", "SCORE_UNIT", "SCORER_CODE_PINS"}
UNBOUND_FIELDS = {
    "SCORE_ADMISSION_BYTES": None,
    "SCORE_ADMISSION_SHA256": "UNBOUND_ROOT_AFTER_ACTUAL_SCORE_ADMISSION",
    "SCORE_LAUNCH_BYTES": None,
    "SCORE_LAUNCH_SHA256": "UNBOUND_ROOT_AFTER_ACTUAL_SCORE_ADMISSION",
    "SCORE_INVOCATION_ID": "UNBOUND_ROOT_AFTER_ACTUAL_SCORE_ADMISSION",
}


def read_pinned(path: Path, expected: tuple[int, str]) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected[0]:
        raise ValueError(f"source is not the expected regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise ValueError(f"source changed before read: {path}")
        raw = os.read(descriptor, expected[0] + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(raw) != expected[0]
        or (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        != identity
    ):
        raise ValueError(f"source changed during read: {path}")
    if hashlib.sha256(raw).hexdigest() != expected[1]:
        raise ValueError(f"source SHA-256 differs: {path}")
    return raw


def read_source(path: Path, byte_cap: int = 1_000_000) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > byte_cap:
        raise ValueError(f"candidate source is not a bounded regular file: {path}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise ValueError(f"candidate source changed before read: {path}")
        raw = os.read(descriptor, byte_cap + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or len(raw) > byte_cap:
        raise ValueError(f"candidate source changed size during read: {path}")
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
        raise ValueError(f"candidate source changed during read: {path}")
    return raw


def assignments(tree: ast.Module) -> dict[str, ast.expr]:
    result: dict[str, ast.expr] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            result[node.targets[0].id] = node.value
    return result


def without_rebased_nodes(tree: ast.Module) -> ast.Module:
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in CHANGED
        )
    ]
    return tree


v4_raw = read_pinned(V4, V4_PIN)
core_raw = read_pinned(CORE, CORE_PIN)
v5_raw = read_source(V5)
v4_tree = ast.parse(v4_raw.decode("utf-8", errors="strict"), filename=str(V4))
v5_tree = ast.parse(v5_raw.decode("utf-8", errors="strict"), filename=str(V5))
v4_assignments = assignments(v4_tree)
v5_assignments = assignments(v5_tree)
changed = {
    name
    for name in v4_assignments.keys() | v5_assignments.keys()
    if name not in v4_assignments
    or name not in v5_assignments
    or ast.dump(v4_assignments[name], include_attributes=False)
    != ast.dump(v5_assignments[name], include_attributes=False)
}
if changed != CHANGED:
    raise ValueError(f"V5 changed assignments differ: {sorted(changed)}")
if ast.dump(without_rebased_nodes(v4_tree), include_attributes=False) != ast.dump(
    without_rebased_nodes(v5_tree), include_attributes=False
):
    raise ValueError("V5 differs from frozen V4 outside the three allowed assignments")
values = {
    name: ast.literal_eval(v5_assignments[name])
    for name in (CHANGED - {"SCORE_D"}) | UNBOUND_FIELDS.keys()
}
if values["SCORE_UNIT"] != "computer-vision-homework-fg19-first-cell-score-v2.service":
    raise ValueError("V5 score unit differs")
score_d_expr = v5_assignments["SCORE_D"]
if not (
    isinstance(score_d_expr, ast.BinOp)
    and isinstance(score_d_expr.left, ast.Name)
    and score_d_expr.left.id == "SCRATCH_ROOT"
    and isinstance(score_d_expr.op, ast.Div)
    and isinstance(score_d_expr.right, ast.Constant)
    and score_d_expr.right.value == "fg19_first_cell_f0_s1001_score_v2"
):
    raise ValueError("V5 score directory differs")
v5_code_pins = values["SCORER_CODE_PINS"]
v4_code_pins = ast.literal_eval(v4_assignments["SCORER_CODE_PINS"])
expected_code_pins = dict(v4_code_pins)
expected_code_pins["first_cell_score_core.py"] = CORE_PIN
if v5_code_pins != expected_code_pins:
    raise ValueError("V5 scorer core pin differs")
for name, expected in UNBOUND_FIELDS.items():
    if values[name] != expected:
        raise ValueError(f"V5 actual-receipt literal was bound prematurely: {name}")

proof = {
    "schema": "fg19.first-cell-score-reader-v5-literal-rebase.v1",
    "status": "PASS_EXACT_V4_UNIT_D2_CORE_REBASE",
    "created_utc": datetime.now(UTC).isoformat(),
    "base": {"path": str(V4), "bytes": len(v4_raw), "sha256": hashlib.sha256(v4_raw).hexdigest()},
    "candidate": {
        "path": str(V5),
        "bytes": len(v5_raw),
        "sha256": hashlib.sha256(v5_raw).hexdigest(),
    },
    "core_pin": {
        "path": str(CORE),
        "bytes": len(core_raw),
        "sha256": hashlib.sha256(core_raw).hexdigest(),
    },
    "changed_assignments": sorted(changed),
    "added_or_removed_nodes": [],
    "bound_namespace": {
        "score_unit": values["SCORE_UNIT"],
        "score_directory": "/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_score_v2",
        "result_path_unchanged": ast.dump(v5_assignments["RESULT_PATH"], include_attributes=False)
        == ast.dump(v4_assignments["RESULT_PATH"], include_attributes=False),
        "export_unit_unchanged": ast.dump(v5_assignments["EXPORT_UNIT"], include_attributes=False)
        == ast.dump(v4_assignments["EXPORT_UNIT"], include_attributes=False),
    },
    "score_admission_literals": {name: values[name] for name in sorted(UNBOUND_FIELDS)},
    "scope": {
        "reader_main_called": False,
        "systemd_or_nvidia_query": False,
        "score_receipt_read": False,
        "result_or_payload_read": False,
        "score_result_or_quality_claim": False,
        "future_actual_values_bound": False,
    },
}
if (
    not proof["bound_namespace"]["result_path_unchanged"]
    or not proof["bound_namespace"]["export_unit_unchanged"]
):
    raise ValueError("V5 changed the export/result namespace")
destination = HERE / "SCORE_CAPTURE_V5_REBASE_PROOF.json"
encoded = (json.dumps(proof, indent=2, sort_keys=True) + "\n").encode("utf-8")
descriptor = os.open(
    destination,
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
directory_descriptor = os.open(HERE, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory_descriptor)
finally:
    os.close(directory_descriptor)
print(
    json.dumps(
        {
            "status": proof["status"],
            "proof": str(destination),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        },
        sort_keys=True,
    )
)
