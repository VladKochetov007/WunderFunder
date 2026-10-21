"""Run only the reviewed score-receipt metadata leaves on actual local receipts."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(
    "/Users/nb/Documents/wunder/WunderFunder-maintenance/research/"
    "campaign_2026-10-19/full_gru_tweaks_r1"
)
OUT = BASE / "preparation/first_cell_evaluation_score_leaf_compatibility_v1"
READER = BASE / "preparation/first_cell_evaluation_score_capture_v4/SCORE_RESULT_CAPTURE_TEMPLATE_V4.py"
READER_REVIEW = BASE / "preparation/first_cell_evaluation_score_capture_v4/INDEPENDENT_SOURCE_REVIEW_V4.json"
HELPER = BASE / "preparation/first_cell_evaluation_score_acceptance_v1/score_result_acceptance.py"
HELPER_REVIEW = BASE / "preparation/first_cell_evaluation_score_acceptance_v1/SOURCE_REVIEW_V1.json"
SCORE_D = BASE / "records/first_cell_evaluation_v1/score_v2"
EXPORT_D = BASE / "records/first_cell_evaluation_v1/export_v5"
INPUTS = {
    "score_admission": SCORE_D / "ADMISSION.json",
    "score_launch": SCORE_D / "LAUNCH.json",
    "score_launch_acceptance": SCORE_D / "LAUNCH_ACCEPTANCE.json",
    "export_acceptance": EXPORT_D / "EXPORT_ACCEPTANCE.json",
    "export_exact_stop": EXPORT_D / "exact_stop/EXACT_STOP.json",
}
EXPECTED_INPUTS = {
    "score_admission": (69_650, "1252b2f1d39ef255fac83c722cfc1a380705963a0b2f9dede35dcabed45ba631"),
    "score_launch": (4_193, "8386643a1b5d80cb3cc69d332d6e822c8864c1c29281edeca8a9b0daf2c63818"),
    "score_launch_acceptance": (2_257, "0de2f723210bf3cddbc0b78f8959527202ce7a284f959ea7be17f98c2217a8c8"),
    "export_acceptance": (8_833, "5a9a99121f80eb7fa07e4591318bea81078ca8b2db1032f150cc33afcc728d0d"),
    "export_exact_stop": (1_140, "695e06c60452e969349b64356dcd795bcbb45892328261136f105b5f8673b72f"),
}
EXPECTED_INVOCATION = "fbc58e9d2a7e47629c6a8feb6302aae8"
EXPECTED_READER = (26_354, "88968f6dd73d4b489c4fc57308906c5e390137a49389ab991408408f353b9be6")
EXPECTED_READER_REVIEW = (5_655, "b05b06a6e56888348e30ebe0ded0272a5e9f3ca709037feb975b938b4d0bb7d0")
EXPECTED_HELPER = (45_795, "5987d527382f0ce33e72b4c17ae020d03485df3057b398ae285940ead29d528b")
EXPECTED_HELPER_REVIEW = (5_695, "31dd246873448a214f772be5793b13cff12b7fcb1d79d4cdc64009fcaf8a56f0")
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 4_000_000


def stable_read(path: Path, cap: int = MAX_FILE_BYTES) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size <= 0 or before.st_size > cap:
        raise ValueError(f"not a bounded regular file: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise ValueError(f"file changed before read: {path}")
        pieces: list[bytes] = []
        remaining = cap + 1
        while remaining:
            block = os.read(fd, min(65_536, remaining))
            if not block:
                break
            pieces.append(block)
            remaining -= len(block)
        raw = b"".join(pieces)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if len(raw) != before.st_size or len(raw) > cap:
        raise ValueError(f"file size changed during read: {path}")
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
        raise ValueError(f"file changed during read: {path}")
    return raw


def pin(path: Path, expected: tuple[int, str] | None = None) -> tuple[bytes, dict[str, object]]:
    raw = stable_read(path)
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and (len(raw), digest) != expected:
        raise ValueError(f"pinned bytes/SHA-256 mismatch: {path}")
    return raw, {"path": str(path), "bytes": len(raw), "sha256": digest}


def assignments(tree: ast.Module) -> dict[str, ast.expr]:
    found: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            found[node.targets[0].id] = node.value
    return found


def safe_literal(name: str, values: dict[str, ast.expr], visiting: set[str] | None = None) -> object:
    visiting = set() if visiting is None else visiting
    if name in visiting or name not in values:
        raise ValueError(f"source constant is absent or cyclic: {name}")
    visiting.add(name)

    def evaluate(node: ast.expr) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return safe_literal(node.id, values, visiting)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(left, Path) and isinstance(right, str):
                return left / right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
            if len(node.args) == 1 and not node.keywords:
                value = evaluate(node.args[0])
                if isinstance(value, str):
                    return Path(value)
        raise ValueError(f"non-literal source expression for {name}")

    result = evaluate(values[name])
    visiting.remove(name)
    return result


def compile_leaf(
    raw: bytes,
    source_path: Path,
    function_names: tuple[str, ...],
    namespace: dict[str, object],
) -> dict[str, object]:
    text = raw.decode("utf-8", errors="strict")
    tree = ast.parse(text, filename=str(source_path))
    defs = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    selected = []
    for name in function_names:
        node = defs.get(name)
        if not isinstance(node, ast.FunctionDef) or node.decorator_list:
            raise ValueError(f"expected undecorated synchronous leaf function: {name}")
        selected.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(source_path), "exec"), namespace)
    for name in function_names:
        function = namespace.get(name)
        if getattr(getattr(function, "__code__", None), "co_filename", None) != str(source_path):
            raise ValueError(f"leaf origin mismatch: {name}")
    return namespace


def strict_object(raw: bytes, label: str, cap: int) -> dict[str, object]:
    # The already source-bound helper parser is used for duplicate-key/constant rejection.
    return helper_ns["_json_object"](raw, label, cap)  # type: ignore[no-any-return]


if sys.dont_write_bytecode is not True:
    raise RuntimeError("run this source-bound leaf check with Python bytecode writes disabled")

input_raw: dict[str, bytes] = {}
input_pins: dict[str, dict[str, object]] = {}
for key, path in INPUTS.items():
    raw, metadata = pin(path, EXPECTED_INPUTS[key])
    input_raw[key] = raw
    input_pins[key] = metadata

reader_raw, reader_pin = pin(READER, EXPECTED_READER)
reader_review_raw, reader_review_pin = pin(READER_REVIEW, EXPECTED_READER_REVIEW)
helper_raw, helper_pin = pin(HELPER, EXPECTED_HELPER)
helper_review_raw, helper_review_pin = pin(HELPER_REVIEW, EXPECTED_HELPER_REVIEW)
if sum(map(len, input_raw.values())) + len(reader_raw) + len(reader_review_raw) + len(helper_raw) + len(helper_review_raw) > MAX_TOTAL_BYTES:
    raise ValueError("metadata input aggregate exceeds reviewed cap")

reader_review = json.loads(reader_review_raw)
helper_review = json.loads(helper_review_raw)
if (
    reader_review.get("schema") != "fg19.first-cell-score-result-capture-source-review.v4"
    or reader_review.get("status") != "PASS_SOURCE_ONLY_READER_V4_HARD_BOUNDED_UNBOUND"
    or reader_review.get("candidate") != {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_capture_v4/SCORE_RESULT_CAPTURE_TEMPLATE_V4.py",
        "bytes": len(reader_raw),
        "sha256": hashlib.sha256(reader_raw).hexdigest(),
    }
):
    raise ValueError("reader source-review pin or status mismatch")
helper_file_rows = helper_review.get("reviewed_files")
helper_row = next(
    (row for row in helper_file_rows if row.get("path") == "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_acceptance_v1/score_result_acceptance.py"),
    None,
)
if (
    helper_review.get("schema") != "fg19.first-cell-score-result-acceptance-source-review.v1"
    or helper_review.get("status") != "PASS_SOURCE_ONLY_ROOT_RESULT_ACCEPTANCE_V1"
    or helper_row != {
        "path": "research/campaign_2026-10-19/full_gru_tweaks_r1/preparation/first_cell_evaluation_score_acceptance_v1/score_result_acceptance.py",
        "bytes": len(helper_raw),
        "sha256": hashlib.sha256(helper_raw).hexdigest(),
        "role": "Reviewed standard-library result and receipt validator; no filesystem reads or scorer imports in build_root_acceptance.",
    }
):
    raise ValueError("result-helper source-review pin or status mismatch")

reader_tree = ast.parse(reader_raw.decode("utf-8"), filename=str(READER))
reader_values = assignments(reader_tree)
reader_constants = {
    name: (
        ast.literal_eval(reader_values[name])
        if name == "SCORER_CODE_PINS"
        else safe_literal(name, reader_values)
    )
    for name in ("PROJECT_BASE", "SCRATCH_ROOT", "SCORE_D", "EXPORT_D", "RESULT_PATH", "SCORE_UNIT", "EXPORT_UNIT", "SCORER_CODE_PINS")
}
expected_reader_constants = {
    "PROJECT_BASE": Path("/home/grinchenkoeg/computer-vision-homework-20261018"),
    "SCRATCH_ROOT": Path("/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp"),
    "SCORE_D": Path("/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_score_v1"),
    "EXPORT_D": Path("/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_evaluation_v3"),
    "RESULT_PATH": Path("/home/grinchenkoeg/computer-vision-homework-20261018/research_tmp/fg19_first_cell_f0_s1001_evaluation_v3/RESULT.json"),
    "SCORE_UNIT": "computer-vision-homework-fg19-first-cell-score-v1.service",
    "EXPORT_UNIT": "computer-vision-homework-fg19-first-cell-export-v3.service",
    "SCORER_CODE_PINS": {
        "first_cell_score_contract.py": (20_165, "b6e3f7a9f9663f519037c54a2077f70d83fbf0b2ba54dc3e158eb5e3b173479d"),
        "first_cell_score_core.py": (27_226, "30c456e7db875e2c075a6200c857c375087b18d9c952c354ae9bd3dceb0a0143"),
        "first_cell_score_predictions.py": (11_056, "88c4ffafaa05a02f40825122e425d310ef4376276ecdfb6b0223aacd6f37ec7d"),
        "score_first_cell.py": (10_630, "b2546281deb54f944577d4db19a8bd7d903cc0f5b755bdb21b2b962696f56dd6"),
    },
}
if reader_constants != expected_reader_constants:
    raise ValueError("reader source constants differ from reviewed fixed identity")

helper_tree = ast.parse(helper_raw.decode("utf-8"), filename=str(HELPER))
helper_values = assignments(helper_tree)
helper_constants = {
    name: ast.literal_eval(helper_values[name])
    for name in ("SCORE_UNIT", "EXPORT_UNIT", "EXPORT_INVOCATION_ID", "RESULT_PATH", "EXPECTED_SCORE_CODE_PINS", "MAX_RECEIPT_BYTES")
}
expected_helper_constants = {
    "SCORE_UNIT": expected_reader_constants["SCORE_UNIT"],
    "EXPORT_UNIT": expected_reader_constants["EXPORT_UNIT"],
    "EXPORT_INVOCATION_ID": "78fa25d0abc44d199e3696d172dbc4f9",
    "RESULT_PATH": str(expected_reader_constants["RESULT_PATH"]),
    "EXPECTED_SCORE_CODE_PINS": expected_reader_constants["SCORER_CODE_PINS"],
    "MAX_RECEIPT_BYTES": 1_000_000,
}
if helper_constants != expected_helper_constants:
    raise ValueError("helper source constants differ from reviewed fixed identity")

import hashlib as hashlib_module
import json as json_module
import re as re_module

reader_ns: dict[str, object] = {
    "hashlib": hashlib_module,
    "json": json_module,
    "re": re_module,
    "Path": Path,
    **reader_constants,
}
reader_ns = compile_leaf(reader_raw, READER, ("_json_object", "_validate_receipt_metadata"), reader_ns)
helper_ns: dict[str, object] = {
    "hashlib": hashlib_module,
    "json": json_module,
    "Any": object,
    **helper_constants,
}
helper_ns = compile_leaf(
    helper_raw,
    HELPER,
    ("_fail", "_sha256", "_strict_equal", "_unique_object", "_reject_constant", "_json_object", "_validate_score_receipts"),
    helper_ns,
)

admission = strict_object(input_raw["score_admission"], "ROOT score admission", 1_000_000)
launch = strict_object(input_raw["score_launch"], "ROOT score launch", 1_000_000)
launch_acceptance = strict_object(input_raw["score_launch_acceptance"], "ROOT score launch acceptance", 1_000_000)
if (
    launch_acceptance.get("schema") != "fg19.first-cell-score-launch-acceptance.v1"
    or launch_acceptance.get("status") != "ACCEPTED_SINGLE_CPU_SCORE_LAUNCH_NOT_RESULT"
    or launch_acceptance.get("unit") != expected_reader_constants["SCORE_UNIT"]
    or launch_acceptance.get("invocation_id") != EXPECTED_INVOCATION
    or launch_acceptance.get("receipts")
    != {
        name: {"bytes": input_pins[key]["bytes"], "sha256": input_pins[key]["sha256"]}
        for name, key in (("ADMISSION.json", "score_admission"), ("LAUNCH.json", "score_launch"))
    }
):
    raise ValueError("ROOT actual launch acceptance does not bind these exact score receipts")

reader_result = reader_ns["_validate_receipt_metadata"](
    input_raw["score_admission"], input_raw["score_launch"], EXPECTED_INVOCATION
)
helper_result = helper_ns["_validate_score_receipts"](
    input_raw["score_admission"],
    input_raw["score_launch"],
    EXPECTED_INVOCATION,
    input_raw["export_acceptance"],
    input_raw["export_exact_stop"],
)
admission_after = stable_read(INPUTS["score_admission"])
launch_after = stable_read(INPUTS["score_launch"])
if admission_after != input_raw["score_admission"] or launch_after != input_raw["score_launch"]:
    raise ValueError("actual score receipts changed during the pure metadata check")
reader_after = stable_read(READER)
helper_after = stable_read(HELPER)
if reader_after != reader_raw or helper_after != helper_raw:
    raise ValueError("reviewed leaf source changed during the check")
if any(name == "wnn" or name.startswith("wnn.") for name in sys.modules):
    raise ValueError("unexpected WNN module was loaded")

report = {
    "schema": "fg19.first-cell-score-receipt-leaf-compatibility.v1",
    "status": "PASS_ACTUAL_RECEIPT_METADATA_LEAVES_ONLY",
    "checked_utc": datetime.now(UTC).isoformat(),
    "identity": {
        "unit": expected_reader_constants["SCORE_UNIT"],
        "invocation_id": EXPECTED_INVOCATION,
        "export_unit": expected_reader_constants["EXPORT_UNIT"],
        "export_invocation_id": helper_constants["EXPORT_INVOCATION_ID"],
        "result_path": str(expected_reader_constants["RESULT_PATH"]),
    },
    "source_pins": {
        "reader": reader_pin,
        "reader_review": reader_review_pin,
        "helper": helper_pin,
        "helper_review": helper_review_pin,
        "leaf_origins": {
            "reader": {name: reader_ns[name].__code__.co_filename for name in ("_json_object", "_validate_receipt_metadata")},
            "helper": {name: helper_ns[name].__code__.co_filename for name in ("_fail", "_sha256", "_strict_equal", "_unique_object", "_reject_constant", "_json_object", "_validate_score_receipts")},
        },
        "pre_post_source_bytes_identical": True,
    },
    "receipt_pins": input_pins,
    "launch_acceptance_receipt_joins": True,
    "reader_leaf": {
        "function": "_validate_receipt_metadata",
        "result": reader_result,
    },
    "acceptance_helper_leaf": {
        "function": "_validate_score_receipts",
        "result": {
            "admission_schema": helper_result[0].get("schema"),
            "admission_status": helper_result[0].get("status"),
            "admission_unit": helper_result[0].get("unit"),
            "launch_schema": helper_result[1].get("schema"),
            "launch_invocation_id": helper_result[1].get("invocation_id"),
            "launch_admission_sha256": helper_result[1].get("admission_sha256"),
            "nested_systemd_unit": helper_result[1]["systemctl_after_submission"]["properties"].get("Id"),
            "nested_systemd_invocation_id": helper_result[1]["systemctl_after_submission"]["properties"].get("InvocationID"),
        },
    },
    "scope": {
        "executed_functions_only": [
            "capture_v4._json_object",
            "capture_v4._validate_receipt_metadata",
            "acceptance_v1._fail",
            "acceptance_v1._sha256",
            "acceptance_v1._strict_equal",
            "acceptance_v1._unique_object",
            "acceptance_v1._reject_constant",
            "acceptance_v1._json_object",
            "acceptance_v1._validate_score_receipts",
        ],
        "full_entrypoints_imported_or_called": False,
        "systemd_or_b_queries": 0,
        "model_prediction_data_cache_array_payload_reads": 0,
        "scoring_or_quality_claim": False,
        "result_json_read": False,
    },
}

encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
if len(encoded) > MAX_FILE_BYTES:
    raise ValueError("compatibility report exceeds cap")
destination = OUT / "ACTUAL_SCORE_RECEIPT_LEAF_COMPATIBILITY_V1.json"
fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
try:
    offset = 0
    while offset < len(encoded):
        offset += os.write(fd, encoded[offset:])
    os.fsync(fd)
finally:
    os.close(fd)
directory_fd = os.open(OUT, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(directory_fd)
finally:
    os.close(directory_fd)
print(json.dumps({"status": report["status"], "report": str(destination), "bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest(), "reader_leaf": "pass", "acceptance_helper_leaf": "pass"}, sort_keys=True))
