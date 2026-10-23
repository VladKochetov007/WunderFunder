"""Apply the reviewed stdlib score-result validator to one exact ROOT capture.

This is a one-shot receipt writer. If any write fails, preserve every output
already present and inspect the exact files, source, and raw inputs read-only;
never rerun acceptance or blindly retry this caller.
"""

from __future__ import annotations

import argparse
import ast
import base64
import datetime as dt
import hashlib
import json
import os
import stat
import types
from pathlib import Path
from typing import Any

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
C = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1"
PREP = C / "preparation/first_cell_evaluation_score_acceptance_v1"
RECORD = C / "records/first_cell_evaluation_v1/score_v2"
EXPORT = C / "records/first_cell_evaluation_v1/export_v5"
HELPER = PREP / "score_result_acceptance.py"
HELPER_PIN = (45_795, "5987d527382f0ce33e72b4c17ae020d03485df3057b398ae285940ead29d528b")
HELPER_REVIEW = PREP / "SOURCE_REVIEW_V1.json"
HELPER_REVIEW_PIN = (5_695, "31dd246873448a214f772be5793b13cff12b7fcb1d79d4cdc64009fcaf8a56f0")
HELPER_REVIEW_SCHEMA = "fg19.first-cell-score-result-acceptance-source-review.v1"
HELPER_REVIEW_STATUS = "PASS_SOURCE_ONLY_ROOT_RESULT_ACCEPTANCE_V1"

READER_TEMPLATE = (
    C / "preparation/first_cell_evaluation_score_capture_v4/SCORE_RESULT_CAPTURE_TEMPLATE_V4.py"
)
READER_TEMPLATE_PIN = (
    26_354,
    "88968f6dd73d4b489c4fc57308906c5e390137a49389ab991408408f353b9be6",
)
READER_REVIEW = (
    C / "preparation/first_cell_evaluation_score_capture_v4/INDEPENDENT_SOURCE_REVIEW_V4.json"
)
READER_REVIEW_PIN = (
    5_655,
    "b05b06a6e56888348e30ebe0ded0272a5e9f3ca709037feb975b938b4d0bb7d0",
)
READER_REVIEW_SCHEMA = "fg19.first-cell-score-result-capture-source-review.v4"
READER_REVIEW_STATUS = "PASS_SOURCE_ONLY_READER_V4_HARD_BOUNDED_UNBOUND"

ROOT_CAPTURE_DRIVER = (
    C / "preparation/first_cell_evaluation_score_capture_v2/ROOT_CAPTURE_SCORE_STATUS_V2.py"
)
ROOT_CAPTURE_DRIVER_PIN = (
    8_895,
    "e7b479e8237e6ca1f23417ce7a685c3e08fc4bb772fcfc33775634e680f8748c",
)
ROOT_CAPTURE_DRIVER_REVIEW = (
    C
    / "preparation/first_cell_evaluation_score_capture_v2/ROOT_SCORE_STATUS_DRIVER_SOURCE_REVIEW_V2.json"
)
ROOT_CAPTURE_DRIVER_REVIEW_PIN = (
    5_662,
    "ae7d68b74b61423d4ef15f93ca9e1c30581287229fd03d9028c2a80ee54b264e",
)
ROOT_CAPTURE_DRIVER_REVIEW_SCHEMA = "fg19.first-cell-score-root-status-driver-source-review.v2"
ROOT_CAPTURE_DRIVER_REVIEW_STATUS = (
    "PASS_SOURCE_ONLY_SCORE_STATUS_DRIVER_V2_PINNED_READER_V4_CALLER_PATH_CONSTRAINED"
)

SCORE_ADMISSION_BODY = (
    C / "preparation/first_cell_evaluation_score_admission_v12/REMOTE_SUBMISSION_V12.py"
)
SCORE_ADMISSION_REVIEW = (
    C
    / "preparation/first_cell_evaluation_score_admission_v12/ACTUAL_SCORE_ADMISSION_SOURCE_REVIEW_V12.json"
)
SCORE_ADMISSION_REVIEW_SCHEMA = "fg19.first-cell-score-admission-actual-source-review.v12"
SCORE_ADMISSION_REVIEW_STATUS = "PASS_SOURCE_ONLY_ACTUAL_SCORE_BODY_V12"

SCORE_UNIT = "computer-vision-homework-fg19-first-cell-score-v1.service"
EXPORT_UNIT = "computer-vision-homework-fg19-first-cell-export-v3.service"
MAX_METADATA_BYTES = 1_000_000
MAX_CAPTURE_BYTES = 4_000_000
MAX_READER_BODY_BYTES = 1_000_000
MAX_SCORE_RESULT_BYTES = 1_000_000
MAX_STDERR_BYTES = 4_096


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fail(message: str) -> None:
    raise ValueError(message)


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_int(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{label} must be a strict integer >= {minimum}")
    return value


def _strict_equal(left: Any, right: Any) -> bool:
    """Compare parsed JSON values without bool/int equivalence."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _fail(f"nonstandard JSON number is forbidden: {value}")


def json_object(raw: bytes, label: str, cap: int = MAX_METADATA_BYTES) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > cap:
        _fail(f"{label} is empty, not bytes, or oversized")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def read(path: Path, cap: int = MAX_METADATA_BYTES) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
        _fail(f"unsafe or oversized local metadata input: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            _fail("local metadata changed before read")
        parts: list[bytes] = []
        remaining = cap + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            parts.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(parts)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size or len(raw) > cap:
        _fail("local metadata changed size during read")
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity:
        _fail("local metadata changed during read")
    return raw


def pin(path: Path, raw: bytes) -> dict[str, object]:
    return {"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": sha(raw)}


def require_fixed_review(
    path: Path,
    raw: bytes,
    expected_path: Path,
    expected_pin: tuple[int, str],
    schema: str,
    status: str,
    candidate_path: Path,
) -> dict[str, Any]:
    if path != expected_path or (len(raw), sha(raw)) != expected_pin:
        _fail(f"fixed independent review bytes/path differ: {expected_path.name}")
    review = json_object(raw, f"{expected_path.name} review")
    if review.get("schema") != schema or review.get("status") != status:
        _fail(f"fixed independent review schema/status differ: {expected_path.name}")
    candidate_relative = str(candidate_path.relative_to(ROOT))
    matches = []
    for row in review.get("reviewed_files", []):
        if isinstance(row, dict) and row.get("path") == candidate_relative:
            matches.append(row)
    if candidate_path == HELPER:
        expected_candidate = {
            "path": candidate_relative,
            "bytes": HELPER_PIN[0],
            "sha256": HELPER_PIN[1],
        }
        if len(matches) != 1 or any(
            matches[0].get(key) != value for key, value in expected_candidate.items()
        ):
            _fail("helper review does not identify the exact helper source")
    else:
        candidate = review.get("candidate")
        if not isinstance(candidate, dict) or any(
            candidate.get(key) != value
            for key, value in {
                "path": candidate_relative,
                "bytes": READER_TEMPLATE_PIN[0],
                "sha256": READER_TEMPLATE_PIN[1],
            }.items()
        ):
            _fail("reader review does not identify the exact V4 template")
    return review


def validate_score_launch_sources(
    body_raw: bytes,
    review_raw: bytes,
    acceptance_raw: bytes,
    transport_raw: bytes,
    stdout_raw: bytes,
    stderr_raw: bytes,
    expected_body_bytes: int,
    expected_body_sha256: str,
    expected_review_bytes: int,
    expected_review_sha256: str,
    admission_raw: bytes,
    launch_raw: bytes,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    body_relative = str(SCORE_ADMISSION_BODY.relative_to(ROOT))
    review_relative = str(SCORE_ADMISSION_REVIEW.relative_to(ROOT))
    if (len(body_raw), sha(body_raw)) != (expected_body_bytes, expected_body_sha256):
        _fail("actual score-admission body differs from root-bound source pins")
    if (len(review_raw), sha(review_raw)) != (expected_review_bytes, expected_review_sha256):
        _fail("actual score-admission review differs from root-bound source pins")
    review = json_object(review_raw, "actual score-admission source review")
    reviewed_body_rows = []
    candidate = review.get("candidate")
    if isinstance(candidate, dict):
        reviewed_body_rows.append(candidate)
    rows = review.get("reviewed_files", [])
    if isinstance(rows, list):
        reviewed_body_rows.extend(row for row in rows if isinstance(row, dict))
    body_rows = [row for row in reviewed_body_rows if row.get("path") == body_relative]
    if any(
        row.get("bytes") != expected_body_bytes or row.get("sha256") != expected_body_sha256
        for row in body_rows
    ):
        _fail("actual score-admission source review contains a conflicting body pin")
    if (
        review.get("schema") != SCORE_ADMISSION_REVIEW_SCHEMA
        or review.get("status") != SCORE_ADMISSION_REVIEW_STATUS
        or not body_rows
    ):
        _fail("actual score-admission source review does not bind the fixed reviewed body")

    acceptance = json_object(acceptance_raw, "score LAUNCH_ACCEPTANCE.json")
    launch = json_object(launch_raw, "score LAUNCH.json")
    admission = json_object(admission_raw, "score ADMISSION.json")
    if set(acceptance) != {
        "schema",
        "status",
        "created_utc",
        "unit",
        "invocation_id",
        "receipts",
        "body",
        "source_review",
        "raw_transport",
        "scientific_fits_started",
        "quality_result",
        "note",
    }:
        _fail("score launch acceptance key set differs")
    if (
        acceptance.get("schema") != "fg19.first-cell-score-launch-acceptance.v1"
        or acceptance.get("status") != "ACCEPTED_SINGLE_CPU_SCORE_LAUNCH_NOT_RESULT"
        or acceptance.get("unit") != SCORE_UNIT
        or type(acceptance.get("scientific_fits_started")) is not int
        or acceptance.get("scientific_fits_started") != 0
        or acceptance.get("quality_result") is not None
    ):
        _fail("score launch acceptance identity/status differs")
    if not _strict_equal(
        acceptance.get("body"),
        {"path": body_relative, "bytes": expected_body_bytes, "sha256": expected_body_sha256},
    ) or not _strict_equal(
        acceptance.get("source_review"),
        {
            "path": review_relative,
            "bytes": expected_review_bytes,
            "sha256": expected_review_sha256,
        },
    ):
        _fail("score launch acceptance does not bind the exact reviewed admission body")

    transport = json_object(transport_raw, "score launch TRANSPORT.json")
    expected_transport_keys = {
        "schema",
        "started_utc",
        "finished_utc",
        "body_bytes",
        "body_sha256",
        "source_review_bytes",
        "source_review_sha256",
        "ssh_returncode",
        "timed_out",
        "stdout_bytes",
        "stdout_sha256",
        "stderr_bytes",
        "stderr_sha256",
        "persisted_before_parse",
        "retry_performed",
    }
    if set(transport) != expected_transport_keys:
        _fail("score launch transport fields differ")
    if (
        transport.get("schema") != "fg19.first-cell-score-root-ssh-transport.v1"
        or type(transport.get("ssh_returncode")) is not int
        or transport.get("ssh_returncode") != 0
        or transport.get("timed_out") is not False
        or transport.get("persisted_before_parse") is not True
        or transport.get("retry_performed") is not False
        or len(stdout_raw) > MAX_CAPTURE_BYTES
        or len(stderr_raw) > MAX_STDERR_BYTES
        or stderr_raw
    ):
        _fail("score launch transport is not one successful, non-retried capture")
    for label, raw in (
        ("body", body_raw),
        ("source review", review_raw),
        ("stdout", stdout_raw),
        ("stderr", stderr_raw),
    ):
        byte_key = "source_review_bytes" if label == "source review" else f"{label}_bytes"
        hash_key = "source_review_sha256" if label == "source review" else f"{label}_sha256"
        if (transport.get(byte_key), transport.get(hash_key)) != (len(raw), sha(raw)):
            _fail(f"score launch transport {label} byte/SHA join differs")
    invocation = launch.get("invocation_id")
    receipt_pins = {
        "ADMISSION.json": {"bytes": len(admission_raw), "sha256": sha(admission_raw)},
        "LAUNCH.json": {"bytes": len(launch_raw), "sha256": sha(launch_raw)},
    }
    if (
        admission.get("schema") != "fg19.first-cell-score-admission.v1"
        or admission.get("unit") != SCORE_UNIT
        or launch.get("schema") != "fg19.first-cell-score-launch.v1"
        or launch.get("unit") != SCORE_UNIT
        or not _is_hex(invocation, 32)
        or launch.get("admission_sha256") != receipt_pins["ADMISSION.json"]["sha256"]
        or acceptance.get("invocation_id") != invocation
        or not _strict_equal(acceptance.get("receipts"), receipt_pins)
    ):
        _fail("score launch acceptance and exact original receipts do not join")
    if not _strict_equal(
        acceptance.get("raw_transport"),
        {"path": str(RECORD / "TRANSPORT.json"), **transport},
    ):
        _fail("score launch acceptance does not bind the exact durable raw transport")
    reply = json_object(stdout_raw, "score launch raw stdout", MAX_CAPTURE_BYTES)
    if set(reply) != {"unit", "outcome", "admission_base64", "launch_base64"} or (
        reply.get("unit"),
        reply.get("outcome"),
    ) != (SCORE_UNIT, "submitted"):
        _fail("score launch raw stdout is not the exact one-submission four-key reply")
    for name, key, expected_raw in (
        ("ADMISSION.json", "admission_base64", admission_raw),
        ("LAUNCH.json", "launch_base64", launch_raw),
    ):
        encoded = reply.get(key)
        if not isinstance(encoded, str):
            _fail(f"score launch raw stdout lacks {key}")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as error:
            raise ValueError(f"score launch {key} is invalid base64") from error
        if decoded != expected_raw or base64.b64encode(decoded).decode("ascii") != encoded:
            _fail(f"score launch raw stdout {name} does not join the original receipt")
    return acceptance, transport, review


def render_bound_reader(template_raw: bytes, actual_values: dict[str, object]) -> bytes:
    text = template_raw.decode("utf-8", errors="strict")
    original = ast.parse(text)
    lines = text.splitlines(keepends=True)
    substitutions = []
    original_values: dict[str, ast.expr] = {}
    for node in original.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in actual_values
        ):
            name = node.targets[0].id
            original_values[name] = node.value
            substitutions.append(
                (node.lineno - 1, node.end_lineno, f"{name} = {actual_values[name]!r}\n")
            )
    if set(original_values) != set(actual_values) or len(substitutions) != 5:
        _fail("reviewed reader template does not contain exactly the five binding constants")
    for start, end, replacement in reversed(substitutions):
        lines[start:end] = [replacement]
    rendered = "".join(lines).encode("utf-8")
    bound_tree = ast.parse(rendered.decode("utf-8", errors="strict"))
    normalized_count = 0
    for node in bound_tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in original_values
        ):
            node.value = original_values[node.targets[0].id]
            normalized_count += 1
    if normalized_count != 5 or ast.dump(bound_tree, include_attributes=False) != ast.dump(
        original, include_attributes=False
    ):
        _fail("bound score reader differs beyond the five actual receipt/invocation constants")
    return rendered


def validate_reader_binding(
    template_raw: bytes,
    review_raw: bytes,
    body_raw: bytes,
    ast_binding_raw: bytes,
    admission_raw: bytes,
    launch_raw: bytes,
) -> dict[str, Any]:
    admission = json_object(admission_raw, "score admission")
    launch = json_object(launch_raw, "score launch")
    invocation = launch.get("invocation_id")
    if (
        admission.get("schema") != "fg19.first-cell-score-admission.v1"
        or admission.get("unit") != SCORE_UNIT
        or launch.get("schema") != "fg19.first-cell-score-launch.v1"
        or launch.get("unit") != SCORE_UNIT
        or not _is_hex(invocation, 32)
    ):
        _fail("score receipt unit or actual invocation is malformed for reader binding")
    actual_values = {
        "SCORE_INVOCATION_ID": invocation,
        "SCORE_ADMISSION_BYTES": len(admission_raw),
        "SCORE_ADMISSION_SHA256": sha(admission_raw),
        "SCORE_LAUNCH_BYTES": len(launch_raw),
        "SCORE_LAUNCH_SHA256": sha(launch_raw),
    }
    expected_body = render_bound_reader(template_raw, actual_values)
    if body_raw != expected_body:
        _fail("captured REMOTE.py is not the exact five-literal V4 reader binding")
    proof = json_object(ast_binding_raw, "score reader AST_BINDING.json")
    expected_proof = {
        "schema": "fg19.first-cell-score-reader-actual-binding.v1",
        "status": "EXACT_FIVE_LITERAL_BINDING_AST_OTHERWISE_IDENTICAL",
        "template_bytes": len(template_raw),
        "template_sha256": sha(template_raw),
        "source_review_bytes": len(review_raw),
        "source_review_sha256": sha(review_raw),
        "actual_reader_bytes": len(body_raw),
        "actual_reader_sha256": sha(body_raw),
        "actual_literal_values": actual_values,
        "all_other_AST_identical": True,
    }
    if not _strict_equal(proof, expected_proof):
        _fail("reader AST binding receipt differs from the exact template, reviews, or receipts")
    return actual_values


def validate_status_transport(
    capture_raw: bytes,
    transport_raw: bytes,
    stderr_raw: bytes,
    reader_raw: bytes,
) -> dict[str, Any]:
    transport = json_object(transport_raw, "score status TRANSPORT.json")
    expected_keys = {
        "schema",
        "started_utc",
        "finished_utc",
        "body_bytes",
        "body_sha256",
        "ssh_returncode",
        "timed_out",
        "stdout_bytes",
        "stdout_sha256",
        "stderr_bytes",
        "stderr_sha256",
        "persisted_before_parse",
        "retry_performed",
    }
    if set(transport) != expected_keys:
        _fail("score status transport fields differ")
    if (
        transport.get("schema") != "fg19.first-cell-score-status-root-transport.v1"
        or type(transport.get("ssh_returncode")) is not int
        or transport.get("ssh_returncode") != 0
        or transport.get("timed_out") is not False
        or transport.get("persisted_before_parse") is not True
        or transport.get("retry_performed") is not False
        or len(capture_raw) > MAX_CAPTURE_BYTES
        or len(stderr_raw) > MAX_STDERR_BYTES
        or stderr_raw
    ):
        _fail("score status transport is not one successful, persisted, non-retried response")
    expected = {
        "body_bytes": len(reader_raw),
        "body_sha256": sha(reader_raw),
        "stdout_bytes": len(capture_raw),
        "stdout_sha256": sha(capture_raw),
        "stderr_bytes": len(stderr_raw),
        "stderr_sha256": sha(stderr_raw),
    }
    if any(transport.get(key) != value for key, value in expected.items()):
        _fail("score status raw stdout/stderr/reader body transport joins differ")
    for name in ("started_utc", "finished_utc"):
        value = transport.get(name)
        if not isinstance(value, str) or not value:
            _fail(f"score status transport {name} is malformed")
        try:
            dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"score status transport {name} is malformed") from error
    return transport


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--score-body-bytes", type=int, required=True)
    parser.add_argument("--score-body-sha256", required=True)
    parser.add_argument("--score-review-bytes", type=int, required=True)
    parser.add_argument("--score-review-sha256", required=True)
    args = parser.parse_args()
    if not args.capture.is_absolute() or args.capture.parent != RECORD:
        _fail("capture must be a direct child of the fixed ROOT score record")
    record_stat = RECORD.lstat()
    capture_stat = args.capture.lstat()
    if not stat.S_ISDIR(record_stat.st_mode) or not stat.S_ISDIR(capture_stat.st_mode):
        _fail("score record and capture must be real directories, not symlinks")
    if args.capture.resolve().parent != RECORD.resolve():
        _fail("score capture resolved outside the fixed ROOT score record")
    if (
        (RECORD / "SCORE_ACCEPTANCE.json").exists()
        or (RECORD / "SCORE_ACCEPTANCE.json").is_symlink()
        or (RECORD / "ROOT_ACCEPTANCE_BINDING.json").exists()
        or (RECORD / "ROOT_ACCEPTANCE_BINDING.json").is_symlink()
    ):
        _fail("ROOT score acceptance already exists; never repeat this action")
    _strict_int(args.score_body_bytes, "score body bytes", 1)
    _strict_int(args.score_review_bytes, "score review bytes", 1)
    if not _is_hex(args.score_body_sha256, 64) or not _is_hex(args.score_review_sha256, 64):
        _fail("root-bound score body/review SHA-256 is malformed")

    helper_raw = read(HELPER)
    if (len(helper_raw), sha(helper_raw)) != HELPER_PIN:
        _fail("reviewed stdlib helper source pin differs")
    helper_review_raw = read(HELPER_REVIEW)
    require_fixed_review(
        HELPER_REVIEW,
        helper_review_raw,
        HELPER_REVIEW,
        HELPER_REVIEW_PIN,
        HELPER_REVIEW_SCHEMA,
        HELPER_REVIEW_STATUS,
        HELPER,
    )
    reader_template_raw = read(READER_TEMPLATE, MAX_READER_BODY_BYTES)
    reader_review_raw = read(READER_REVIEW)
    require_fixed_review(
        READER_REVIEW,
        reader_review_raw,
        READER_REVIEW,
        READER_REVIEW_PIN,
        READER_REVIEW_SCHEMA,
        READER_REVIEW_STATUS,
        READER_TEMPLATE,
    )
    if (len(reader_template_raw), sha(reader_template_raw)) != READER_TEMPLATE_PIN:
        _fail("reviewed V4 bounded score reader template pin differs")
    driver_raw = read(ROOT_CAPTURE_DRIVER)
    driver_review_raw = read(ROOT_CAPTURE_DRIVER_REVIEW)
    if (len(driver_raw), sha(driver_raw)) != ROOT_CAPTURE_DRIVER_PIN or (
        len(driver_review_raw),
        sha(driver_review_raw),
    ) != ROOT_CAPTURE_DRIVER_REVIEW_PIN:
        _fail("reviewed ROOT status driver source or review pin differs")
    driver_review = json_object(driver_review_raw, "ROOT status-driver source review")
    if (
        driver_review.get("schema") != ROOT_CAPTURE_DRIVER_REVIEW_SCHEMA
        or driver_review.get("status") != ROOT_CAPTURE_DRIVER_REVIEW_STATUS
        or not any(
            isinstance(row, dict)
            and row.get("path") == str(ROOT_CAPTURE_DRIVER.relative_to(ROOT))
            and row.get("bytes") == ROOT_CAPTURE_DRIVER_PIN[0]
            and row.get("sha256") == ROOT_CAPTURE_DRIVER_PIN[1]
            for row in driver_review.get("reviewed_files", [])
        )
    ):
        _fail("ROOT score-status driver review does not pin its exact source")

    score_body_raw = read(SCORE_ADMISSION_BODY, MAX_READER_BODY_BYTES)
    score_review_raw = read(SCORE_ADMISSION_REVIEW)
    if (len(score_body_raw), sha(score_body_raw)) != (
        args.score_body_bytes,
        args.score_body_sha256,
    ) or (len(score_review_raw), sha(score_review_raw)) != (
        args.score_review_bytes,
        args.score_review_sha256,
    ):
        _fail("actual reviewed score-admission body/review differs from root-bound pins")

    inputs = {
        "result_raw": args.capture / "RESULT.json",
        "capture_raw": args.capture / "STDOUT.raw.json",
        "score_admission_raw": RECORD / "ADMISSION.json",
        "score_launch_raw": RECORD / "LAUNCH.json",
        "export_acceptance_raw": EXPORT / "EXPORT_ACCEPTANCE.json",
        "export_exact_stop_raw": EXPORT / "exact_stop/EXACT_STOP.json",
        "export_binding_raw": C
        / "preparation/first_cell_evaluation_export_v4/EXPORT_BINDING_V4.json",
        "export_manifest_raw": EXPORT / "status_20261021T1107Z/EXPORT_MANIFEST_V1.json",
    }
    values = {
        key: read(path, MAX_CAPTURE_BYTES if key == "capture_raw" else MAX_SCORE_RESULT_BYTES)
        for key, path in inputs.items()
    }
    transport_raw = read(args.capture / "TRANSPORT.json")
    stderr_raw = read(args.capture / "STDERR.raw.txt", MAX_STDERR_BYTES)
    reader_raw = read(args.capture / "REMOTE.py", MAX_READER_BODY_BYTES)
    ast_binding_raw = read(args.capture / "AST_BINDING.json")
    launch_acceptance_raw = read(RECORD / "LAUNCH_ACCEPTANCE.json")
    launch_transport_raw = read(RECORD / "TRANSPORT.json")
    launch_stdout_raw = read(RECORD / "STDOUT.raw.json", MAX_CAPTURE_BYTES)
    launch_stderr_raw = read(RECORD / "STDERR.raw.txt", MAX_STDERR_BYTES)

    score_admission_raw = values["score_admission_raw"]
    score_launch_raw = values["score_launch_raw"]
    launch_acceptance, launch_transport, _ = validate_score_launch_sources(
        score_body_raw,
        score_review_raw,
        launch_acceptance_raw,
        launch_transport_raw,
        launch_stdout_raw,
        launch_stderr_raw,
        args.score_body_bytes,
        args.score_body_sha256,
        args.score_review_bytes,
        args.score_review_sha256,
        score_admission_raw,
        score_launch_raw,
    )
    actual_reader_values = validate_reader_binding(
        reader_template_raw,
        reader_review_raw,
        reader_raw,
        ast_binding_raw,
        score_admission_raw,
        score_launch_raw,
    )
    validate_status_transport(values["capture_raw"], transport_raw, stderr_raw, reader_raw)
    if read(inputs["result_raw"], MAX_SCORE_RESULT_BYTES) != values["result_raw"]:
        _fail("original captured RESULT.json changed before acceptance creation")

    # Compile the exact source bytes whose pin was checked; do not let a path-based
    # loader import a concurrently replaced helper file.
    helper_source = helper_raw.decode("utf-8", errors="strict")
    module = types.ModuleType("fg19_root_score_acceptance_v1")
    module.__file__ = str(HELPER)
    exec(compile(helper_source, str(HELPER), "exec"), module.__dict__)
    helper_after_raw = read(HELPER)
    if (len(helper_after_raw), sha(helper_after_raw)) != HELPER_PIN:
        _fail("reviewed helper changed across source-bound load")
    acceptance = module.build_root_acceptance(**values)
    acceptance_raw = module.write_root_acceptance_exclusive(
        RECORD / "SCORE_ACCEPTANCE.json", acceptance
    )

    # These two create-only receipts cannot be made atomic together. If the
    # binding write fails after acceptance is durable, preserve both paths and
    # use a separately reviewed binding-only recovery if one is needed.
    local_inputs = {key: pin(inputs[key], raw) for key, raw in values.items()}
    binding = {
        "schema": "fg19.first-cell-score-root-local-acceptance-binding.v2",
        "status": "EXACT_REVIEWED_CALLER_HELPER_READER_AND_ORIGINAL_METADATA_BOUND",
        "helper": pin(HELPER, helper_raw),
        "helper_review": pin(HELPER_REVIEW, helper_review_raw),
        "reader_template": pin(READER_TEMPLATE, reader_template_raw),
        "reader_review": pin(READER_REVIEW, reader_review_raw),
        "root_capture_driver": pin(ROOT_CAPTURE_DRIVER, driver_raw),
        "root_capture_driver_review": pin(ROOT_CAPTURE_DRIVER_REVIEW, driver_review_raw),
        "actual_score_admission_body": pin(SCORE_ADMISSION_BODY, score_body_raw),
        "actual_score_admission_source_review": pin(SCORE_ADMISSION_REVIEW, score_review_raw),
        "actual_score_launch_acceptance": pin(
            RECORD / "LAUNCH_ACCEPTANCE.json", launch_acceptance_raw
        ),
        "actual_score_launch_transport": pin(RECORD / "TRANSPORT.json", launch_transport_raw),
        "actual_score_launch_stdout": pin(RECORD / "STDOUT.raw.json", launch_stdout_raw),
        "actual_score_launch_stderr": pin(RECORD / "STDERR.raw.txt", launch_stderr_raw),
        "actual_reader_literals": actual_reader_values,
        "score_launch_identity": {
            "unit": launch_acceptance["unit"],
            "invocation_id": launch_acceptance["invocation_id"],
            "body": launch_acceptance["body"],
            "source_review": launch_acceptance["source_review"],
            "raw_transport_schema": launch_transport["schema"],
        },
        "score_status_transport": pin(args.capture / "TRANSPORT.json", transport_raw),
        "score_status_reader": pin(args.capture / "REMOTE.py", reader_raw),
        "score_status_ast_binding": pin(args.capture / "AST_BINDING.json", ast_binding_raw),
        "score_status_capture": pin(args.capture / "STDOUT.raw.json", values["capture_raw"]),
        "score_status_stderr": pin(args.capture / "STDERR.raw.txt", stderr_raw),
        "score_launch_acceptance": pin(RECORD / "LAUNCH_ACCEPTANCE.json", launch_acceptance_raw),
        "score_launch_transport_joined": True,
        "score_status_transport_joined": True,
        "inputs": local_inputs,
        "acceptance": pin(RECORD / "SCORE_ACCEPTANCE.json", acceptance_raw),
        "result_raw_preserved_unchanged": True,
        "campaign_scorer_imported": False,
        "array_payload_read": False,
        "remote_action_performed": False,
    }
    binding_raw = (json.dumps(binding, indent=2, allow_nan=False) + "\n").encode("utf-8")
    binding_path = RECORD / "ROOT_ACCEPTANCE_BINDING.json"
    descriptor = os.open(binding_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(binding_raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(RECORD, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps({"acceptance": binding["acceptance"], "result": acceptance["result"]}))


if __name__ == "__main__":
    main()
