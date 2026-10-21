"""Prepare and, only after two independent source gates, capture one exact export stop."""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import os
import re
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
C = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1"
PREPARATION = C / "preparation/first_cell_evaluation_admission_v1"
RECORD_ROOT = C / "records/first_cell_evaluation_v1/export_v4"
STOP_DIRECTORY = RECORD_ROOT / "exact_stop"
ACCEPTANCE_PATH = RECORD_ROOT / "EXPORT_ACCEPTANCE.json"
TEMPLATE_PATH = PREPARATION / "STOP_EXPORT_TEMPLATE_V2.py"
TEMPLATE_REVIEW_PATH = PREPARATION / "STOP_EXPORT_SOURCE_REVIEW_V2.json"
TEMPLATE_REVIEW_PROOF_PATH = PREPARATION / "SECOND_UNIT_READER_STOP_IDENTITY_PROOF_V1.json"
DRIVER_PATH = Path(__file__).resolve()
DRIVER_REVIEW_PATH = PREPARATION / "ROOT_STOP_EXPORT_ORCHESTRATOR_SOURCE_REVIEW_V1.json"
BOUND_BODY_PATH = STOP_DIRECTORY / "REMOTE.py"
PREPARATION_BINDING_PATH = STOP_DIRECTORY / "PREPARATION.json"
BODY_REVIEW_PATH = STOP_DIRECTORY / "SOURCE_REVIEW.json"
ATTEMPT_PATH = STOP_DIRECTORY / "EXECUTION_ATTEMPT.json"
STDOUT_PATH = STOP_DIRECTORY / "STDOUT.raw.json"
STDERR_PATH = STOP_DIRECTORY / "STDERR.raw.txt"
TRANSPORT_PATH = STOP_DIRECTORY / "TRANSPORT.json"
RECEIPT_PATH = STOP_DIRECTORY / "EXACT_STOP.json"

UNIT = "computer-vision-homework-fg19-first-cell-export-v2.service"
TEMPLATE_BYTES = 6_299
TEMPLATE_SHA256 = "3e4591816313f457926fbb56f0d22dea95a91eb7cca1315d5244469936d5a683"
TEMPLATE_REVIEW_BYTES = 2_833
TEMPLATE_REVIEW_SHA256 = "65bcbc6b6d0137c77a08ca2e43ef03671185cc9c22805308b88a319e5acf393a"
TEMPLATE_REVIEW_SCHEMA = "fg19.first-cell-export-stop-source-review.v2"
TEMPLATE_REVIEW_STATUS = "PASS_SOURCE_ONLY_STOP_V2_NEW_UNIT_FORK_UNBOUND"
TEMPLATE_REVIEW_PROOF_BYTES = 1_715
TEMPLATE_REVIEW_PROOF_SHA256 = "a9b06e92d95b009e62b8cc78291281d7e035a19726a16c285fddf029ac05cc3e"

DRIVER_REVIEW_SCHEMA = "fg19.first-cell-export-stop-orchestrator-source-review.v1"
DRIVER_REVIEW_STATUS = "PASS_SOURCE_ONLY_STOP_ORCHESTRATOR_V1"
BODY_REVIEW_SCHEMA = "fg19.first-cell-export-actual-stop-source-review.v1"
BODY_REVIEW_STATUS = "PASS_SOURCE_ONLY_ACTUAL_EXPORT_STOP_V1"
MAX_METADATA_BYTES = 4_000_000
MAX_REPLY_BYTES = 1_000_000


def sha(raw: bytes) -> str:
    """Return the SHA-256 of exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def require(condition: object, message: str) -> None:
    """Stop before mutation or action when a gate does not hold."""
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    """Return an aware UTC timestamp."""
    return datetime.now(UTC).isoformat()


def stable_file(path: Path, cap: int) -> tuple[bytes, dict[str, int]]:
    """Read one bounded regular file with no-follow and stable inode checks."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), f"not a regular file: {path}")
    require(not stat.S_ISLNK(before.st_mode), f"symlink refused: {path}")
    require(before.st_size <= cap, f"file exceeds cap: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        require(
            (opened.st_dev, opened.st_ino, opened.st_size)
            == (before.st_dev, before.st_ino, before.st_size),
            f"file changed before read: {path}",
        )
        raw = bytearray()
        while len(raw) <= cap:
            block = os.read(descriptor, min(65_536, cap + 1 - len(raw)))
            if not block:
                break
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(len(raw) == before.st_size and len(raw) <= cap, f"file changed or too large: {path}")
    require(
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
        f"file changed while read: {path}",
    )
    return bytes(raw), {
        "bytes": after.st_size,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
    }


def file_pin(path: Path, cap: int) -> tuple[bytes, dict[str, object]]:
    """Return exact file bytes and a repository-relative metadata pin."""
    raw, _ = stable_file(path, cap)
    return raw, {"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": sha(raw)}


def write_exclusive(path: Path, raw: bytes) -> None:
    """Durably create one new file without replacing an existing artifact."""
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def fsync_directory(path: Path) -> None:
    """Persist newly created directory entries."""
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_fixed_directory(path: Path) -> None:
    """Require the configured export record root to be an ordinary directory."""
    metadata = path.lstat()
    require(stat.S_ISDIR(metadata.st_mode), f"not a directory: {path}")
    require(not stat.S_ISLNK(metadata.st_mode), f"symlink directory refused: {path}")
    require(path.resolve(strict=True) == path, f"directory redirected: {path}")


def validate_template_and_review() -> tuple[bytes, dict[str, object], bytes, dict[str, object]]:
    """Check the reviewed V2 conditional stop source and its independent review."""
    template_raw, template_pin = file_pin(TEMPLATE_PATH, 100_000)
    review_raw, review_pin = file_pin(TEMPLATE_REVIEW_PATH, 100_000)
    proof_raw, proof_pin = file_pin(TEMPLATE_REVIEW_PROOF_PATH, 100_000)
    require(
        (len(template_raw), sha(template_raw)) == (TEMPLATE_BYTES, TEMPLATE_SHA256),
        "V2 stop template pin differs",
    )
    require(
        (len(review_raw), sha(review_raw)) == (TEMPLATE_REVIEW_BYTES, TEMPLATE_REVIEW_SHA256),
        "conditional V2 stop review pin differs",
    )
    require(
        (len(proof_raw), sha(proof_raw))
        == (TEMPLATE_REVIEW_PROOF_BYTES, TEMPLATE_REVIEW_PROOF_SHA256),
        "V2 new-unit identity proof pin differs",
    )
    review = json.loads(review_raw)
    proof = json.loads(proof_raw)
    require(isinstance(review, dict), "conditional stop review root is not an object")
    require(
        (review.get("schema"), review.get("status"))
        == (TEMPLATE_REVIEW_SCHEMA, TEMPLATE_REVIEW_STATUS),
        "conditional stop review status differs",
    )
    reviewed_files = review.get("reviewed_files")
    require(
        isinstance(reviewed_files, list) and len(reviewed_files) == 2,
        "conditional review file list differs",
    )
    expected_files = {
        str(TEMPLATE_PATH.relative_to(ROOT)): (TEMPLATE_BYTES, TEMPLATE_SHA256),
        str(TEMPLATE_REVIEW_PROOF_PATH.relative_to(ROOT)): (
            TEMPLATE_REVIEW_PROOF_BYTES,
            TEMPLATE_REVIEW_PROOF_SHA256,
        ),
    }
    parsed_files = {}
    for item in reviewed_files:
        require(isinstance(item, dict), "conditional review file row malformed")
        name = item.get("path")
        require(isinstance(name, str), "conditional review path is malformed")
        require(name not in parsed_files, "conditional review duplicates a file")
        parsed_files[name] = (item.get("bytes"), item.get("sha256"))
    require(
        parsed_files == expected_files, "conditional review does not bind exact template and proof"
    )
    require(isinstance(proof, dict), "V2 identity proof root is not an object")
    require(
        proof.get("schema") == "fg19.first-cell-export-second-unit-observer-identity-fork.v1"
        and proof.get("status") == "LOCAL_LITERAL_IDENTITY_ONLY_NOT_ADMITTED",
        "V2 new-unit identity proof schema/status differs",
    )
    proof_files = proof.get("files")
    require(
        isinstance(proof_files, list) and len(proof_files) == 2,
        "V2 identity proof file list differs",
    )
    stop_row = next(
        (
            row
            for row in proof_files
            if isinstance(row, dict) and row.get("new") == TEMPLATE_PATH.name
        ),
        None,
    )
    require(
        isinstance(stop_row, dict)
        and (stop_row.get("bytes"), stop_row.get("sha256")) == (TEMPLATE_BYTES, TEMPLATE_SHA256)
        and stop_row.get("inverse_AST_identical") is True
        and stop_row.get("inverse_bytes_identical") is True,
        "V2 stop template identity proof does not bind the exact fork",
    )
    return template_raw, template_pin, review_raw, review_pin


def validate_driver_review() -> tuple[bytes, dict[str, object]]:
    """Require an independent review of this exact local orchestrator before use."""
    body, candidate = file_pin(DRIVER_PATH, 1_000_000)
    review_raw, review_pin = file_pin(DRIVER_REVIEW_PATH, 100_000)
    review = json.loads(review_raw)
    require(isinstance(review, dict), "orchestrator source review root is not an object")
    require(
        (review.get("schema"), review.get("status"))
        == (DRIVER_REVIEW_SCHEMA, DRIVER_REVIEW_STATUS),
        "orchestrator source review did not pass its exact schema/status gate",
    )
    require(review.get("candidate") == candidate, "orchestrator review candidate differs")
    require(
        review.get("b_action_performed") is False, "orchestrator review must remain source-only"
    )
    return review_raw, review_pin


def validate_acceptance() -> tuple[bytes, dict[str, object], dict[str, object]]:
    """Read and validate the durable exact export acceptance from the fixed record."""
    raw, identity = file_pin(ACCEPTANCE_PATH, MAX_METADATA_BYTES)
    acceptance = json.loads(raw)
    require(isinstance(acceptance, dict), "export acceptance root is not an object")
    expected_keys = {
        "schema",
        "status",
        "created_utc",
        "unit",
        "invocation_id",
        "native_terminal",
        "source_review",
        "binding",
        "launch_acceptance",
        "native_raw_capture",
        "native_raw_transport",
        "manifest_capture",
        "membership_sha256",
        "evaluation_groups",
        "outputs",
        "exposure",
        "quality_result",
        "array_payloads_read_or_hashed_by_ROOT",
        "stop_status",
    }
    require(set(acceptance) == expected_keys, "export acceptance exact key set differs")
    invocation = acceptance.get("invocation_id")
    require(
        (acceptance.get("schema"), acceptance.get("status"), acceptance.get("unit"))
        == (
            "fg19.first-cell-export-acceptance.v1",
            "ACCEPTED_FIXED_FIVE_PREDICTION_EXPORTS_NOT_SCORED",
            UNIT,
        ),
        "export acceptance identity/status differs",
    )
    require(
        isinstance(invocation, str) and re.fullmatch(r"[0-9a-f]{32}", invocation) is not None,
        "export invocation is not strict lowercase 32-hex",
    )
    terminal = acceptance.get("native_terminal")
    require(
        isinstance(terminal, dict)
        and set(terminal)
        == {"active_state", "sub_state", "main_pid", "exec_main_code", "exec_main_status"}
        and terminal
        == {
            "active_state": "active",
            "sub_state": "exited",
            "main_pid": 0,
            "exec_main_code": 1,
            "exec_main_status": 0,
        },
        "successful native export terminal tuple differs",
    )
    require(
        all(
            type(terminal[key]) is int for key in ("main_pid", "exec_main_code", "exec_main_status")
        ),
        "successful native export terminal numeric fields must be strict integers",
    )
    require(
        acceptance.get("quality_result") is None
        and acceptance.get("stop_status") == "PENDING_EXACT_STOP"
        and acceptance.get("array_payloads_read_or_hashed_by_ROOT") is False
        and type(acceptance.get("evaluation_groups")) is int
        and acceptance["evaluation_groups"] == 468,
        "export acceptance is scored, already stopped, or has a different evaluation scope",
    )
    outputs = acceptance.get("outputs")
    require(
        isinstance(outputs, dict)
        and set(outputs) == {"control", "ema", "sparse", "nll", "allthree"},
        "accepted output manifest does not contain the fixed five arms",
    )
    return raw, identity, acceptance


def assignment_nodes(tree: ast.Module, names: set[str]) -> dict[str, ast.Assign | ast.AnnAssign]:
    """Locate one top-level assignment for each named binding constant."""
    nodes = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if isinstance(target, ast.Name) and target.id in names:
            require(target.id not in nodes, f"duplicate stop binding constant: {target.id}")
            nodes[target.id] = node
    require(set(nodes) == names, "stop source binding constant set differs")
    return nodes


def bind_body(template_raw: bytes, acceptance_raw: bytes, acceptance: dict[str, object]) -> bytes:
    """Bind exactly the invocation and three ROOT-acceptance literals into V2 source."""
    text = template_raw.decode("utf-8", errors="strict")
    original_tree = ast.parse(text, filename=str(TEMPLATE_PATH))
    names = {"INVOCATION", "ACCEPTANCE_BYTES", "ACCEPTANCE_SHA256", "ACCEPTANCE_BASE64"}
    original_nodes = assignment_nodes(original_tree, names)
    expected_template_literals = {
        "INVOCATION": "UNBOUND_ACTUAL_EXPORT_INVOCATION",
        "ACCEPTANCE_BYTES": 0,
        "ACCEPTANCE_SHA256": "UNBOUND_ROOT_EXPORT_ACCEPTANCE",
        "ACCEPTANCE_BASE64": "UNBOUND_ROOT_EXPORT_ACCEPTANCE_BASE64",
    }
    require(
        {name: ast.literal_eval(node.value) for name, node in original_nodes.items()}
        == expected_template_literals,
        "template does not contain the exact four unbound literals",
    )
    encoded_acceptance = base64.b64encode(acceptance_raw).decode("ascii")
    actual_values: dict[str, object] = {
        "INVOCATION": acceptance["invocation_id"],
        "ACCEPTANCE_BYTES": len(acceptance_raw),
        "ACCEPTANCE_SHA256": sha(acceptance_raw),
        "ACCEPTANCE_BASE64": encoded_acceptance,
    }
    lines = text.splitlines(keepends=True)
    edits = []
    for name, node in original_nodes.items():
        require(node.end_lineno is not None, f"missing source end line for {name}")
        replacement = f"{name} = {actual_values[name]!r}\n"
        edits.append((node.lineno - 1, node.end_lineno, replacement))
    for start, end, replacement in reversed(edits):
        lines[start:end] = [replacement]
    bound_raw = "".join(lines).encode("utf-8")
    bound_tree = ast.parse(bound_raw.decode("utf-8"), filename=str(BOUND_BODY_PATH))
    bound_nodes = assignment_nodes(bound_tree, names)
    for name, node in bound_nodes.items():
        node.value = copy.deepcopy(original_nodes[name].value)
    require(
        ast.dump(bound_tree, include_attributes=False)
        == ast.dump(original_tree, include_attributes=False),
        "bound stop source changed outside the four reviewed literals",
    )
    require(
        not any(
            isinstance(node, ast.Name) and node.id == "__file__" for node in ast.walk(bound_tree)
        ),
        "remote stdin stop body may not inspect its own file path",
    )
    return bound_raw


def preparation_record(
    template_pin: dict[str, object],
    template_review_pin: dict[str, object],
    driver_review_pin: dict[str, object],
    acceptance_raw: bytes,
    acceptance_pin: dict[str, object],
    acceptance: dict[str, object],
    body: bytes,
) -> dict[str, object]:
    """Build the exact four-literal AST binding record."""
    encoded = base64.b64encode(acceptance_raw).decode("ascii")
    body_pin = {
        "path": str(BOUND_BODY_PATH.relative_to(ROOT)),
        "bytes": len(body),
        "sha256": sha(body),
    }
    return {
        "schema": "fg19.first-cell-export-exact-stop-preparation.v1",
        "status": "PREPARED_EXACT_FOUR_LITERAL_BINDING_AST_OTHERWISE_IDENTICAL",
        "scope": "Local exact source binding only; no stop, SSH, worker query, or model/array/data payload access.",
        "unit": UNIT,
        "invocation_id": acceptance["invocation_id"],
        "template": template_pin,
        "conditional_template_review": template_review_pin,
        "orchestrator_review": driver_review_pin,
        "root_acceptance": acceptance_pin,
        "actual_literals": {
            "INVOCATION": acceptance["invocation_id"],
            "ACCEPTANCE_BYTES": acceptance_pin["bytes"],
            "ACCEPTANCE_SHA256": acceptance_pin["sha256"],
            "ACCEPTANCE_BASE64_ASCII_CHARACTERS": len(encoded),
            "ACCEPTANCE_BASE64_SHA256": sha(encoded.encode("ascii")),
        },
        "actual_stop_body": body_pin,
        "all_other_ast_identical": True,
        "source_review_contract": {
            "path": str(BODY_REVIEW_PATH.relative_to(ROOT)),
            "schema": BODY_REVIEW_SCHEMA,
            "status": BODY_REVIEW_STATUS,
            "candidate": body_pin,
            "preparation_binding_path": str(PREPARATION_BINDING_PATH.relative_to(ROOT)),
        },
    }


def prepare() -> dict[str, object]:
    """Create a stop-body folder only after the actual ROOT export acceptance exists."""
    validate_fixed_directory(RECORD_ROOT)
    _, driver_review_pin = validate_driver_review()
    template, template_pin, _, template_review_pin = validate_template_and_review()
    acceptance_raw, acceptance_pin, acceptance = validate_acceptance()
    body = bind_body(template, acceptance_raw, acceptance)
    require(
        not os.path.lexists(STOP_DIRECTORY),
        "exact-stop output already exists; inspect, never replace",
    )
    STOP_DIRECTORY.mkdir(mode=0o700, exist_ok=False)
    encoded_record = preparation_record(
        template_pin,
        template_review_pin,
        driver_review_pin,
        acceptance_raw,
        acceptance_pin,
        acceptance,
        body,
    )
    write_exclusive(BOUND_BODY_PATH, body)
    binding_raw = (json.dumps(encoded_record, sort_keys=True, indent=2) + "\n").encode("utf-8")
    write_exclusive(PREPARATION_BINDING_PATH, binding_raw)
    fsync_directory(STOP_DIRECTORY)
    return {
        "status": encoded_record["status"],
        "unit": UNIT,
        "invocation_id": acceptance["invocation_id"],
        "body": encoded_record["actual_stop_body"],
        "preparation_binding": {
            "path": str(PREPARATION_BINDING_PATH.relative_to(ROOT)),
            "bytes": len(binding_raw),
            "sha256": sha(binding_raw),
        },
        "remote_action_performed": False,
    }


def validate_body_review(
    review_raw: bytes,
    body_pin: dict[str, object],
    preparation_pin: dict[str, object],
) -> dict[str, object]:
    """Require the independent exact-body review for this bound stop source."""
    review = json.loads(review_raw)
    require(isinstance(review, dict), "actual stop source review root is not an object")
    require(
        (review.get("schema"), review.get("status")) == (BODY_REVIEW_SCHEMA, BODY_REVIEW_STATUS),
        "actual stop source review schema/status differs",
    )
    require(review.get("candidate") == body_pin, "actual stop review body pin differs")
    require(
        review.get("preparation_binding") == preparation_pin,
        "actual stop review preparation pin differs",
    )
    require(review.get("b_action_performed") is False, "source review must not perform B action")
    return review


def validate_remote_receipt(
    raw: bytes, acceptance: dict[str, object], acceptance_sha256: str
) -> dict[str, object]:
    """Check the exact stop JSON schema and unit-state transition after durable capture."""
    value = json.loads(raw.decode("utf-8", errors="strict"))
    require(isinstance(value, dict), "remote exact stop receipt is not an object")
    expected_keys = {
        "schema",
        "captured_utc",
        "unit",
        "invocation_id",
        "acceptance_sha256",
        "before",
        "after",
        "populated_before",
        "populated_after",
        "stop_returncode",
        "stop_stdout",
        "stop_stderr",
        "scope",
    }
    require(set(value) == expected_keys, "exact stop receipt field set differs")
    require(
        (value.get("schema"), value.get("unit"), value.get("invocation_id"))
        == ("fg19-first-cell-export-exact-owned-stop.v1", UNIT, acceptance["invocation_id"]),
        "exact stop schema/unit/invocation differs",
    )
    require(
        value.get("acceptance_sha256") == acceptance_sha256, "exact stop acceptance digest differs"
    )
    try:
        captured = datetime.fromisoformat(value["captured_utc"])
    except (TypeError, ValueError) as error:
        raise ValueError("exact stop timestamp is invalid") from error
    require(
        captured.tzinfo is not None and captured.utcoffset() is not None,
        "exact stop timestamp lacks timezone",
    )
    before = value.get("before")
    after = value.get("after")
    require(
        isinstance(before, dict)
        and (
            before.get("Id"),
            before.get("InvocationID"),
            before.get("ActiveState"),
            before.get("SubState"),
            before.get("MainPID"),
            before.get("ExecMainCode"),
            before.get("ExecMainStatus"),
            before.get("Result"),
        )
        == (UNIT, acceptance["invocation_id"], "active", "exited", "0", "1", "0", "success"),
        "pre-stop exact successful exited identity differs",
    )
    require(
        isinstance(after, dict)
        and (
            after.get("LoadState"),
            after.get("ActiveState"),
            after.get("SubState"),
            after.get("MainPID"),
            after.get("InvocationID"),
        )
        == ("not-found", "inactive", "dead", "0", ""),
        "post-stop exact unit absence tuple differs",
    )
    require(
        value.get("populated_before") == {}
        and value.get("populated_after") == {}
        and type(value.get("stop_returncode")) is int
        and value["stop_returncode"] == 0
        and isinstance(value.get("stop_stdout"), str)
        and isinstance(value.get("stop_stderr"), str),
        "exact stop return or all-UID campaign population differs",
    )
    require(
        value.get("scope")
        == "Only the one successful exited export unit. All old native/support and foreign units excluded; no file or payload mutation.",
        "exact stop scope statement differs",
    )
    return value


def execute() -> dict[str, object]:
    """Run one reviewed exact stop, persist raw streams before parsing, and never retry."""
    validate_fixed_directory(RECORD_ROOT)
    validate_fixed_directory(STOP_DIRECTORY)
    driver_review_raw, driver_review_pin = validate_driver_review()
    template, template_pin, template_review_raw, template_review_pin = (
        validate_template_and_review()
    )
    acceptance_raw, acceptance_pin, acceptance = validate_acceptance()
    body_raw, body_file_pin = file_pin(BOUND_BODY_PATH, MAX_METADATA_BYTES)
    prep_raw, prep_file_pin = file_pin(PREPARATION_BINDING_PATH, 100_000)
    preparation = json.loads(prep_raw)
    require(isinstance(preparation, dict), "exact stop preparation binding is malformed")
    expected_body = bind_body(template, acceptance_raw, acceptance)
    require(
        body_raw == expected_body, "prepared stop body is not exact four-literal template binding"
    )
    expected_body_pin = {
        "path": str(BOUND_BODY_PATH.relative_to(ROOT)),
        "bytes": len(body_raw),
        "sha256": sha(body_raw),
    }
    require(
        (
            preparation.get("schema"),
            preparation.get("status"),
            preparation.get("unit"),
            preparation.get("invocation_id"),
        )
        == (
            "fg19.first-cell-export-exact-stop-preparation.v1",
            "PREPARED_EXACT_FOUR_LITERAL_BINDING_AST_OTHERWISE_IDENTICAL",
            UNIT,
            acceptance["invocation_id"],
        )
        and preparation.get("actual_stop_body") == expected_body_pin
        and preparation.get("root_acceptance") == acceptance_pin
        and preparation.get("template") == template_pin
        and preparation.get("conditional_template_review") == template_review_pin
        and preparation.get("all_other_ast_identical") is True,
        "preparation binding does not match current inputs/body",
    )
    body_review_raw, body_review_pin = file_pin(BODY_REVIEW_PATH, 100_000)
    validate_body_review(body_review_raw, expected_body_pin, prep_file_pin)
    require(
        not any(
            os.path.lexists(path)
            for path in (ATTEMPT_PATH, STDOUT_PATH, STDERR_PATH, TRANSPORT_PATH, RECEIPT_PATH)
        ),
        "exact-stop remote attempt or output already exists; inspect, never retry",
    )
    attempt = {
        "schema": "fg19.first-cell-export-stop-attempt.v1",
        "status": "ONE_REMOTE_ATTEMPT_INTENT_DURABLY_RECORDED",
        "created_utc": utc_now(),
        "unit": UNIT,
        "invocation_id": acceptance["invocation_id"],
        "body": expected_body_pin,
        "preparation_binding": prep_file_pin,
        "actual_body_review": body_review_pin,
        "orchestrator_source_review": driver_review_pin,
        "root_acceptance": acceptance_pin,
        "retry_performed": False,
    }
    write_exclusive(ATTEMPT_PATH, (json.dumps(attempt, indent=2, sort_keys=True) + "\n").encode())
    fsync_directory(STOP_DIRECTORY)

    started = utc_now()
    timed_out = False
    launch_error = None
    try:
        result = subprocess.run(
            [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "-o",
                "ConnectionAttempts=1",
                "beleriand",
                "python3 -E -B -",
            ],
            input=body_raw,
            capture_output=True,
            timeout=90,
            check=False,
        )
        stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr, returncode = error.stdout or b"", error.stderr or b"", None
        timed_out = True
    except OSError as error:
        stdout, stderr, returncode = b"", b"", None
        launch_error = f"{type(error).__name__}: {error}"
    finished = utc_now()
    write_exclusive(STDOUT_PATH, stdout)
    write_exclusive(STDERR_PATH, stderr)
    transport = {
        "schema": "fg19.first-cell-export-stop-root-transport.v1",
        "started_utc": started,
        "finished_utc": finished,
        "body_bytes": len(body_raw),
        "body_sha256": sha(body_raw),
        "source_review_bytes": len(body_review_raw),
        "source_review_sha256": sha(body_review_raw),
        "orchestrator_source_review_bytes": len(driver_review_raw),
        "orchestrator_source_review_sha256": sha(driver_review_raw),
        "preparation_binding_bytes": len(prep_raw),
        "preparation_binding_sha256": sha(prep_raw),
        "root_acceptance_bytes": len(acceptance_raw),
        "root_acceptance_sha256": sha(acceptance_raw),
        "ssh_returncode": returncode,
        "timed_out": timed_out,
        "launch_error": launch_error,
        "stdout_bytes": len(stdout),
        "stdout_sha256": sha(stdout),
        "stderr_bytes": len(stderr),
        "stderr_sha256": sha(stderr),
        "persisted_before_parse": True,
        "retry_performed": False,
    }
    write_exclusive(
        TRANSPORT_PATH, (json.dumps(transport, indent=2, sort_keys=True) + "\n").encode()
    )
    fsync_directory(STOP_DIRECTORY)
    if returncode != 0 or timed_out or launch_error is not None:
        raise RuntimeError(
            f"Exact stop outcome unknown or failed; durable capture retained, never retry: {transport}"
        )
    if stderr or len(stdout) > MAX_REPLY_BYTES:
        raise ValueError(
            f"Unexpected bounded stop reply; durable capture retained, never retry: {transport}"
        )
    receipt = validate_remote_receipt(stdout, acceptance, acceptance_pin["sha256"])
    write_exclusive(RECEIPT_PATH, stdout)
    fsync_directory(STOP_DIRECTORY)
    return {
        "schema": receipt["schema"],
        "unit": UNIT,
        "invocation_id": acceptance["invocation_id"],
        "receipt": {
            "path": str(RECEIPT_PATH.relative_to(ROOT)),
            "bytes": len(stdout),
            "sha256": sha(stdout),
        },
        "transport": transport,
        "acceptance": acceptance_pin,
        "source_review": body_review_pin,
        "remote_action_performed": True,
    }


def main() -> None:
    """Require an explicit one-time phase; no default action is performed."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--prepare", action="store_true", help="Bind the exact ROOT acceptance; no SSH."
    )
    group.add_argument(
        "--execute", action="store_true", help="One reviewed exact stop; never retry."
    )
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(), sort_keys=True))
    else:
        print(json.dumps(execute(), sort_keys=True))


if __name__ == "__main__":
    main()
