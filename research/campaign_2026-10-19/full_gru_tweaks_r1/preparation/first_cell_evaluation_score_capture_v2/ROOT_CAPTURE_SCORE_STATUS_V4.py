from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
C = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1"
TEMPLATE = (
    C / "preparation/first_cell_evaluation_score_capture_v6/SCORE_RESULT_CAPTURE_TEMPLATE_V6.py"
)
TEMPLATE_SHA = "049e2e9245356a4ac5f39596348e186d7af9a1e04aeb747255b7f62843324de0"
REVIEW = TEMPLATE.parent / "INDEPENDENT_SOURCE_REVIEW_V6.json"
REVIEW_SHA = "5e067c1d9972be3f00f462a075084b5ea315e3708bdb18669a0cf52be1a2d8ed"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    template, review = TEMPLATE.read_bytes(), REVIEW.read_bytes()
    if sha(template) != TEMPLATE_SHA or sha(review) != REVIEW_SHA:
        raise ValueError("conditional metadata-reader source/review pin differs")
    admission_raw = (args.record / "ADMISSION.json").read_bytes()
    launch_raw = (args.record / "LAUNCH.json").read_bytes()
    launch = json.loads(launch_raw)
    actual_values = {
        "SCORE_INVOCATION_ID": launch["invocation_id"],
        "SCORE_ADMISSION_BYTES": len(admission_raw),
        "SCORE_ADMISSION_SHA256": sha(admission_raw),
        "SCORE_LAUNCH_BYTES": len(launch_raw),
        "SCORE_LAUNCH_SHA256": sha(launch_raw),
    }
    text = template.decode()
    original = ast.parse(text)
    lines = text.splitlines(keepends=True)
    substitutions = []
    for node in original.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name in actual_values:
                substitutions.append(
                    (node.lineno - 1, node.end_lineno, f"{name} = {actual_values[name]!r}\n")
                )
    if len(substitutions) != 5:
        raise ValueError("conditional reader does not have exactly five binding constants")
    for start, end, replacement in reversed(substitutions):
        lines[start:end] = [replacement]
    body = "".join(lines).encode()
    bound = ast.parse(body)
    original_constants = {
        node.targets[0].id: node.value
        for node in original.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in actual_values
    }
    for node in bound.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in original_constants
        ):
            node.value = original_constants[node.targets[0].id]
    if ast.dump(bound, include_attributes=False) != ast.dump(original, include_attributes=False):
        raise ValueError("reader changed beyond the five actual receipt/invocation literals")
    if not args.tag or Path(args.tag).name != args.tag:
        raise ValueError("capture tag must be one private folder name")
    record = args.record / args.tag
    record.mkdir(exist_ok=False)
    write(record / "REMOTE.py", body)
    proof = {
        "schema": "fg19.first-cell-score-reader-actual-binding.v1",
        "status": "EXACT_FIVE_LITERAL_BINDING_AST_OTHERWISE_IDENTICAL",
        "template_bytes": len(template),
        "template_sha256": sha(template),
        "source_review_bytes": len(review),
        "source_review_sha256": sha(review),
        "actual_reader_bytes": len(body),
        "actual_reader_sha256": sha(body),
        "actual_literal_values": actual_values,
        "all_other_AST_identical": True,
    }
    write(record / "AST_BINDING.json", (json.dumps(proof, indent=2) + "\n").encode())
    started, timed_out = datetime.now(UTC).isoformat(), False
    try:
        completed = subprocess.run(
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
            input=body,
            capture_output=True,
            timeout=85,
            check=False,
        )
        stdout, stderr, code = completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr, code = error.stdout or b"", error.stderr or b"", None
        timed_out = True
    finished = datetime.now(UTC).isoformat()
    write(record / "STDOUT.raw.json", stdout)
    write(record / "STDERR.raw.txt", stderr)
    transport = {
        "schema": "fg19.first-cell-score-status-root-transport.v1",
        "started_utc": started,
        "finished_utc": finished,
        "body_bytes": len(body),
        "body_sha256": sha(body),
        "ssh_returncode": code,
        "timed_out": timed_out,
        "stdout_bytes": len(stdout),
        "stdout_sha256": sha(stdout),
        "stderr_bytes": len(stderr),
        "stderr_sha256": sha(stderr),
        "persisted_before_parse": True,
        "retry_performed": False,
    }
    write(record / "TRANSPORT.json", (json.dumps(transport, indent=2) + "\n").encode())
    fd = os.open(record, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    if code != 0 or timed_out or stderr or len(stdout) > 4_000_000:
        raise RuntimeError(f"Exact bounded reader outcome retained; never blind retry: {transport}")
    observation = json.loads(stdout)
    if (
        observation.get("schema") != "fg19.first-cell-score-bounded-status.v1"
        or observation.get("invocation_id") != launch["invocation_id"]
        or observation.get("unit_name") != launch["unit"]
        or observation.get("unit", {}).get("Id") != launch["unit"]
        or observation["unit"].get("InvocationID") != launch["invocation_id"]
    ):
        raise ValueError("score reader native identity differs")
    files = observation.get("files")
    if not isinstance(files, dict) or set(files) not in (
        {"ADMISSION.json", "LAUNCH.json", "unit.log"},
        {"ADMISSION.json", "LAUNCH.json", "unit.log", "RESULT.json"},
    ):
        raise ValueError("score reader file membership differs")
    for name in ("ADMISSION.json", "LAUNCH.json"):
        item = files[name]
        raw = base64.b64decode(item["base64"], validate=True)
        if (len(raw), sha(raw)) != (item["bytes"], item["sha256"]) or raw != (
            args.record / name
        ).read_bytes():
            raise ValueError("strict original score receipt join differs")
        write(record / name, raw)
    log = files["unit.log"]
    log_raw = base64.b64decode(log["tail_base64"], validate=True)
    if (len(log_raw), sha(log_raw)) != (log["captured_bytes"], log["tail_sha256"]):
        raise ValueError("score log tail byte join differs")
    write(record / "unit.log.tail", log_raw)
    if "RESULT.json" in files:
        props = observation["unit"]
        if observation.get("terminal_success") is not True or (
            props.get("ActiveState"),
            props.get("SubState"),
            props.get("MainPID"),
            props.get("ExecMainCode"),
            props.get("ExecMainStatus"),
            props.get("Result"),
        ) != ("active", "exited", "0", "1", "0", "success"):
            raise ValueError("score result was not captured at strict native success")
        item = files["RESULT.json"]
        raw = base64.b64decode(item["base64"], validate=True)
        if (len(raw), sha(raw)) != (item["bytes"], item["sha256"]):
            raise ValueError("score result byte join differs")
        write(record / "RESULT.json", raw)
    summary = {
        "schema": observation["schema"],
        "captured_utc": observation["captured_utc"],
        "unit": observation["unit_name"],
        "invocation_id": observation["invocation_id"],
        "systemd": observation["unit"],
        "terminal_success": observation["terminal_success"],
        "result_capture": observation["result_capture"],
        "captured_files": list(files),
        "array_payloads_read_or_hashed": False,
        "transport": transport,
    }
    write(record / "OBSERVATION.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
