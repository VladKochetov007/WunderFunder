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
TEMPLATE = C / "preparation/first_cell_evaluation_admission_v1/READ_EXPORT_TEMPLATE_V4.py"
TEMPLATE_SHA = "208a185c19b3a7fa4fb620192d0dae22bd701a8856dc3f44aa0951ddb7b4254d"
REVIEW = TEMPLATE.parent / "THIRD_UNIT_ROOT_TOOLS_SOURCE_REVIEW_V1.json"
REVIEW_SHA = "8a548c071bac10223bfcbb3de327161d19bed123cd48ee35e03a371cddb8fc0f"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fsync_directory(path: Path) -> None:
    """Persist directory entries after create-only capture writes."""
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write(path: Path, raw: bytes) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError(f"capture parent is not an enumerated real directory: {path.parent}")
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def write_terminal_metadata(record: Path, terminal: dict[str, object]) -> list[str]:
    """Persist only the manifest and five fixed sidecars under enumerated arm dirs."""
    expected = {"EXPORT_MANIFEST_V1.json"} | {
        f"{arm}/PREDICTION_SIDECAR.json" for arm in ("control", "ema", "sparse", "nll", "allthree")
    }
    if not isinstance(terminal, dict) or set(terminal) != expected:
        raise ValueError("terminal metadata must contain the manifest and five fixed sidecars")
    decoded = {}
    for name in sorted(expected):
        item = terminal[name]
        if not isinstance(item, dict):
            raise ValueError("terminal metadata capture row is not an object")
        raw = base64.b64decode(item["base64"], validate=True)
        if type(item.get("bytes")) is not int or (len(raw), sha(raw)) != (
            item["bytes"],
            item["sha256"],
        ):
            raise ValueError("strict terminal metadata join differs")
        decoded[name] = raw
    arms = ("control", "ema", "sparse", "nll", "allthree")
    for arm in arms:
        os.mkdir(record / arm, 0o700)
    fsync_directory(record)
    write(record / "EXPORT_MANIFEST_V1.json", decoded["EXPORT_MANIFEST_V1.json"])
    for arm in arms:
        arm_dir = record / arm
        write(arm_dir / "PREDICTION_SIDECAR.json", decoded[f"{arm}/PREDICTION_SIDECAR.json"])
        fsync_directory(arm_dir)
    fsync_directory(record)
    return sorted(expected)


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
        "INVOCATION": launch["invocation_id"],
        "ADMISSION_BYTES": len(admission_raw),
        "ADMISSION_SHA256": sha(admission_raw),
        "LAUNCH_BYTES": len(launch_raw),
        "LAUNCH_SHA256": sha(launch_raw),
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
    fsync_directory(record.parent)
    write(record / "REMOTE.py", body)
    proof = {
        "schema": "fg19.first-cell-export-reader-actual-binding.v1",
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
        "schema": "fg19.first-cell-export-status-root-transport.v1",
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
        observation["invocation_id"] != launch["invocation_id"]
        or observation["unit"] != launch["unit"]
    ):
        raise ValueError("actual reader unit/invocation differs")
    for name in ("ADMISSION.json", "LAUNCH.json"):
        item = observation[name]
        raw = base64.b64decode(item["base64"], validate=True)
        if (len(raw), sha(raw)) != (item["bytes"], item["sha256"]) or raw != (
            args.record / name
        ).read_bytes():
            raise ValueError("strict original receipt base64 join differs")
        write(record / name, raw)
    log = observation["log_tail"]
    raw = base64.b64decode(log["base64"], validate=True)
    if (len(raw), sha(raw)) != (log["bytes"], log["sha256"]):
        raise ValueError("strict log-tail base64 join differs")
    write(record / "unit.log.tail", raw)
    metadata_files = write_terminal_metadata(record, observation["terminal_metadata"])
    summary = {
        "schema": observation["schema"],
        "captured_utc": observation["captured_utc"],
        "unit": observation["unit"],
        "invocation_id": observation["invocation_id"],
        "systemd": observation["systemd"],
        "terminal_metadata_files": metadata_files,
        "array_payloads_read_or_hashed": False,
        "transport": transport,
    }
    write(record / "OBSERVATION.json", (json.dumps(summary, indent=2) + "\n").encode())
    fsync_directory(record)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
