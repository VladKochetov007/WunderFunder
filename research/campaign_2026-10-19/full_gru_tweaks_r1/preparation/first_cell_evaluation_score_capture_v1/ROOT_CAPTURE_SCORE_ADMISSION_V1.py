from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
UNIT = "computer-vision-homework-fg19-first-cell-score-v1.service"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--body-sha", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha", required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    body = args.body.read_bytes()
    review_raw = args.review.read_bytes()
    if sha(body) != args.body_sha or sha(review_raw) != args.review_sha:
        raise ValueError("exact reviewed input pin changed before submission")
    review = json.loads(review_raw)
    if not str(review["status"]).startswith("PASS_SOURCE_ONLY"):
        raise ValueError("independent exact body source gate did not pass")
    if args.body_sha not in review_raw.decode("utf-8"):
        raise ValueError("body SHA is not bound by the source review")
    args.record.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    timed_out = False
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
            input=body,
            capture_output=True,
            timeout=110,
            check=False,
        )
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        stdout, stderr, code = error.stdout or b"", error.stderr or b"", None
        timed_out = True
    finished = datetime.now(UTC).isoformat()
    write(args.record / "STDOUT.raw.json", stdout)
    write(args.record / "STDERR.raw.txt", stderr)
    transport = {
        "schema": "fg19.first-cell-score-root-ssh-transport.v1",
        "started_utc": started,
        "finished_utc": finished,
        "body_bytes": len(body),
        "body_sha256": sha(body),
        "source_review_bytes": len(review_raw),
        "source_review_sha256": sha(review_raw),
        "ssh_returncode": code,
        "timed_out": timed_out,
        "stdout_bytes": len(stdout),
        "stdout_sha256": sha(stdout),
        "stderr_bytes": len(stderr),
        "stderr_sha256": sha(stderr),
        "persisted_before_parse": True,
        "retry_performed": False,
    }
    write(args.record / "TRANSPORT.json", (json.dumps(transport, indent=2) + "\n").encode())
    descriptor = os.open(args.record, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if code != 0 or timed_out:
        raise RuntimeError(
            f"Unknown/failed admission; inspect exact durable records, never retry: {transport}"
        )
    if len(stdout) > 4_000_000 or stderr:
        raise ValueError("Unexpected bounded outer reply; durable raw retained, never retry")
    reply = json.loads(stdout)
    if set(reply) != {"unit", "outcome", "admission_base64", "launch_base64"} or (
        reply["unit"],
        reply["outcome"],
    ) != (UNIT, "submitted"):
        raise ValueError("Exact four-key reply differs; inspect durable outcome, never retry")
    receipts = {}
    values = {}
    for filename, key in (("ADMISSION.json", "admission_base64"), ("LAUNCH.json", "launch_base64")):
        raw = base64.b64decode(reply[key], validate=True)
        value = json.loads(raw)
        if value["unit"] != UNIT:
            raise ValueError("Original receipt unit differs")
        write(args.record / filename, raw)
        receipts[filename] = {"bytes": len(raw), "sha256": sha(raw)}
        values[filename] = value
    admission, launch = values["ADMISSION.json"], values["LAUNCH.json"]
    invocation = launch["invocation_id"]
    if (
        not isinstance(invocation, str)
        or len(invocation) != 32
        or any(c not in "0123456789abcdef" for c in invocation)
    ):
        raise ValueError("Actual invocation invalid; inspect durable outcome, never retry")
    if (
        admission["schema"] != "fg19.first-cell-score-admission.v1"
        or launch["schema"] != "fg19.first-cell-score-launch.v1"
    ):
        raise ValueError("Actual receipt schemas differ")
    if launch["admission_sha256"] != receipts["ADMISSION.json"]["sha256"]:
        raise ValueError("Actual launch does not join original admission bytes")
    acceptance = {
        "schema": "fg19.first-cell-score-launch-acceptance.v1",
        "status": "ACCEPTED_SINGLE_CPU_SCORE_LAUNCH_NOT_RESULT",
        "created_utc": datetime.now(UTC).isoformat(),
        "unit": UNIT,
        "invocation_id": invocation,
        "receipts": receipts,
        "body": {"path": str(args.body.relative_to(ROOT)), "bytes": len(body), "sha256": sha(body)},
        "source_review": {
            "path": str(args.review.relative_to(ROOT)),
            "bytes": len(review_raw),
            "sha256": sha(review_raw),
        },
        "raw_transport": {"path": str(args.record / "TRANSPORT.json"), **transport},
        "scientific_fits_started": 0,
        "quality_result": None,
        "note": "ROOT acceptance binds exact stdin bytes; no worker stdin self-read or worker body-SHA claim.",
    }
    raw = (json.dumps(acceptance, indent=2) + "\n").encode()
    write(args.record / "LAUNCH_ACCEPTANCE.json", raw)
    descriptor = os.open(args.record, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(
        json.dumps(
            {
                "unit": UNIT,
                "invocation_id": invocation,
                "transport": transport,
                "receipts": receipts,
                "root_acceptance": {"bytes": len(raw), "sha256": sha(raw)},
            }
        )
    )


if __name__ == "__main__":
    main()
