"""Apply the independently reviewed stdlib scalar-result validator on ROOT."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

ROOT = Path("/Users/nb/Documents/wunder/WunderFunder-maintenance")
C = ROOT / "research/campaign_2026-10-19/full_gru_tweaks_r1"
PREP = C / "preparation/first_cell_evaluation_score_acceptance_v1"
RECORD = C / "records/first_cell_evaluation_v1/score_v1"
EXPORT = C / "records/first_cell_evaluation_v1/export_v5"
HELPER = PREP / "score_result_acceptance.py"
HELPER_PIN = (45795, "5987d527382f0ce33e72b4c17ae020d03485df3057b398ae285940ead29d528b")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read(path: Path, cap: int = 1_000_000) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
        raise ValueError(f"unsafe or oversized local metadata input: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != identity:
            raise ValueError("local metadata changed before read")
        raw = os.read(descriptor, cap + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        len(raw) != before.st_size
        or len(raw) > cap
        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != identity
    ):
        raise ValueError("local metadata changed during read")
    return raw


def pin(path: Path, raw: bytes) -> dict[str, object]:
    return {"path": str(path.relative_to(ROOT)), "bytes": len(raw), "sha256": sha(raw)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--helper-review", type=Path, required=True)
    parser.add_argument("--helper-review-sha", required=True)
    args = parser.parse_args()
    if not args.capture.is_absolute() or args.capture.parent != RECORD:
        raise ValueError("capture must be a direct child of the one fixed ROOT score record")
    if not args.helper_review.is_absolute():
        raise ValueError("helper review path must be absolute")
    args.helper_review.relative_to(PREP)
    if (RECORD / "SCORE_ACCEPTANCE.json").exists() or (
        RECORD / "ROOT_ACCEPTANCE_BINDING.json"
    ).exists():
        raise ValueError("ROOT score acceptance already exists; never repeat this action")
    helper_raw = read(HELPER)
    review_raw = read(args.helper_review)
    review = json.loads(review_raw)
    if (len(helper_raw), sha(helper_raw)) != HELPER_PIN or (
        sha(review_raw) != args.helper_review_sha
        or not str(review.get("status", "")).startswith("PASS_SOURCE_ONLY")
        or HELPER_PIN[1] not in review_raw.decode()
    ):
        raise ValueError("independent exact helper source gate is not bound")
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
        key: read(path, 4_000_000 if key == "capture_raw" else 1_000_000)
        for key, path in inputs.items()
    }
    transport_raw = read(args.capture / "TRANSPORT.json")
    transport = json.loads(transport_raw)
    stderr_raw = read(args.capture / "STDERR.raw.txt")
    reader_raw = read(args.capture / "REMOTE.py")
    if (
        transport.get("ssh_returncode") != 0
        or transport.get("timed_out") is not False
        or transport.get("persisted_before_parse") is not True
        or transport.get("retry_performed") is not False
        or stderr_raw
        or (len(values["capture_raw"]), sha(values["capture_raw"]))
        != (transport.get("stdout_bytes"), transport.get("stdout_sha256"))
        or (len(stderr_raw), sha(stderr_raw))
        != (transport.get("stderr_bytes"), transport.get("stderr_sha256"))
        or (len(reader_raw), sha(reader_raw))
        != (transport.get("body_bytes"), transport.get("body_sha256"))
    ):
        raise ValueError("ROOT status raw-first transport/body joins differ")
    spec = importlib.util.spec_from_file_location("fg19_root_score_acceptance_v1", HELPER)
    if spec is None or spec.loader is None or Path(spec.origin or "") != HELPER:
        raise ValueError("local pure helper import origin differs")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__) != HELPER or (len(read(HELPER)), sha(read(HELPER))) != HELPER_PIN:
        raise ValueError("local pure helper changed during import")
    acceptance = module.build_root_acceptance(**values)
    acceptance_raw = module.write_root_acceptance_exclusive(
        RECORD / "SCORE_ACCEPTANCE.json", acceptance
    )
    binding = {
        "schema": "fg19.first-cell-score-root-local-acceptance-binding.v1",
        "status": "EXACT_REVIEWED_STDLIB_HELPER_AND_ORIGINAL_METADATA_INPUTS_BOUND",
        "helper": pin(HELPER, helper_raw),
        "helper_review": pin(args.helper_review, review_raw),
        "inputs": {key: pin(inputs[key], raw) for key, raw in values.items()},
        "status_transport": pin(args.capture / "TRANSPORT.json", transport_raw),
        "status_reader": pin(args.capture / "REMOTE.py", reader_raw),
        "acceptance": pin(RECORD / "SCORE_ACCEPTANCE.json", acceptance_raw),
        "result_raw_preserved_unchanged": read(inputs["result_raw"]) == values["result_raw"],
        "campaign_scorer_imported": False,
        "array_payload_read": False,
        "remote_action_performed": False,
    }
    raw = (json.dumps(binding, indent=2, allow_nan=False) + "\n").encode()
    with (RECORD / "ROOT_ACCEPTANCE_BINDING.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(RECORD, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps({"acceptance": binding["acceptance"], "result": acceptance["result"]}))


if __name__ == "__main__":
    main()
