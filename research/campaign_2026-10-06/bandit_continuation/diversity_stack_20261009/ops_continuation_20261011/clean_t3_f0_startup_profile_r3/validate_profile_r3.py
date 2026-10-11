from __future__ import annotations

import copy
import hashlib
import importlib
import io
import json
import os
import tarfile
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
OPS = PACKET.parent
R1 = OPS / "clean_t3_fold0_h390_r1"
BUNDLE = PACKET / "bundle_profile_r3"
SOURCE = PACKET / "source_snapshot"
PROFILE_NAME = "clean_t3_f0_muonall_h384_e3_s3_t4_startup_profile_r3"
DATASET = "vladkochetov07/wnn-clean-t3-f0-startup-profile-20261011-r3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_digest_telemetry() -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(SOURCE / "src"))
    portability = importlib.import_module("wnn.portability")
    old_env = os.environ.get("WNN_STARTUP_DIAGNOSTICS")
    os.environ["WNN_STARTUP_DIAGNOSTICS"] = "1"
    old_interval = portability.PROGRESS_EVERY_BYTES
    portability.PROGRESS_EVERY_BYTES = 1 << 20
    payload = bytes(range(251)) * 10_000
    try:
        with tempfile.TemporaryDirectory(prefix="wnn-startup-profile-r3-") as temp:
            path = Path(temp) / "digest-fixture.bin"
            path.write_bytes(payload)
            stat = path.stat()
            captured = io.StringIO()
            with redirect_stderr(captured):
                observed = portability._content_digest(
                    str(path.resolve()),
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
            expected = hashlib.sha256(payload).hexdigest()
            if observed != expected:
                raise AssertionError("diagnostic full-content digest changed")
            events = [json.loads(line) for line in captured.getvalue().splitlines()]
            names = [event["event"] for event in events]
            if names[0] != "portable_sha256_begin" or names[-1] != "portable_sha256_complete":
                raise AssertionError("hash begin/complete telemetry missing")
            if "portable_sha256_progress" not in names:
                raise AssertionError("bounded hash progress event missing")
            if events[-1]["bytes_read"] != len(payload):
                raise AssertionError("hash telemetry byte count differs from fixture")
            return {
                "fixture_bytes": len(payload),
                "digest_matches_reference": True,
                "events": names,
            }
    finally:
        portability.PROGRESS_EVERY_BYTES = old_interval
        if old_env is None:
            os.environ.pop("WNN_STARTUP_DIAGNOSTICS", None)
        else:
            os.environ["WNN_STARTUP_DIAGNOSTICS"] = old_env


def validate() -> dict[str, Any]:
    manifest = json.loads((BUNDLE / "bundle-manifest.json").read_text())
    spec_path = PACKET / "teacher_profile_specs" / f"{PROFILE_NAME}.json"
    spec = json.loads(spec_path.read_text())
    old_spec = json.loads(
        (
            R1 / "teacher_profile_specs/clean_t3_f0_muonall_h384_e3_s3_t4_profile1_r1.json"
        ).read_text()
    )
    r1_manifest = json.loads((R1 / "bundle_teacher_profile_r1/bundle-manifest.json").read_text())
    kernel = json.loads((BUNDLE / "kernel/kernel-metadata.json").read_text())
    checks: list[str] = []

    if manifest["dataset"] != DATASET or kernel["id"] != f"{DATASET}-runner":
        raise AssertionError("dataset or title-derived kernel id mismatch")
    checks.append("title-derived private dataset/kernel id")
    if manifest["private"] is not True or kernel["is_private"] is not True:
        raise AssertionError("R3 profile bundle must remain private")
    checks.append("private metadata")

    archive = BUNDLE / "bundle.wnn"
    if sha256(archive) != manifest["archive_sha256"]:
        raise AssertionError("archive SHA-256 differs from bundle manifest")
    if archive.stat().st_size >= 1_000_000:
        raise AssertionError("kernel bundle exceeds the conservative 1 MB envelope")
    checks.append("archive digest and size envelope")

    if manifest["external_mounts"] != r1_manifest["external_mounts"]:
        raise AssertionError("external mounts differ from reviewed R1 profile inputs")
    if len(manifest["spec_paths"]) != 1:
        raise AssertionError("R3 must contain exactly one profile spec")
    checks.append("same external mounts and one worker")

    with tarfile.open(archive, "r:gz") as tar:
        member_name = manifest["spec_paths"][0]
        member = tar.extractfile(member_name)
        if member is None:
            raise AssertionError("profile spec missing from archive")
        archived_spec = json.loads(member.read())
    if archived_spec != spec:
        raise AssertionError("archived profile spec differs from local frozen recipe")

    expected = copy.deepcopy(old_spec)
    expected["name"] = PROFILE_NAME
    packet_prefix = PACKET.relative_to(PACKET.parents[5]).as_posix()
    expected["argv"][1] = f"{packet_prefix}/source_snapshot/scripts/cleanfold_exec.py"
    expected["argv"][expected["argv"].index("--name") + 1] = PROFILE_NAME
    expected["argv"].append("--grad-stats")
    expected["limits"]["timeout_seconds"] = 1800
    expected["environment"].update(
        {
            "PYTHONUNBUFFERED": "1",
            "WNN_STARTUP_DIAGNOSTICS": "1",
            "WNN_STARTUP_STACK_DUMP_SECONDS": "60",
        }
    )
    expected["code"] = spec["code"]
    expected["code_sha256"] = spec["code_sha256"]
    if spec != expected:
        raise AssertionError("recipe differs beyond the declared diagnostic/profile changes")
    if spec["inputs"] != old_spec["inputs"]:
        raise AssertionError("input membership or hashes changed")
    if spec["artifacts"]:
        raise AssertionError("profile spec unexpectedly declares fit artifacts")
    if spec["limits"]["timeout_seconds"] != 1800:
        raise AssertionError("worker startup/profile budget is not 1800 seconds")
    if "--grad-stats" not in spec["argv"]:
        raise AssertionError("finite-gradient diagnostics are not enabled")
    if (
        "--profile-steps" not in spec["argv"]
        or spec["argv"][spec["argv"].index("--profile-steps") + 1] != "25"
    ):
        raise AssertionError("profile update shape changed")
    if (
        "--profile-warmup" not in spec["argv"]
        or spec["argv"][spec["argv"].index("--profile-warmup") + 1] != "5"
    ):
        raise AssertionError("profile warmup changed")
    checks.append("profile command/inputs preserved with explicit diagnostic delta")

    external_paths = set(manifest["external_mounts"])
    for row in manifest["files"]:
        if row["path"] in external_paths:
            continue
        local = Path.cwd() / row["path"]
        if (
            not local.is_file()
            or local.stat().st_size != row["size"]
            or sha256(local) != row["sha256"]
        ):
            raise AssertionError(f"bundle code descriptor mismatch: {row['path']}")
    checks.append("all local code hashes and sizes match archive manifest")

    source_manifest_path = SOURCE / "SOURCE_MANIFEST_H390.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    observed_sources = {
        path.relative_to(SOURCE).as_posix(): sha256(path)
        for path in sorted(SOURCE.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    if source_manifest["sources"] != observed_sources:
        raise AssertionError("source snapshot manifest does not cover exact source tree")
    if "src/sitecustomize.py" not in observed_sources:
        raise AssertionError("diagnostic sitecustomize is not hash-bound")
    checks.append("diagnostic source snapshot fully hash-bound")

    parent_sources = {
        path.relative_to(R1 / "source_snapshot").as_posix(): sha256(path)
        for path in sorted((R1 / "source_snapshot").rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    allowed_deltas = {
        "scripts/train_rnn.py",
        "src/wnn/portability.py",
        "src/wnn/splits.py",
        "src/sitecustomize.py",
    }
    if set(observed_sources) - set(parent_sources) != {"src/sitecustomize.py"}:
        raise AssertionError("source snapshot added an unexpected file")
    for name, digest in observed_sources.items():
        if name not in allowed_deltas and parent_sources.get(name) != digest:
            raise AssertionError(f"unexpected source mutation: {name}")
    if not all(
        observed_sources.get(name) != parent_sources.get(name)
        for name in ("scripts/train_rnn.py", "src/wnn/portability.py", "src/wnn/splits.py")
    ):
        raise AssertionError("declared diagnostic source deltas were not applied")
    checks.append("only declared diagnostic source deltas from R1")

    portability_path = SOURCE / "src/wnn/portability.py"
    portability_text = portability_path.read_text()
    splits_text = (SOURCE / "src/wnn/splits.py").read_text()
    trainer_path = SOURCE / "scripts/train_rnn.py"
    trainer_text = trainer_path.read_text()
    parent_trainer_text = (R1 / "source_snapshot/scripts/train_rnn.py").read_text()
    old_profile_return = """        if args.profile_steps:
            med = timer.report(args.profile_warmup, args.batch)
            print("PROFILE", json.dumps({k: round(v, 2) for k, v in med.items()}), flush=True)
            return
"""
    new_profile_return = """        if args.profile_steps:
            update_counts = epoch_counter.finish(cumulative_updates)
            print("PROFILE_UPDATE_COUNTS", json.dumps(update_counts), flush=True)
            med = timer.report(args.profile_warmup, args.batch)
            print("PROFILE", json.dumps({k: round(v, 2) for k, v in med.items()}), flush=True)
            return
"""
    if parent_trainer_text.count(
        old_profile_return
    ) != 1 or trainer_text != parent_trainer_text.replace(
        old_profile_return, new_profile_return, 1
    ):
        raise AssertionError("trainer differs beyond the exact profile update-counter diagnostic")
    checks.append("profile-only update counter is the sole train_rnn delta")

    sitecustomize_text = (SOURCE / "src/sitecustomize.py").read_text()
    if "return sha256_file(Path(path), startup_progress=True)" not in portability_text:
        raise AssertionError("full portable input hash path was bypassed")
    if (
        "campaign_split_load_begin" not in splits_text
        or "campaign_split_load_complete" not in splits_text
    ):
        raise AssertionError("split startup phase telemetry missing")
    if "dump_traceback_later(interval, repeat=True" not in sitecustomize_text:
        raise AssertionError("periodic stack heartbeat missing")
    if 'print("PROFILE_UPDATE_COUNTS", json.dumps(update_counts), flush=True)' not in trainer_text:
        raise AssertionError("actual optimizer update counters are not emitted")
    checks.append("full hash and split verification retained with diagnostics")

    digest_test = verify_digest_telemetry()
    checks.append("tiny-file digest and byte-progress regression check")
    return {
        "schema": "wnn.kaggle.startup_profile_static_validation.r3",
        "status": "PASS",
        "checks": checks,
        "digest_fixture": digest_test,
        "dataset_id": DATASET,
        "kernel_id": kernel["id"],
        "archive_sha256": manifest["archive_sha256"],
        "bundle_manifest_sha256": sha256(BUNDLE / "bundle-manifest.json"),
        "source_manifest_sha256": sha256(source_manifest_path),
        "worker_count": 1,
        "worker_timeout_seconds": spec["limits"]["timeout_seconds"],
        "fit_admission": False,
        "provider_actions": 0,
    }


def main() -> None:
    receipt = validate()
    write_json(PACKET / "STATIC_VALIDATION_R3.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
