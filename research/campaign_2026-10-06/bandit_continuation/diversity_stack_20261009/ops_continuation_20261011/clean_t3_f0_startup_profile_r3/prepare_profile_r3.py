from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent
OPS = PACKET.parent
R1 = OPS / "clean_t3_fold0_h390_r1"
R1_SOURCE = R1 / "source_snapshot"
R1_MANIFEST = R1 / "bundle_teacher_profile_r1/bundle-manifest.json"
R1_SPEC = R1 / "teacher_profile_specs/clean_t3_f0_muonall_h384_e3_s3_t4_profile1_r1.json"
SOURCE = PACKET / "source_snapshot"
DATASET = "vladkochetov07/wnn-clean-t3-f0-startup-profile-20261011-r3"
PROFILE_NAME = "clean_t3_f0_muonall_h384_e3_s3_t4_startup_profile_r3"
PROGRESS_EVERY_BYTES = 1 << 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def patch_source() -> dict[str, str]:
    portability_path = SOURCE / "src/wnn/portability.py"
    portability_text = portability_path.read_text()
    if "portable_sha256_begin" not in portability_text:
        old_imports = (
            "import hashlib\nimport json\nfrom functools import lru_cache\n"
            "from pathlib import Path\n"
        )
        new_imports = (
            "import hashlib\nimport json\nimport os\nimport sys\nimport time\n"
            "from functools import lru_cache\nfrom pathlib import Path\n"
        )
        if old_imports not in portability_text:
            raise ValueError("unexpected portability.py imports")
        portability_text = portability_text.replace(old_imports, new_imports, 1)
        old_hash = '''def sha256_file(path: Path) -> str:
    """Hash a file without materializing its contents in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
'''
        new_hash = '''PROGRESS_EVERY_BYTES = 1 << 30


def _startup_event(event: str, **fields: object) -> None:
    """Emit opt-in startup telemetry without changing checked input bytes."""
    if os.environ.get("WNN_STARTUP_DIAGNOSTICS") != "1":
        return
    print(json.dumps({"event": event, **fields}, sort_keys=True), file=sys.stderr, flush=True)


def sha256_file(path: Path, *, startup_progress: bool = False) -> str:
    """Hash a file without materializing it; optionally report bounded progress."""
    digest = hashlib.sha256()
    total = path.stat().st_size if startup_progress else 0
    started = time.monotonic()
    bytes_read = 0
    next_progress = PROGRESS_EVERY_BYTES
    if startup_progress:
        _startup_event("portable_sha256_begin", file=path.name, bytes_total=total)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
            bytes_read += len(block)
            if startup_progress and bytes_read >= next_progress:
                _startup_event(
                    "portable_sha256_progress",
                    file=path.name,
                    bytes_total=total,
                    bytes_read=bytes_read,
                    elapsed_seconds=round(time.monotonic() - started, 3),
                )
                next_progress += PROGRESS_EVERY_BYTES
    result = digest.hexdigest()
    if startup_progress:
        _startup_event(
            "portable_sha256_complete",
            file=path.name,
            bytes_total=total,
            bytes_read=bytes_read,
            elapsed_seconds=round(time.monotonic() - started, 3),
            sha256=result,
        )
    return result
'''
        if old_hash not in portability_text:
            raise ValueError("unexpected portability.py hashing implementation")
        portability_text = portability_text.replace(old_hash, new_hash, 1)
        old_call = "    return sha256_file(Path(path))\n"
        if old_call not in portability_text:
            raise ValueError("unexpected portable digest call")
        portability_text = portability_text.replace(
            old_call, "    return sha256_file(Path(path), startup_progress=True)\n", 1
        )
        portability_path.write_text(portability_text)

    splits_path = SOURCE / "src/wnn/splits.py"
    splits_text = splits_path.read_text()
    if "campaign_split_load_begin" not in splits_text:
        old_imports = (
            "import hashlib\nimport json\nimport os\nfrom collections.abc import Iterable\n"
        )
        new_imports = (
            "import hashlib\nimport json\nimport os\nimport sys\nimport time\n"
            "from collections.abc import Iterable\n"
        )
        if old_imports not in splits_text:
            raise ValueError("unexpected splits.py imports")
        splits_text = splits_text.replace(old_imports, new_imports, 1)
        start = (
            "    path = path.resolve()\n"
            "    raw = path.read_bytes()\n"
            "    manifest = json.loads(raw)\n"
        )
        start_new = (
            "    started = time.monotonic()\n"
            '    diagnostics = os.environ.get("WNN_STARTUP_DIAGNOSTICS") == "1"\n'
            "    if diagnostics:\n"
            '        print(json.dumps({"event": "campaign_split_load_begin", '
            '"split_manifest": path.name}), file=sys.stderr, flush=True)\n'
            "    path = path.resolve()\n"
            "    raw = path.read_bytes()\n"
            "    manifest = json.loads(raw)\n"
            "    if diagnostics:\n"
            '        print(json.dumps({"event": "campaign_split_sources", '
            '"source_count": len(manifest.get("sources", {}))}), '
            "file=sys.stderr, flush=True)\n"
        )
        if start not in splits_text:
            raise ValueError("unexpected load_campaign_split prologue")
        splits_text = splits_text.replace(start, start_new, 1)
        end = "    return result\n"
        end_new = (
            "    if diagnostics:\n"
            '        print(json.dumps({"event": "campaign_split_load_complete", '
            '"elapsed_seconds": round(time.monotonic() - started, 3), '
            '"fit_train_groups": len(result.train_groups), '
            '"fit_valid_groups": len(result.valid_groups), '
            '"evaluation_groups": len(result.evaluation_groups)}), '
            "file=sys.stderr, flush=True)\n"
            "    return result\n"
        )
        if end not in splits_text:
            raise ValueError("unexpected split return")
        splits_text = splits_text.replace(end, end_new, 1)
        splits_path.write_text(splits_text)

    sitecustomize = SOURCE / "src/sitecustomize.py"
    expected_sitecustomize = '''"""Profile-only stack heartbeat for long startup validation steps."""

from __future__ import annotations

import faulthandler
import json
import os
import sys

if os.environ.get("WNN_STARTUP_DIAGNOSTICS") == "1":
    interval = float(os.environ.get("WNN_STARTUP_STACK_DUMP_SECONDS", "60"))
    if interval <= 0:
        raise ValueError("startup stack interval must be positive")
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(interval, repeat=True, file=sys.stderr)
    print(
        json.dumps(
            {"event": "startup_stack_watchdog_armed", "interval_seconds": interval},
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
'''
    if not sitecustomize.is_file():
        sitecustomize.write_text(expected_sitecustomize)
    elif sitecustomize.read_text() != expected_sitecustomize:
        raise ValueError("sitecustomize diagnostic source differs from frozen text")

    trainer_path = SOURCE / "scripts/train_rnn.py"
    trainer_text = trainer_path.read_text()
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
    if "PROFILE_UPDATE_COUNTS" not in trainer_text:
        if old_profile_return not in trainer_text:
            raise ValueError("unexpected profile-return block in train_rnn.py")
        trainer_path.write_text(trainer_text.replace(old_profile_return, new_profile_return, 1))
    elif new_profile_return not in trainer_text:
        raise ValueError("profile update-counter source differs from frozen text")
    return {
        "portability.py": sha256(R1_SOURCE / "src/wnn/portability.py"),
        "splits.py": sha256(R1_SOURCE / "src/wnn/splits.py"),
        "train_rnn.py": sha256(R1_SOURCE / "scripts/train_rnn.py"),
    }


def descriptor(root: Path, path: str) -> dict[str, Any]:
    local = root / path
    return {"path": path, "size": local.stat().st_size, "sha256": sha256(local)}


def build(root: Path) -> None:
    root = root.resolve()
    r1_manifest = json.loads(R1_MANIFEST.read_text())
    r1_prefix = R1.relative_to(root).as_posix()
    packet_prefix = PACKET.relative_to(root).as_posix()
    external_map = r1_manifest["external_mounts"]
    external_paths = set(external_map)
    old_sources = patch_source()
    source_manifest_path = SOURCE / "SOURCE_MANIFEST_H390.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["sources"] = {
        path.relative_to(SOURCE).as_posix(): sha256(path)
        for path in sorted(SOURCE.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    write_json(source_manifest_path, source_manifest)

    code = []
    for row in r1_manifest["files"]:
        old_path = row["path"]
        if old_path in external_paths:
            continue
        new_path = old_path.replace(r1_prefix + "/", packet_prefix + "/", 1)
        code.append(descriptor(root, new_path))
    code.append(descriptor(root, f"{packet_prefix}/source_snapshot/src/sitecustomize.py"))
    code.sort(key=lambda row: row["path"])

    spec = json.loads(R1_SPEC.read_text())
    spec["name"] = PROFILE_NAME
    spec["argv"][1] = f"{packet_prefix}/source_snapshot/scripts/cleanfold_exec.py"
    spec["argv"][spec["argv"].index("--name") + 1] = PROFILE_NAME
    spec["argv"].append("--grad-stats")
    spec["code"] = code
    spec["limits"]["timeout_seconds"] = 1800
    spec["environment"].update(
        {
            "PYTHONUNBUFFERED": "1",
            "WNN_STARTUP_DIAGNOSTICS": "1",
            "WNN_STARTUP_STACK_DUMP_SECONDS": "60",
        }
    )
    sys.path.insert(0, str(root / "scripts"))
    from kaggle_common import canonical_sha256  # noqa: PLC0415

    spec["code_sha256"] = canonical_sha256(code)
    spec_path = PACKET / "teacher_profile_specs" / f"{PROFILE_NAME}.json"
    write_json(spec_path, spec)

    external_args = []
    asset_datasets = set()
    for logical, mapping in sorted(external_map.items()):
        if isinstance(mapping, str):
            external_args.append(f"{logical}={mapping}")
        else:
            external_args.append(f"{logical}={mapping['dataset']}:{mapping['filename']}")
            asset_datasets.add(mapping["dataset"])
    from kaggle_bundle import build_bundle  # noqa: PLC0415

    bundle = PACKET / "bundle_profile_r3"
    if bundle.exists():
        raise FileExistsError(f"refusing to overwrite bundle {bundle}")
    build_bundle(
        root=root,
        specs=[spec_path],
        destination=bundle,
        dataset=DATASET,
        external=external_args,
        asset_datasets=sorted(asset_datasets),
    )

    write_json(
        PACKET / "SOURCE_DELTA_R3.json",
        {
            "schema": "wnn.kaggle.startup_profile_source_delta.r3",
            "parent_source_manifest_sha256": sha256(R1_SOURCE / "SOURCE_MANIFEST_H390.json"),
            "new_source_manifest_sha256": sha256(source_manifest_path),
            "source_changes": [
                {
                    "path": "src/wnn/portability.py",
                    "parent_sha256": old_sources["portability.py"],
                    "new_sha256": sha256(SOURCE / "src/wnn/portability.py"),
                    "meaning": "opt-in byte-count/time telemetry around unchanged full-content SHA-256",
                },
                {
                    "path": "src/wnn/splits.py",
                    "parent_sha256": old_sources["splits.py"],
                    "new_sha256": sha256(SOURCE / "src/wnn/splits.py"),
                    "meaning": "opt-in load begin/source-count/end telemetry around unchanged membership validation",
                },
                {
                    "path": "scripts/train_rnn.py",
                    "parent_sha256": old_sources["train_rnn.py"],
                    "new_sha256": sha256(SOURCE / "scripts/train_rnn.py"),
                    "meaning": "profile-only actual attempted/completed optimizer update counters before the existing early return",
                },
                {
                    "path": "src/sitecustomize.py",
                    "parent_sha256": None,
                    "new_sha256": sha256(SOURCE / "src/sitecustomize.py"),
                    "meaning": "profile-only 60-second repeating Python stack heartbeat when explicitly enabled",
                },
            ],
            "full_hash_checks_preserved": True,
            "source_manifest_resealed": True,
            "single_profile_worker": PROFILE_NAME,
            "qualification": "one actual T4 only; no two-device concurrency claim",
        },
    )
    write_json(
        PACKET / "PROFILE_DIAGNOSIS_R3.json",
        {
            "schema": "wnn.kaggle.startup_profile_diagnosis.r3",
            "r1_kernel_id": "vladkochetov07/wnn-clean-t3-f0-teachers-20261011-r1-runner",
            "r1_archive_sha256": r1_manifest["archive_sha256"],
            "r1_terminal_capture": "clean_t3_f0_profile_collection_r2/raw_terminal_capture_r1/provider_output/attempt_001",
            "configured_portable_hash_bytes_per_worker": {
                row["path"]: int(row["size"])
                for row in spec["inputs"]
                if row["path"] in {"datasets/train.parquet", "datasets/valid.parquet"}
            },
            "configured_bytes_for_two_complete_r1_hashers": 2
            * sum(
                int(row["size"])
                for row in spec["inputs"]
                if row["path"] in {"datasets/train.parquet", "datasets/valid.parquet"}
            ),
            "r1_observation": {
                "profile_workers": 2,
                "worker_timeout_seconds": 240,
                "worker_log_bytes": [0, 0],
                "peak_ram_bytes": [581152768, 587087872],
                "peak_gpu_bytes": [0, 0],
                "finished_workers": 0,
                "runner_exit": "RuntimeError: Timeout for clean_t3_f0_muonall_h384_e3_s3_t4_profile1_r1",
            },
            "source_order": [
                "train_rnn.main calls load_campaign_split before configure_torch",
                "load_campaign_split calls verify_portable_source for mounted train and valid parquet",
                "_content_digest streams full SHA-256 input bytes in 1 MiB blocks",
            ],
            "interpretation": "The source order and resource observations are consistent with a pre-GPU identity-hash bottleneck. R1 produced no worker stack or hash progress logs, so the exact stalled frame is not proven.",
            "correction": {
                "single_worker": True,
                "timeout_seconds": 1800,
                "kernel_process_timeout_seconds": 6000,
                "full_content_hashing_retained": True,
                "unbuffered_output": True,
                "hash_progress_and_split_phase_events": True,
                "periodic_stack_heartbeat_seconds": 60,
                "gradient_stats": "finite-gradient/update evidence; diagnostic synchronization/compute overhead must be disclosed",
                "actual_update_counters": "profile-only optimizer attempts, AMP skips, completed updates, and completed batches",
            },
            "fit_admission": False,
            "unknowns": [
                "actual hash and startup durations",
                "actual one-T4 profile time and post-hash memory",
                "two-worker concurrent startup and I/O behavior",
                "full-fit or export throughput and cohort feasibility",
            ],
        },
    )
    write_json(
        PACKET / "PROFILE_CONTRACT_R3.json",
        {
            "schema": "wnn.kaggle.startup_profile_contract.r3",
            "dataset_id": DATASET,
            "kernel_id": f"{DATASET}-runner",
            "profile_name": PROFILE_NAME,
            "workers": 1,
            "device": "one actual Tesla T4; no concurrency qualification",
            "worker_timeout_seconds": 1800,
            "kernel_process_wall_seconds_max": 6000,
            "logical_gpu_reservation_ceiling_seconds": 12000,
            "profile_updates": {"warmup": 5, "measured": 25},
            "fit_admission": False,
            "full_identity_checks": "unchanged and required",
            "gradient_stats": "enabled for finite-gradient/update evidence; adds diagnostic device reductions and is not a throughput benchmark",
            "update_counter_output": "PROFILE_UPDATE_COUNTS reports actual optimizer attempts, AMP skips, completed updates, and completed batches",
            "checkpoint_reload_validation": "not covered by this profile; keep as a separate full-fit release/integrity gate",
            "not_qualified": [
                "two-device concurrency",
                "full-fit or export ETA",
                "cohort admission",
                "checkpoint save/reload correctness",
            ],
            "root_release_required": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    build(args.root)


if __name__ == "__main__":
    main()
