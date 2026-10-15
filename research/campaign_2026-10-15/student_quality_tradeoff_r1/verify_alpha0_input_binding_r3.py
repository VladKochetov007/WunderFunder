#!/usr/bin/env python3
"""Stream alpha-0 input bytes once, then verify stable file identity per stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import stat
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CAMPAIGN = Path(__file__).resolve().parent
REGISTRY = ROOT / "research/campaign_2026-10-04/portable_inputs.json"
E520_SPEC = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/fit_v4_observer_fallback_candidate_v1/E520_GRADER_SPEC.json"
)
H390_START = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/H390_R3_EXECUTION_START_R1.json"
)
A9_P1_ATTEMPT = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/runtime/attempts/clean-t3-f0-control-s3-a9-r2-p1-20261012.json"
)
R2_VALIDATION = CAMPAIGN / "ALPHA0_SOURCEBOUND_VALIDATION_R2.json"
STATIC_RECEIPT = CAMPAIGN / "ALPHA0_DATASET_BYTE_BINDING_R3.json"
STAGES = ("p1", "p2", "p3")
R2_CONTRACTS = {part: CAMPAIGN / f"ALPHA0_{part.upper()}_FIT_CONTRACT_R2.json" for part in STAGES}
R3_BINDINGS = {part: CAMPAIGN / f"ALPHA0_{part.upper()}_INPUT_BINDING_R3.json" for part in STAGES}

PINS = {
    "registry": "90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c",
    "e520_spec": "2ea67ab2f701d38de48c9aadae1b6b00d879f9da75fb87e1ec31e969080b853f",
    "h390_start": "09228ed1e70ba8a43a102cc58d4b43d9ed315ebfc35a5fd1b48f387be7864067",
    "a9_p1_attempt": "738f4dee1a9fbabde48b2238bb08e9c4d3f5b1ee4bbb4ce3875d213809420c58",
    "r2_validation": "4d5ae05c6a496446a84d91746a31a25bc20d042f18f681db50f7b6d28aaacb99",
}
DATA_PIN_SOURCE_PATHS = {
    "registry": "research/campaign_2026-10-04/portable_inputs.json",
    "e520_spec": "research/campaign_2026-10-06/bandit_continuation/own_baseline_E520/fit_v4_observer_fallback_candidate_v1/E520_GRADER_SPEC.json",
    "h390_start": "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/H390_R3_EXECUTION_START_R1.json",
}


def sha256_file(path: Path, progress: bool = False) -> tuple[str, int, dict[str, int]]:
    """Stream a regular file and reject any concurrent identity change."""
    if path.is_symlink():
        raise ValueError(f"refusing symlink: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    size = 0
    next_report = 5 << 30
    started = time.monotonic()
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {path}")
        while chunk := os.read(fd, 8 << 20):
            digest.update(chunk)
            previous = size
            size += len(chunk)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, previous, len(chunk), os.POSIX_FADV_DONTNEED)
            if progress and size >= next_report:
                print(
                    f"HASH_PROGRESS {path.name} bytes={size} elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
                next_report += 5 << 30
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = identity_dict(after)
    if identity != identity_dict(before) or identity != identity_dict(
        path.stat(follow_symlinks=False)
    ):
        raise RuntimeError(f"file identity changed while hashing: {path}")
    if size != before.st_size:
        raise RuntimeError(f"read {size} bytes, expected {before.st_size}: {path}")
    return digest.hexdigest(), size, identity


def identity_dict(info: os.stat_result) -> dict[str, int]:
    """Return stable identity fields for later no-rehash checks."""
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size_bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {path}")
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def current_cgroup() -> dict[str, str]:
    """Record the active support-task cap and peak memory counters."""
    line = next(
        (
            row
            for row in Path("/proc/self/cgroup").read_text().splitlines()
            if row.startswith("0::")
        ),
        None,
    )
    if line is None:
        return {"status": "cgroup-v2-path-unavailable"}
    path = Path("/sys/fs/cgroup") / line.partition("::")[2].lstrip("/")
    result = {"path": str(path)}
    for name in ("cpu.max", "memory.current", "memory.max", "memory.peak", "memory.swap.current"):
        value = path / name
        if value.exists():
            result[name] = value.read_text().strip()
    return result


def frozen_data_pins() -> tuple[dict[str, Any], dict[str, str]]:
    """Cross-check exact official data hashes in three pre-A9 source receipts."""
    source_hashes = {
        "registry": sha(REGISTRY),
        "e520_spec": sha(E520_SPEC),
        "h390_start": sha(H390_START),
        "a9_p1_attempt": sha(A9_P1_ATTEMPT),
        "r2_validation": sha(R2_VALIDATION),
    }
    if source_hashes != PINS:
        raise ValueError(f"pinned source receipt mismatch: {source_hashes}")
    registry = read_json(REGISTRY)["sources"]
    e520 = read_json(E520_SPEC)["input_identity"]
    h390 = read_json(H390_START)
    recent = h390["frozen_inputs"]["portable_inputs"]
    if h390["recorded_utc"] >= "2026-10-12T11:01:29Z":
        raise ValueError("H390 input pin does not predate A9 P1")
    result = {}
    for name in ("train", "valid"):
        entry = registry[f"datasets/{name}.parquet"]
        sha256 = entry["sha256"]
        size = int(entry["original_identity"]["bytes"])
        recent_key = f"{name}_parquet"
        if sha256 != e520[f"{name}_parquet_sha256"] or sha256 != recent[recent_key]["sha256"]:
            raise ValueError(f"historical full-file SHA pins disagree: {name}")
        if size != int(recent[recent_key]["bytes"]):
            raise ValueError(f"historical byte lengths disagree: {name}")
        result[name] = {
            "path": f"datasets/{name}.parquet",
            "sha256": sha256,
            "bytes": size,
            "footer_bytes": int(entry["original_identity"]["footer_bytes"]),
            "footer_sha256": entry["original_identity"]["footer_sha256"],
            "original_mtime_ns": int(entry["original_identity"]["mtime_ns"]),
        }
    return result, source_hashes


def a9_endpoint_specs() -> dict[str, dict[str, Any]]:
    """Resolve P1/P2/P3 reusable checkpoints from R2 and A9 provenance."""
    result = {}
    for part, epoch in (("p1", 1), ("p2", 3), ("p3", 5)):
        contract = read_json(R2_CONTRACTS[part])
        run_dir = ROOT / contract["stage"]["a9_source_run_directory"]
        name = f"model_ep{epoch}.pt"
        provenance_path = run_dir / "provenance.json"
        provenance = read_json(provenance_path)
        result[part] = {
            "run": contract["stage"]["a9_source_worker_argv"][
                contract["stage"]["a9_source_worker_argv"].index("--name") + 1
            ],
            "path": str(run_dir / name),
            "relative_path": str(Path(contract["stage"]["a9_source_run_directory"]) / name),
            "provenance_path": str(provenance_path),
            "provenance_sha256": sha(provenance_path),
            "provenance_sha256_claim": provenance["checkpoint_sha256"][name],
            "epoch": epoch,
        }
    return result


def alpha0_outputs() -> dict[str, dict[str, Any]]:
    result = {}
    for part in STAGES:
        contract = read_json(R2_CONTRACTS[part])
        stage = contract["stage"]
        result[part] = {
            "run_name": contract["run_name"],
            "directory": str(ROOT / stage["output_directory"]),
            "files": [str(ROOT / name) for name in stage["expected_outputs"]],
        }
    return result


def require_outputs_absent(parts: tuple[str, ...]) -> dict[str, Any]:
    outputs = alpha0_outputs()
    report = {}
    for part in parts:
        item = outputs[part]
        present = [path for path in [item["directory"], *item["files"]] if Path(path).exists()]
        if present:
            raise FileExistsError(f"alpha-0 {part} outputs exist: {present}")
        report[part] = {
            "run_name": item["run_name"],
            "absent_paths": [item["directory"], *item["files"]],
        }
    return report


def validate_stage_binding(
    stage: str,
    datasets: dict[str, Any],
    checkpoints: dict[str, Any],
    source_receipts_sha256: dict[str, str],
) -> dict[str, Any]:
    """Match the current input evidence to its immutable R2 and R3 pins."""
    binding = read_json(R3_BINDINGS[stage])
    base_path = R2_CONTRACTS[stage]
    base = read_json(base_path)
    if binding.get("schema") != "wnn.student-warmparent-alpha0-stage-input-binding.r3":
        raise ValueError(f"invalid R3 input binding schema: {stage}")
    if binding.get("fit_authorized") or binding.get("launch_authorized"):
        raise ValueError(f"R3 binding must not authorize a fit or launch: {stage}")
    if sha(base_path) != binding["base_r2_contract_sha256"]:
        raise ValueError(f"R2 base fit contract changed: {stage}")
    if sha(R2_VALIDATION) != binding["r2_source_validation_sha256"]:
        raise ValueError("R2 source-validation receipt changed")
    for source, expected in binding["source_receipts_sha256"].items():
        if source_receipts_sha256.get(source) != expected:
            raise ValueError(f"historical data-pin source receipt changed: {source}")
    a9_attempt_path = ROOT / binding["a9_attempt_receipt_path"]
    if sha(a9_attempt_path) != binding["a9_attempt_receipt_sha256"]:
        raise ValueError(f"{stage} A9 attempt receipt changed")
    a9_attempt = read_json(a9_attempt_path)
    train_source = a9_attempt["command"][a9_attempt["command"].index("--train-source") + 1]
    if a9_attempt.get("exit_status") != 0 or train_source != "train+valid":
        raise ValueError(f"{stage} A9 source attempt is not successful train+valid")
    for name in ("train", "valid"):
        pin = binding["datasets"][name]
        observed = datasets[name]
        pin_sources = pin["pin_source_receipts"]
        if set(pin_sources) != set(DATA_PIN_SOURCE_PATHS):
            raise ValueError(f"{stage} {name} data pin source set differs")
        for source, expected_path in DATA_PIN_SOURCE_PATHS.items():
            if (
                pin_sources[source]["path"] != expected_path
                or pin_sources[source]["sha256"] != source_receipts_sha256[source]
            ):
                raise ValueError(f"{stage} {name} source pin receipt differs: {source}")
        if (
            pin["path"] != observed["path"]
            or pin["sha256"] != observed["observed_sha256"]
            or pin["bytes"] != observed["observed_bytes"]
            or pin["original_mtime_ns"] != observed["original_mtime_ns"]
        ):
            raise ValueError(f"{stage} binding differs from full {name} byte receipt")
    checkpoint = checkpoints[stage]
    checkpoint_pin = binding["a9_reusable_checkpoint"]
    if (
        checkpoint_pin["path"] != checkpoint["relative_path"]
        or checkpoint_pin["sha256"] != checkpoint["sha256"]
        or checkpoint_pin["bytes"] != checkpoint["bytes"]
        or checkpoint_pin["provenance_sha256"] != checkpoint["provenance_sha256"]
    ):
        raise ValueError(f"{stage} binding differs from its rehashed A9 endpoint")
    if base["source_bindings"]["alpha0_command_sha256"] != binding["alpha0_worker_argv_sha256"]:
        raise ValueError(f"{stage} base contract worker argv differs")
    return binding


def establish(stage: str, receipt_path: Path) -> None:
    """Fully stream train/valid and rehash all three A9 endpoints before P1."""
    if stage != "p1":
        raise ValueError("full byte hashing is one-time before P1; P2/P3 use identity checks")
    start = time.monotonic()
    pins, source_hashes = frozen_data_pins()
    outputs = require_outputs_absent(STAGES)
    datasets = {}
    for name in ("train", "valid"):
        expected = pins[name]
        observed_hash, size, identity = sha256_file(ROOT / expected["path"], progress=True)
        if observed_hash != expected["sha256"] or size != expected["bytes"]:
            raise ValueError(f"full streamed dataset does not match frozen pin: {name}")
        if identity["mtime_ns"] != expected["original_mtime_ns"]:
            raise ValueError(f"registered original mtime differs: {name}")
        datasets[name] = {
            **expected,
            "observed_sha256": observed_hash,
            "observed_bytes": size,
            "file_identity": identity,
        }

    checkpoints = {}
    for part, spec in a9_endpoint_specs().items():
        observed_hash, size, identity = sha256_file(Path(spec["path"]))
        if observed_hash != spec["provenance_sha256_claim"]:
            raise ValueError(f"A9 {part} endpoint SHA does not match its provenance")
        checkpoints[part] = {
            **spec,
            "sha256": observed_hash,
            "bytes": size,
            "file_identity": identity,
        }

    a9_p1 = read_json(A9_P1_ATTEMPT)
    if (
        a9_p1.get("exit_status") != 0
        or a9_p1["command"][a9_p1["command"].index("--train-source") + 1] != "train+valid"
    ):
        raise ValueError("A9 P1 successful train+valid attempt binding failed")
    direct_hashes = [
        key
        for key in a9_p1["source_sha256"]
        if key.endswith("train.parquet") or key.endswith("valid.parquet")
    ]
    if direct_hashes:
        raise ValueError("A9 P1 direct data-hash inventory changed; review before continuing")

    validate_stage_binding("p1", datasets, checkpoints, source_hashes)

    receipt = {
        "schema": "wnn.student-warmparent-alpha0-dataset-byte-binding.r3",
        "status": "PASS_FULL_DATA_AND_A9_CHECKPOINT_SHA; NO_FIT_NO_LAUNCH",
        "fit_authorized": False,
        "launch_authorized": False,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_receipts_sha256": source_hashes,
        "datasets": datasets,
        "a9_reusable_checkpoints": checkpoints,
        "a9_p1_attempt": {
            "path": str(A9_P1_ATTEMPT),
            "sha256": source_hashes["a9_p1_attempt"],
            "started_utc": a9_p1["started_utc"],
            "direct_raw_parquet_sha256_present": False,
            "train_source": "train+valid",
        },
        "alpha0_output_absence": outputs,
        "identity_reuse": {
            "dataset_full_stream_passes": 1,
            "a9_checkpoint_full_hash_passes": 1,
            "future_stage_identity_fields": [
                "device",
                "inode",
                "size_bytes",
                "mtime_ns",
                "ctime_ns",
            ],
            "any_identity_change_requires_full_rehash": True,
            "no_34gb_rehash_per_stage": True,
        },
        "historical_binding_limit": (
            "A9 P1's successful attempt/source receipt does not contain raw train/valid SHA-256 or an "
            "attempt-time inode/ctime record. The pre-A9 official registry and root receipts agree on "
            "these complete-file pins; this run streamed current bytes against them and binds inode/size/mtime/ctime."
        ),
        "runtime": {
            "wall_seconds": time.monotonic() - start,
            "process_cpu_seconds": time.process_time(),
            "process_ru_maxrss_raw_platform_units": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "process_ru_maxrss_used_for_cap_assessment": False,
            "cap_assessment_source": "cgroup v2 memory.max/current/peak",
            "python_version": platform.python_version(),
            "task_cpu_quota_env": os.environ.get("WNN_CPU_QUOTA"),
            "task_memory_max_env_bytes": os.environ.get("WNN_MEM_MAX"),
            "task_runtime_max_env": os.environ.get("WNN_RUNTIME_MAX"),
            "cgroup": current_cgroup(),
        },
        "no_fit_no_launch": True,
    }
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": str(receipt_path),
                "datasets": {k: v["observed_sha256"] for k, v in datasets.items()},
                "checkpoints": {k: v["sha256"] for k, v in checkpoints.items()},
                "elapsed_seconds": receipt["runtime"]["wall_seconds"],
                "no_fit_no_launch": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def verify_stage(stage: str, receipt_path: Path, receipt_out: Path) -> None:
    """Verify saved byte hashes through unchanged file identity and stage outputs."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage: {stage}")
    receipt = read_json(receipt_path)
    if (
        receipt.get("schema") != "wnn.student-warmparent-alpha0-dataset-byte-binding.r3"
        or receipt.get("status") != "PASS_FULL_DATA_AND_A9_CHECKPOINT_SHA; NO_FIT_NO_LAUNCH"
        or receipt.get("fit_authorized")
        or receipt.get("launch_authorized")
    ):
        raise ValueError("not a successful full dataset/checkpoint receipt")
    source_hashes = receipt["source_receipts_sha256"]
    if sha(A9_P1_ATTEMPT) != source_hashes["a9_p1_attempt"]:
        raise ValueError("A9 P1 input-source attempt receipt changed")
    current_pins, current_source_hashes = frozen_data_pins()
    if current_source_hashes != source_hashes:
        raise ValueError("historical whole-file data pin receipts changed")
    for name in ("train", "valid"):
        receipt_pin = receipt["datasets"][name]
        expected_pin = current_pins[name]
        if (
            receipt_pin["path"] != expected_pin["path"]
            or receipt_pin["observed_sha256"] != expected_pin["sha256"]
            or receipt_pin["observed_bytes"] != expected_pin["bytes"]
            or receipt_pin["footer_sha256"] != expected_pin["footer_sha256"]
        ):
            raise ValueError(f"{name} full-byte receipt differs from current historical pins")
    for name in ("train", "valid"):
        item = receipt["datasets"][name]
        current = identity_dict((ROOT / item["path"]).stat(follow_symlinks=False))
        if current != item["file_identity"]:
            raise RuntimeError(f"{name} file identity changed; a fresh full stream is required")
    for item in receipt["a9_reusable_checkpoints"].values():
        current = identity_dict(Path(item["path"]).stat(follow_symlinks=False))
        if current != item["file_identity"]:
            raise RuntimeError(
                "A9 checkpoint identity changed; a fresh full checkpoint hash is required"
            )
    binding = validate_stage_binding(
        stage,
        receipt["datasets"],
        receipt["a9_reusable_checkpoints"],
        current_source_hashes,
    )
    for item in receipt["a9_reusable_checkpoints"].values():
        if sha(Path(item["provenance_path"])) != item["provenance_sha256"]:
            raise RuntimeError("A9 checkpoint provenance changed; revalidate before continuing")
    checkpoint = receipt["a9_reusable_checkpoints"][stage]
    outputs = require_outputs_absent((stage,))
    result = {
        "schema": "wnn.student-warmparent-alpha0-stage-input-identity-check.r3",
        "status": "PASS_STAT_IDENTITY_AND_STAGE_OUTPUT_ABSENCE; NO_FIT_NO_LAUNCH",
        "stage": stage,
        "created_utc": datetime.now(UTC).isoformat(),
        "full_hash_receipt_path": str(receipt_path),
        "full_hash_receipt_sha256": sha(receipt_path),
        "base_r2_contract_sha256": binding["base_r2_contract_sha256"],
        "dataset_identity_matches": True,
        "a9_checkpoint_identity_matches": True,
        "a9_checkpoint_sha256": checkpoint["sha256"],
        "stage_outputs_absent": outputs[stage],
        "no_full_rehash_repeated": True,
        "no_fit_no_launch": True,
    }
    write_json(receipt_out, result)
    print(json.dumps(result, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("establish", "verify-stage"), required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "establish":
        if args.receipt is not None:
            parser.error("--receipt is only accepted by verify-stage")
        establish(args.stage, args.receipt_out)
    else:
        if args.receipt is None:
            parser.error("verify-stage requires --receipt")
        verify_stage(args.stage, args.receipt, args.receipt_out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"INPUT_BINDING_FAILED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
