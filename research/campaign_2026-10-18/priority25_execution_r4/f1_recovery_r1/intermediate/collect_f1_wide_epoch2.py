#!/usr/bin/env python3
"""Collect the exact F1 wide continuation epoch-2 generation without loading Torch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

PACKET = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
V3_COMMON = PACKET / "checkpoints/f1_checkpoint_common.py"
V3_COMMON_SHA256 = "6531e841cedaf9c5cd960dbe44c29a11fb89b65e15a781c07cea9c3fd2b5284e"
STAGE1_RECEIPT = Path("runtime/receipts/RANK01_WIDE_F1_S1001_STAGE1_VERIFICATION.json")
STAGE1_RECEIPT_SHA256 = "b9cf2eb40fe3ba8018c2c98c015d5069bc5eca422a2c384de743f0426866bc81"
STAGE1_STATE_SHA256 = "8cb3dde79550355303cbede0d0953864046b7594c16af648423d4a3c9af92255"
ARTIFACT_NAMES = ("model_ep2.pt", "provenance.json", "training_state.pt")
EXPECTED_PROVENANCE_COUNTS = {
    "optimizer_attempts": 44_790,
    "amp_skipped_updates": 0,
    "optimizer_completed_updates": 44_790,
    "ema_updates": 0,
    "completed_batches": 4_479,
}
ONE_EPOCH_COUNTS = {
    "optimizer_attempts": 14_930,
    "amp_skipped_updates": 0,
    "optimizer_completed_updates": 14_930,
    "ema_updates": 0,
    "completed_batches": 1_493,
}


class CollectionMismatch(ValueError):
    """A producer, parent, plan or file generation differs from the F1 boundary."""


def require(condition: bool, label: str) -> None:
    if not condition:
        raise CollectionMismatch(f"{label} differs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path.name} JSON object")
    return cast(dict[str, object], value)


def read_json_with_sha(path: Path) -> tuple[dict[str, object], str]:
    """Parse JSON and hash the same captured bytes, even if its producer rewrites later."""
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    require(isinstance(value, dict), f"{path.name} JSON object")
    return cast(dict[str, object], value), hashlib.sha256(raw).hexdigest()


def stats(path: Path) -> dict[str, int]:
    value = path.stat(follow_symlinks=False)
    return {"inode": value.st_ino, "size": value.st_size, "mtime_ns": value.st_mtime_ns}


def load_v3_common() -> object:
    """Import the hash-pinned V3 retry binder without copying or editing it."""
    require(sha256_file(V3_COMMON) == V3_COMMON_SHA256, "V3 common helper SHA256")
    spec = importlib.util.spec_from_file_location("f1_recovery_v3_common", V3_COMMON)
    require(spec is not None and spec.loader is not None, "V3 common import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_attempt(
    record: Mapping[str, object], spec: Mapping[str, object], plan_sha: str
) -> dict[str, object]:
    attempt_id = str(spec["attempt"])
    require(record.get("id") == attempt_id, "attempt ID")
    require(record.get("unit") == f"wnn-bandit-{attempt_id}.service", "attempt unit")
    require(record.get("command") == spec["command"], "frozen trainer command")
    require(
        str(record.get("working_directory", "")).endswith("/source-a266b036"),
        "attempt source root",
    )
    require(
        (record.get("contract"), record.get("contract_sha256"))
        == (
            str(
                REMOTE_ROOT
                / "research/campaign_2026-10-18/priority25_execution_r4/f1_beleriand_fallback/F1_BELERIAND_PLAN.json"
            ),
            plan_sha,
        ),
        "base-plan contract identity",
    )
    status = record.get("status")
    if status == "running":
        require(record.get("exit_status") is None, "running attempt exit status")
    else:
        require(status == "finished", "attempt status")
        require(
            type(record.get("exit_status")) is int and record["exit_status"] == 0,
            "finished attempt exit status",
        )
    command = cast(list[str], spec["command"])
    require("--resume-state" in command, "continuation resume flag")
    resume_path = command[command.index("--resume-state") + 1]
    require(
        resume_path
        == str(REMOTE_ROOT / "data/artifacts/runs/rr18_r01_wide_f1_s1001/training_state.pt"),
        "frozen continuation parent path",
    )
    return {
        "id": attempt_id,
        "status": status,
        "exit_status": record.get("exit_status"),
        "command": command,
    }


def validate_stage1_receipt(
    receipt: Mapping[str, object], state_sha256: str, common: object
) -> None:
    require(
        (
            receipt.get("schema"),
            receipt.get("status"),
            receipt.get("phase"),
            receipt.get("arm"),
            receipt.get("run"),
            receipt.get("next_epoch"),
            receipt.get("plan_sha256"),
            receipt.get("fold1_manifest_sha256"),
            receipt.get("portable_registry_sha256"),
            receipt.get("source_archive_manifest_sha256"),
            receipt.get("source_revision"),
            receipt.get("completed_update_counts"),
            receipt.get("planned_schedule_updates"),
            receipt.get("planned_schedule_batches"),
        )
        == (
            "priority25.f1_checkpoint_verification.v1",
            "passed",
            "stage1",
            "wide",
            "rr18_r01_wide_f1_s1001",
            2,
            common.PLAN_SHA256,
            common.FOLD1_SHA256,
            common.REGISTRY_SHA256,
            common.ARCHIVE_MANIFEST_SHA256,
            common.SOURCE_REVISION,
            {
                "optimizer_attempts": 29_860,
                "amp_skipped_updates": 0,
                "optimizer_completed_updates": 29_860,
                "ema_updates": 0,
                "completed_batches": 2_986,
            },
            89_580,
            8_958,
        ),
        "verified stage1 receipt identity",
    )
    common.validate_v2_stage1_helper_identity(receipt)
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, dict), "stage1 receipt artifacts")
    state = artifacts.get("training_state.pt")
    require(
        isinstance(state, dict) and state.get("sha256") == state_sha256 == STAGE1_STATE_SHA256,
        "stage1 training-state SHA",
    )
    attempt = receipt.get("attempt")
    require(isinstance(attempt, dict), "stage1 attempt receipt")
    spec = common.arm_spec(common.read_json(common.ORIGINAL_PLAN_PATH), "wide", "stage1")
    require(
        (
            attempt.get("id"),
            attempt.get("command"),
            attempt.get("status"),
            attempt.get("exit_status"),
        )
        == (spec["attempt"], spec["command"], "finished", 0),
        "stage1 attempt binding",
    )


def validate_provenance(
    provenance: Mapping[str, object],
    model_sha256: str,
    stage1_receipt: Mapping[str, object],
    common: object,
) -> dict[str, object]:
    require(
        provenance.get("cumulative_updates") == EXPECTED_PROVENANCE_COUNTS,
        "epoch-2 cumulative optimizer counters",
    )
    require(provenance.get("split_manifest_sha256") == common.FOLD1_SHA256, "fold1 provenance SHA")
    require(
        provenance.get("fit_membership_identity") == stage1_receipt.get("fit_membership_identity"),
        "fit membership identity",
    )
    expected_parent = {
        "path": str(REMOTE_ROOT / "data/artifacts/runs/rr18_r01_wide_f1_s1001/training_state.pt"),
        "sha256": STAGE1_STATE_SHA256,
        "next_epoch": 2,
    }
    require(provenance.get("resumed_from") == expected_parent, "stage1 resume parent")
    checkpoint_hashes = provenance.get("checkpoint_sha256")
    require(
        isinstance(checkpoint_hashes, dict)
        and checkpoint_hashes.get("model_ep2.pt") == model_sha256,
        "model_ep2 producer hash",
    )
    diagnostics = provenance.get("update_diagnostics")
    require(
        isinstance(diagnostics, list) and len(diagnostics) == 3, "three completed epoch diagnostics"
    )
    for epoch, row in enumerate(diagnostics):
        require(isinstance(row, dict), f"epoch {epoch} diagnostics mapping")
        require(row.get("epoch_index") == epoch, f"epoch {epoch} diagnostic index")
        require(row.get("epoch") == ONE_EPOCH_COUNTS, f"epoch {epoch} update counts")
        expected_cumulative = {**ONE_EPOCH_COUNTS}
        for key in ("optimizer_attempts", "optimizer_completed_updates", "completed_batches"):
            expected_cumulative[key] *= epoch + 1
        require(row.get("cumulative") == expected_cumulative, f"epoch {epoch} cumulative counts")
        grad = row.get("grad_stats")
        require(
            isinstance(grad, dict) and grad.get("nonfinite_steps") == 0,
            f"epoch {epoch} nonfinite gradient count",
        )
    return expected_parent


def _copy_one(source: Path, target: Path) -> None:
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    shutil.copystat(source, target, follow_symlinks=False)


def copy_generation(
    source: Path,
    destination: Path,
    *,
    expected_hashes: Mapping[str, str] | None = None,
    expected_stats: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, object]]]:
    """Create-only archive of three stable files, removing only our partial on failure."""
    paths = {name: source / name for name in ARTIFACT_NAMES}
    require(
        all(path.is_file() and not path.is_symlink() for path in paths.values()),
        "producer file types",
    )
    before = {name: stats(path) for name, path in paths.items()}
    if expected_stats is not None:
        require(before == expected_stats, "producer generation changed before copy")
    require(all(row["size"] > 0 for row in before.values()), "non-empty producer artifacts")
    require(sum(row["size"] for row in before.values()) <= 100_000_000, "100 MB generation bound")
    before_hash = {name: sha256_file(path) for name, path in paths.items()}
    if expected_hashes is not None:
        require(before_hash == expected_hashes, "producer bytes changed after validation")
    destination.mkdir(parents=True, exist_ok=False)
    try:
        artifacts: dict[str, dict[str, object]] = {}
        for name, path in paths.items():
            target = destination / name
            _copy_one(path, target)
            copied_hash = sha256_file(target)
            require(copied_hash == before_hash[name], f"source/copy hash {name}")
            artifacts[name] = {
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": copied_hash,
            }
        after = {name: stats(path) for name, path in paths.items()}
        require(before == after, "producer file generation changed during copy")
        require(
            {name: sha256_file(path) for name, path in paths.items()} == before_hash,
            "producer bytes changed during copy",
        )
        return after, artifacts
    except Exception:
        shutil.rmtree(destination)
        raise


def collect(
    base: Path,
    attempt_path: Path,
    stage1_receipt_path: Path,
    output_path: Path,
    resource_envelope: Mapping[str, object],
) -> dict[str, object]:
    common = load_v3_common()
    binding = common.validate_recovery_binding()
    plan = common.validate_plan(common.ORIGINAL_PLAN_PATH)
    spec = common.arm_spec(plan, "wide", "terminal")
    require(spec["attempt"] == binding["recovery_attempt_id"], "recovery attempt binding")
    require(resource_envelope.get("cpu_quota") == "25%", "collector CPU envelope")
    require(
        type(resource_envelope.get("memory_max_bytes")) is int
        and 0 < resource_envelope["memory_max_bytes"] <= 512_000_000,
        "collector memory envelope",
    )
    require(resource_envelope.get("memory_swap_max_bytes") == 0, "collector swap envelope")
    require(resource_envelope.get("runtime_seconds") == 120, "collector runtime envelope")
    current = resource_envelope.get("parent_memory_current_bytes")
    maximum = resource_envelope.get("parent_memory_max_bytes")
    require(
        type(current) is int
        and type(maximum) is int
        and current + 8_000_000_000 + 512_000_000 <= min(maximum, 17_000_000_000),
        "parent memory envelope",
    )
    require(base.resolve() == REMOTE_ROOT, "campaign root")
    require(not output_path.exists(), "receipt already exists")
    source = base / "data/artifacts/runs" / str(spec["run"])
    destination = base / "checkpoint_snapshots" / str(spec["run"]) / "epoch2"
    require(not destination.exists(), "epoch-2 snapshot already exists")
    stage1_snapshot = base / "checkpoint_snapshots" / str(spec["stage1_run"]) / "stage1_epoch1"
    stage1_state = stage1_snapshot / "training_state.pt"
    stage1_state_sha256 = sha256_file(stage1_state)
    require(stage1_state_sha256 == STAGE1_STATE_SHA256, "stage1 snapshot state SHA")
    stage1_receipt, stage1_receipt_sha256 = read_json_with_sha(stage1_receipt_path)
    require(stage1_receipt_sha256 == STAGE1_RECEIPT_SHA256, "stage1 receipt SHA")
    validate_stage1_receipt(stage1_receipt, sha256_file(stage1_state), common)
    plan_sha = common.sha256_file(common.ORIGINAL_PLAN_PATH)
    record, attempt_sha = read_json_with_sha(attempt_path)
    attempt = validate_attempt(record, spec, plan_sha)
    before = {name: stats(source / name) for name in ARTIFACT_NAMES}
    require(sum(item["size"] for item in before.values()) <= 100_000_000, "source generation size")
    validated_source_hashes = {name: sha256_file(source / name) for name in ARTIFACT_NAMES}
    provenance, provenance_sha256 = read_json_with_sha(source / "provenance.json")
    require(
        provenance_sha256 == validated_source_hashes["provenance.json"],
        "provenance changed during validation",
    )
    require(
        provenance.get("cumulative_updates") == EXPECTED_PROVENANCE_COUNTS,
        "epoch-2 cumulative optimizer counters",
    )
    model_hash = validated_source_hashes["model_ep2.pt"]
    parent = validate_provenance(provenance, model_hash, stage1_receipt, common)
    after, artifacts = copy_generation(
        source, destination, expected_hashes=validated_source_hashes, expected_stats=before
    )
    output_created = False
    try:
        require(after == before, "producer generation changed across validation and copy")
        require(
            artifacts["provenance.json"]["sha256"] == sha256_file(source / "provenance.json"),
            "copied provenance SHA",
        )
        receipt: dict[str, object] = {
            "schema": "priority25.f1_wide_continuation_epoch2_collection.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "status": "collected",
            "arm": "wide",
            "phase": "continuation_epoch2",
            "run": spec["run"],
            "epoch_index": 2,
            "attempt": attempt,
            "attempt_record_observed": record,
            "attempt_record_path": str(attempt_path),
            "attempt_record_sha256": attempt_sha,
            "source_directory": str(source),
            "snapshot_directory": str(destination),
            "source_stats_before": before,
            "source_stats_after": after,
            "validated_source_sha256": validated_source_hashes,
            "artifacts": artifacts,
            "source_snapshot_hash_match": True,
            "producer_model_hash_matches": True,
            "plan_sha256": common.PLAN_SHA256,
            "recovery_plan_sha256": common.RECOVERY_PLAN_SHA256,
            "rejected_attempt_sha256": common.REJECTED_ATTEMPT_SHA256,
            "recovery_binding": binding,
            "v3_common_helper_sha256": V3_COMMON_SHA256,
            "stage1_verification_receipt_sha256": STAGE1_RECEIPT_SHA256,
            "stage1_verification_receipt_path": str(stage1_receipt_path),
            "stage1_training_state_sha256": STAGE1_STATE_SHA256,
            "resumed_from": parent,
            "completed_update_counts": EXPECTED_PROVENANCE_COUNTS,
            "planned_schedule_updates": common.SCHEDULE_UPDATES,
            "planned_schedule_batches": common.SCHEDULE_BATCHES,
            "zero_skipped_and_nonfinite_updates": True,
            "resource_envelope": dict(resource_envelope),
            "collector_sha256": sha256_file(Path(__file__)),
            "scientific_status": "intermediate checkpoint collection only; tensor inspection, resume parity and quality grade pending",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x") as stream:
            output_created = True
            stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    except Exception:
        if output_created:
            output_path.unlink(missing_ok=True)
        shutil.rmtree(destination)
        raise
    return receipt


def live_resource_envelope() -> dict[str, object]:
    """Fail closed unless invoked in the reviewed 25%/512 MB support unit."""
    cgroup_line = next(
        line
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )
    service_path = cgroup_line.split("::", maxsplit=1)[1]
    service = Path("/sys/fs/cgroup") / service_path.lstrip("/")
    quota, period = (int(value) for value in (service / "cpu.max").read_text().split())
    memory_max = int((service / "memory.max").read_text())
    swap_max = int((service / "memory.swap.max").read_text())
    require(quota > 0 and period > 0 and quota / period <= 0.25, "collector CPU cgroup")
    require(0 < memory_max <= 512_000_000 and swap_max == 0, "collector memory/swap cgroup")
    service_name = service_path.rstrip("/").split("/")[-1]
    require(service_name.endswith(".service"), "collector systemd service")
    runtime = subprocess.check_output(
        ["systemctl", "--user", "show", service_name, "--property=RuntimeMaxUSec", "--value"],
        text=True,
    ).strip()
    require(runtime == "2min", "collector RuntimeMaxUSec")
    parent_line = subprocess.check_output(
        ["systemctl", "--user", "show", "wnn-campaign.slice", "-p", "ControlGroup", "--value"],
        text=True,
    ).strip()
    parent = Path("/sys/fs/cgroup") / parent_line.lstrip("/")
    current, maximum, swap = (
        int((parent / name).read_text())
        for name in ("memory.current", "memory.max", "memory.swap.max")
    )
    require(
        service_path.startswith(parent_line.rstrip("/") + "/"), "collector campaign slice ancestry"
    )
    require(swap == 0, "campaign slice swap")
    require(
        current + 8_000_000_000 + 512_000_000 <= min(maximum, 17_000_000_000),
        "parent headroom for worker and collector",
    )
    return {
        "service": service_name,
        "parent_memory_current_bytes": current,
        "parent_memory_max_bytes": maximum,
        "collector_cpu_max": (service / "cpu.max").read_text().strip(),
        "cpu_quota": "25%",
        "memory_max_bytes": memory_max,
        "memory_swap_max_bytes": swap_max,
        "runtime_seconds": 120,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, default=REMOTE_ROOT)
    args = parser.parse_args()
    base = args.campaign_root
    common = load_v3_common()
    spec = common.arm_spec(common.validate_plan(common.ORIGINAL_PLAN_PATH), "wide", "terminal")
    attempt_path = base / "runtime/attempts" / f"{spec['attempt']}.json"
    stage1_receipt = base / STAGE1_RECEIPT
    output = base / "runtime/receipts/RANK01_WIDE_F1_S1001_EP2_COLLECTION.json"
    receipt = collect(base, attempt_path, stage1_receipt, output, live_resource_envelope())
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
