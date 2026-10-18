from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import terminal_gate as gate

B = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
P = B / "research/campaign_2026-10-18/priority25_execution_r4"
PACKET = P / "f3_terminal_execution_r1"
SUPPORT = P / "f3_checkpoint_support_r1"
PLAN = P / "f3_beleriand_preparation_r1/F3_BELERIAND_PLAN.json"
SUPPORT_MANIFEST_SHA = "7e4fce0d7232a487a769e12d611050bc608f9cc7fcea300bc2063525abfa38dc"
SUPPORT_REVIEW_SHA = "de709897397d7cd775d2aba84fb71f8f7d91504802814f5bcfc26e48e203763b"
FOLD = B / "splits/fold3.json"
REGISTRY = B / "support-source/terminal-v1/PORTABLE_SOURCES.json"
ARCHIVES = B / "support-source/terminal-v1/PORTABLE_ARCHIVES.json"
PYTHON = "/home/grinchenkoeg/WunderFunder/.venv/bin/python"
ATTEMPTS = B / "runtime/attempts"
SEQUENCE = B / "runtime/receipts/RANK01_F3_S1001_SEQUENCE.json"
STAGE1_PATHS = {
    arm: {
        "snapshot": B / f"checkpoint_snapshots/rr18_r01_{arm}_f3_s1001/stage1_epoch1",
        "collection": B / f"receipts/RANK01_{arm.upper()}_F3_S1001_STAGE1_COLLECTION.json",
        "verification": B / f"receipts/RANK01_{arm.upper()}_F3_S1001_STAGE1_VERIFICATION.json",
    }
    for arm in ("wide", "deep")
}
REMOTE_RECORDS = B / "receipts/f3_terminal_execution_r1"
CONTROLLER_UNIT = gate.CONTROLLER
SYSTEMD_FIELDS = (
    "Id,ActiveState,SubState,Result,ExecMainStatus,ExecMainCode,"
    "ExecMainStartTimestampMonotonic,ExecMainExitTimestampMonotonic,"
    "CPUUsageNSec,MemoryPeak,MemoryCurrent,ControlGroup,CPUQuotaPerSecUSec,"
    "MemoryMax,MemorySwapMax,RuntimeMaxUSec"
)


def digest(path: Path) -> str:
    """Hash a regular file without following a final symlink."""
    if path.is_symlink() or not path.is_file():
        raise gate.GateError(f"not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command: list[str], *, timeout: int = 15) -> subprocess.CompletedProcess[str]:
    """Run one bounded system command and retain its raw text output."""
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)


def write_json(path: Path, value: object) -> bytes:
    """Persist one create-only JSON record and return its exact bytes."""
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw


def operation_record_dir(invocation: str, arm: str, operation: str) -> Path:
    """Resolve one existing support record directory without following links."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,63}", invocation):
        raise gate.GateError("invocation is not a safe lowercase slug")
    if arm not in {"wide", "deep"} or operation not in {"collector", "verifier"}:
        raise gate.GateError("unknown support record identity")
    invocation_dir = REMOTE_RECORDS / invocation
    record = invocation_dir / f"{arm}_{operation}"
    if (
        REMOTE_RECORDS.is_symlink()
        or not REMOTE_RECORDS.is_dir()
        or invocation_dir.is_symlink()
        or not invocation_dir.is_dir()
        or record.is_symlink()
        or not record.is_dir()
        or invocation_dir.resolve().parent != REMOTE_RECORDS.resolve()
        or record.resolve().parent != invocation_dir.resolve()
    ):
        raise gate.GateError("support record path is missing or redirected")
    return record


def read_existing_capture(path: Path, unit: str) -> dict[str, object] | None:
    """Read an immutable capture for transport recovery without rewriting it."""
    if not os.path.lexists(path):
        return None
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 10_000_000:
        raise gate.GateError("existing capture evidence is unsafe or exceeds its size bound")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise gate.GateError("existing capture evidence is unreadable") from exc
    if not isinstance(value, dict) or (
        value.get("schema") != "priority25.f3_terminal_support_capture.v1"
        or value.get("status") != "TERMINAL_CAPTURE"
        or value.get("unit") != unit
    ):
        raise gate.GateError("existing capture evidence identity differs")
    return value


def verify_packet(args: argparse.Namespace) -> dict[str, object]:
    """Verify the staged wrapper, its review, and every frozen helper pin."""
    manifest_path = PACKET / "PREPARATION_MANIFEST.json"
    review_path = PACKET / "INDEPENDENT_REVIEW_V1.json"
    if digest(manifest_path) != args.manifest_sha256:
        raise gate.GateError("terminal packet manifest SHA differs")
    if digest(PACKET / "terminal_remote_v1.py") != args.remote_sha256:
        raise gate.GateError("remote wrapper SHA differs")
    if digest(review_path) != args.review_sha256:
        raise gate.GateError("terminal wrapper review SHA differs")
    manifest = json.loads(manifest_path.read_bytes())
    review = json.loads(review_path.read_bytes())
    if review.get("status") != "passed" or not str(review.get("decision", "")).startswith("PASS"):
        raise gate.GateError("terminal wrapper lacks independent PASS review")
    if review.get("manifest_sha256") != args.manifest_sha256:
        raise gate.GateError("review binds a different terminal manifest")
    if review.get("blockers") not in ([], None):
        raise gate.GateError("terminal wrapper review has blockers")
    runtime = manifest.get("runtime_files")
    if not isinstance(runtime, dict) or set(runtime) != {
        "terminal_gate.py",
        "terminal_remote_v1.py",
    }:
        raise gate.GateError("terminal runtime allowlist differs")
    allowed_root = set(runtime) | {
        "README.md",
        "PREPARATION_MANIFEST.json",
        "INDEPENDENT_REVIEW_V1.json",
    }
    if manifest.get("remote_stage_inventory") != sorted(allowed_root):
        raise gate.GateError("manifest remote-stage inventory differs")
    names = {path.name for path in PACKET.iterdir()}
    if names != allowed_root:
        raise gate.GateError(f"staged packet inventory differs: {sorted(names ^ allowed_root)}")
    if any(
        path.name.endswith((".pyc", ".pyo")) or path.name == "__pycache__"
        for path in PACKET.iterdir()
    ):
        raise gate.GateError("staged packet contains Python bytecode")
    for name, pin in runtime.items():
        if digest(PACKET / name) != pin["sha256"] or (PACKET / name).stat().st_size != pin["bytes"]:
            raise gate.GateError(f"runtime source hash differs: {name}")
    readme_pin = manifest["artifacts"]["README.md"]
    if (
        digest(PACKET / "README.md") != readme_pin["sha256"]
        or (PACKET / "README.md").stat().st_size != readme_pin["bytes"]
    ):
        raise gate.GateError("staged README hash differs")
    if digest(SUPPORT / "PREPARATION_MANIFEST.json") != SUPPORT_MANIFEST_SHA:
        raise gate.GateError("checkpoint-support manifest differs")
    if digest(SUPPORT / "INDEPENDENT_REVIEW.json") != SUPPORT_REVIEW_SHA:
        raise gate.GateError("checkpoint-support review differs")
    support_manifest = json.loads((SUPPORT / "PREPARATION_MANIFEST.json").read_bytes())
    for name, item in support_manifest["artifacts"].items():
        path = SUPPORT / name
        if not path.is_file() or digest(path) != item["sha256"]:
            raise gate.GateError(f"checkpoint-support pin differs: {name}")
    expected_remote = {
        "plan": {"path": str(PLAN), "sha256": gate.PLAN_SHA256},
        "fold3": {"path": str(FOLD), "sha256": gate.FOLD_SHA256},
        "portable_registry": {
            "path": str(REGISTRY),
            "sha256": "1f39cbf8b242495e18baaf5d36eeac907c0ad655e951b0efa6a48580b7699378",
        },
        "portable_archives": {
            "path": str(ARCHIVES),
            "sha256": "9a90e46958eaed2784f1f12e6a501292e4c865b1bc6d07e3f2bd233271da7341",
        },
    }
    if manifest.get("expected_remote_bindings") != expected_remote:
        raise gate.GateError("manifest expected remote source bindings differ")
    for path, expected in (
        (FOLD, expected_remote["fold3"]["sha256"]),
        (REGISTRY, expected_remote["portable_registry"]["sha256"]),
        (ARCHIVES, expected_remote["portable_archives"]["sha256"]),
        (PLAN, expected_remote["plan"]["sha256"]),
    ):
        if digest(path) != expected:
            raise gate.GateError(f"frozen input SHA differs: {path}")
    return manifest


def support_common() -> Any:
    """Import the already pinned stage-independent F3 evidence helper."""
    sys.path.insert(0, str(SUPPORT))
    import f3_checkpoint_common

    if (
        Path(f3_checkpoint_common.__file__).resolve()
        != (SUPPORT / "f3_checkpoint_common.py").resolve()
    ):
        raise gate.GateError("checkpoint common import resolved outside pinned support")
    return f3_checkpoint_common


def load_stage1_pair() -> dict[str, object]:
    """Require both already reviewed stage1 collections and verifications."""
    bound: dict[str, object] = {}
    for arm, paths in STAGE1_PATHS.items():
        for key in ("collection", "verification"):
            path = paths[key]
            expected = gate.STAGE1[arm][key]
            if digest(path) != expected:
                raise gate.GateError(f"{arm} stage1 {key} hash differs")
            record = json.loads(path.read_bytes())
            required = {"arm": arm, "phase": "stage1", "plan_sha256": gate.PLAN_SHA256}
            if any(record.get(field) != value for field, value in required.items()):
                raise gate.GateError(f"{arm} stage1 {key} identity differs")
            if key == "verification" and (
                record.get("status") != "passed"
                or record.get("fold3_manifest_sha256") != gate.FOLD_SHA256
                or record.get("next_epoch") != 2
            ):
                raise gate.GateError(f"{arm} stage1 verification is not accepted")
            bound[f"{arm}_{key}"] = {"path": str(path), "sha256": expected}
        if not paths["snapshot"].is_dir() or paths["snapshot"].is_symlink():
            raise gate.GateError(f"{arm} stage1 snapshot missing or symlinked")
    return bound


def sequence_and_attempt(
    arm: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object] | None, str]:
    """Validate exact finished native attempt and live seven-step pointer."""
    common = support_common()
    plan = common.validate_plan(PLAN)
    spec = common.arm_spec(plan, arm, "terminal")
    attempt_path = ATTEMPTS / f"{spec['attempt']}.json"
    attempt_evidence = common.validate_attempt(attempt_path, plan, spec)
    gate.validate_successful_exit(attempt_evidence, str(spec["attempt"]))
    sequence = json.loads(SEQUENCE.read_bytes())
    current = gate.validate_pointer(plan, sequence, arm)
    return plan, attempt_evidence, current, digest(attempt_path)


def collect_paths(arm: str) -> dict[str, Path]:
    """Resolve exact plan-declared terminal collection paths."""
    return {
        "snapshot": B / f"checkpoint_snapshots/rr18_r01_{arm}_f3_s1001_e6/epoch5_terminal",
        "receipt": B / f"receipts/RANK01_{arm.upper()}_F3_S1001_TERMINAL_COLLECTION.json",
    }


def verify_collection_receipt(path: Path, arm: str) -> None:
    """Require a complete terminal collection receipt before structural verify."""
    record = json.loads(path.read_bytes())
    evidence = record.get("attempt_evidence", {})
    if (
        record.get("schema") != "priority25.f3_checkpoint_collection.v1"
        or record.get("arm") != arm
        or record.get("phase") != "terminal"
        or record.get("plan_sha256") != gate.PLAN_SHA256
        or evidence.get("status") != "finished"
        or evidence.get("exit_status") != 0
        or record.get("source_snapshot_hash_match") is not True
    ):
        raise gate.GateError("terminal collection receipt is incomplete or mismatched")


def output_paths(arm: str, operation: str) -> tuple[Path, Path]:
    """Return helper result and terminal archive paths for collision checks."""
    result = (
        B
        / f"receipts/RANK01_{arm.upper()}_F3_S1001_TERMINAL_{'COLLECTION' if operation == 'collector' else 'VERIFICATION'}.json"
    )
    snapshot = B / f"checkpoint_snapshots/rr18_r01_{arm}_f3_s1001_e6/epoch5_terminal"
    return result, snapshot


def validate_output_state(arm: str, operation: str) -> tuple[Path, Path]:
    """Check canonical helper outputs and terminal inputs with lstat semantics."""
    result_path, snapshot_path = output_paths(arm, operation)
    collection_path = collect_paths(arm)["receipt"]
    collection_snapshot = collect_paths(arm)["snapshot"]
    if operation == "collector":
        if os.path.lexists(result_path) or os.path.lexists(snapshot_path):
            raise gate.GateError("terminal collection output or snapshot already exists")
    else:
        if (
            collection_path.is_symlink()
            or not collection_path.is_file()
            or collection_snapshot.is_symlink()
            or not collection_snapshot.is_dir()
        ):
            raise gate.GateError("terminal collection inputs are missing or symlinked")
        verify_collection_receipt(collection_path, arm)
        if os.path.lexists(result_path):
            raise gate.GateError("terminal verification receipt already exists")
    return result_path, snapshot_path


def cgroup_snapshot(
    operation: str, current: dict[str, object] | None
) -> tuple[dict[str, object], str, str]:
    """Read parent bounds and enumerate every populated descendant once."""
    fields = "ControlGroup,MemoryCurrent,MemoryHigh,MemoryMax,MemorySwapMax,CPUQuotaPerSecUSec"
    props = dict(
        line.split("=", 1)
        for line in run(
            ["systemctl", "--user", "show", "wnn-campaign.slice", "-p", fields]
        ).stdout.splitlines()
    )
    parent = Path("/sys/fs/cgroup") / props["ControlGroup"].lstrip("/")
    parent_procs = (parent / "cgroup.procs").read_text().strip()
    top: list[dict[str, object]] = []
    populated: list[str] = []
    roots = {path.name: path for path in parent.iterdir() if path.is_dir()}
    for name, path in roots.items():
        events_path = path / "cgroup.events"
        if not events_path.is_file():
            raise gate.GateError(f"child cgroup lacks events: {name}")
        events = dict(line.split() for line in events_path.read_text().splitlines())
        if events.get("populated") not in {"0", "1"}:
            raise gate.GateError(f"invalid populated state: {name}")
        if events["populated"] != "1":
            continue
        quota, period = (path / "cpu.max").read_text().split()
        if quota == "max":
            raise gate.GateError(f"unbounded child CPU: {name}")
        top.append(
            {
                "unit": name,
                "cpu_cores": int(quota) / int(period),
                "memory_max": int((path / "memory.max").read_text()),
                "memory_current": int((path / "memory.current").read_text()),
                "swap_max": int((path / "memory.swap.max").read_text()),
                "populated": True,
            }
        )
        for descendant in (path, *path.rglob("cgroup.events")):
            event_file = descendant / "cgroup.events" if descendant.is_dir() else descendant
            if not event_file.is_file():
                continue
            values = dict(line.split() for line in event_file.read_text().splitlines())
            if values.get("populated") == "1":
                relative = event_file.parent.relative_to(parent).as_posix()
                populated.append(relative)
    snap = {
        "parent_properties": props,
        "parent_cgroup_procs": parent_procs,
        "top_units": top,
        "populated_paths": sorted(set(populated)),
    }
    admitted = gate.validate_resource_snapshot(snap, current, operation)
    events = (parent / "memory.events").read_text()
    return (
        {**snap, "admission": admitted},
        events,
        run(
            ["findmnt", "-T", str(B / "research_tmp"), "-no", "FSTYPE,SOURCE,TARGET"]
        ).stdout.strip(),
    )


def helper_command(arm: str, operation: str) -> list[str]:
    """Build the frozen collector or verifier invocation from plan paths."""
    common = support_common()
    plan = common.validate_plan(PLAN)
    common.arm_spec(plan, arm, "terminal")
    collect = collect_paths(arm)
    output = (
        collect["receipt"]
        if operation == "collector"
        else B / f"receipts/RANK01_{arm.upper()}_F3_S1001_TERMINAL_VERIFICATION.json"
    )
    tool = SUPPORT / (
        "collect_f3_checkpoint.py" if operation == "collector" else "verify_f3_checkpoint.py"
    )
    argv = [
        PYTHON,
        "-B",
        str(tool),
        "--arm",
        arm,
        "--phase",
        "terminal",
        "--plan",
        str(PLAN),
        "--output",
        str(output),
    ]
    if operation == "collector":
        argv += ["--campaign-root", str(B), "--stage1-snapshot", str(STAGE1_PATHS[arm]["snapshot"])]
    else:
        peer = "deep" if arm == "wide" else "wide"
        argv += [
            "--snapshot-dir",
            str(collect["snapshot"]),
            "--source-root",
            str(B / "source-a266b036"),
            "--collection-receipt",
            str(collect["receipt"]),
            "--fold-manifest",
            str(FOLD),
            "--data-registry",
            str(REGISTRY),
            "--archive-manifest",
            str(ARCHIVES),
            "--stage1-snapshot-dir",
            str(STAGE1_PATHS[arm]["snapshot"]),
            "--stage1-verification",
            str(STAGE1_PATHS[arm]["verification"]),
            "--peer-stage1-snapshot-dir",
            str(STAGE1_PATHS[peer]["snapshot"]),
            "--peer-stage1-verification",
            str(STAGE1_PATHS[peer]["verification"]),
        ]
    return argv


def launch(args: argparse.Namespace) -> dict[str, object]:
    """Perform one fresh fail-closed admission and start bounded support."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,63}", args.invocation):
        raise gate.GateError("invocation is not a safe lowercase slug")
    manifest = verify_packet(args)
    pair = load_stage1_pair()
    plan, native, current, attempt_sha = sequence_and_attempt(args.arm)
    result_path, snapshot_path = validate_output_state(args.arm, args.operation)
    sequence = json.loads(SEQUENCE.read_bytes())
    resource, memory_events, scratch = cgroup_snapshot(args.operation, current)
    if scratch.split()[0] in {"tmpfs", "ramfs"}:
        raise gate.GateError("physical scratch is not on disk")
    stats = os.statvfs(B)
    if stats.f_bavail * stats.f_frsize <= 1_000_000_000:
        raise gate.GateError("less than 1 GB physical free space")
    unit = f"computer-vision-homework-r01-f3-{args.arm}-terminal-{args.operation}-{args.invocation}.service"
    result = REMOTE_RECORDS / args.invocation / f"{args.arm}_{args.operation}"
    if (
        os.path.lexists(result)
        or run(["systemctl", "--user", "show", unit, "-p", "LoadState", "--value"]).stdout.strip()
        != "not-found"
    ):
        raise gate.GateError("remote operation record or support unit already exists")
    command = [
        "systemd-run",
        "--user",
        "--no-block",
        "--unit=" + unit,
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        "--property=Slice=wnn-campaign.slice",
        "--property=CPUQuota=25%" if args.operation == "collector" else "--property=CPUQuota=50%",
        "--property=MemoryMax=512000000"
        if args.operation == "collector"
        else "--property=MemoryMax=1500000000",
        "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=120",
        "--property=TimeoutStopSec=15",
        "--working-directory=" + str(B),
    ]
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TMPDIR": str(B / "research_tmp"),
    }.items():
        command.append(f"--setenv={key}={value}")
    command.extend(helper_command(args.arm, args.operation))
    if REMOTE_RECORDS.is_symlink():
        raise gate.GateError("remote support receipt root is a symlink")

    # Collection or verification may take seconds to stage after the first
    # pointer read. Re-read the sequence, native attempt, populated tree and
    # canonical output collisions immediately before the create-only admission.
    fresh_plan, fresh_native, fresh_current, fresh_attempt_sha = sequence_and_attempt(args.arm)
    if fresh_plan != plan or fresh_native != native or fresh_attempt_sha != attempt_sha:
        raise gate.GateError("native plan or successful attempt changed during preflight")
    pair = load_stage1_pair()
    sequence = json.loads(SEQUENCE.read_bytes())
    if gate.validate_pointer(fresh_plan, sequence, args.arm) != fresh_current:
        raise gate.GateError("sequence pointer changed during final resource admission")
    result_path, snapshot_path = validate_output_state(args.arm, args.operation)
    resource, memory_events, scratch = cgroup_snapshot(args.operation, fresh_current)
    if scratch.split()[0] in {"tmpfs", "ramfs"}:
        raise gate.GateError("physical scratch is not on disk")
    stats = os.statvfs(B)
    free_bytes = stats.f_bavail * stats.f_frsize
    if free_bytes <= 1_000_000_000:
        raise gate.GateError("less than 1 GB physical free space")
    scratch_stats = os.statvfs(B / "research_tmp")
    scratch_free_bytes = scratch_stats.f_bavail * scratch_stats.f_frsize
    if scratch_free_bytes < 2_000_000_000:
        raise gate.GateError("physical research_tmp has less than 2 GB free")
    stable_plan, stable_native, stable_current, stable_attempt_sha = sequence_and_attempt(args.arm)
    stable_sequence = json.loads(SEQUENCE.read_bytes())
    if (
        stable_plan != plan
        or stable_native != native
        or stable_attempt_sha != fresh_attempt_sha
        or gate.validate_pointer(stable_plan, stable_sequence, args.arm) != fresh_current
        or stable_current != fresh_current
    ):
        raise gate.GateError("sequence or successful attempt changed during resource snapshot")
    sequence = stable_sequence
    fresh_current = stable_current
    result_path, snapshot_path = validate_output_state(args.arm, args.operation)
    if (
        os.path.lexists(result)
        or run(["systemctl", "--user", "show", unit, "-p", "LoadState", "--value"]).stdout.strip()
        != "not-found"
    ):
        raise gate.GateError("remote operation record or support unit appeared during preflight")
    REMOTE_RECORDS.mkdir(parents=True, exist_ok=True)
    invocation_dir = REMOTE_RECORDS / args.invocation
    if os.path.lexists(invocation_dir):
        raise gate.GateError("remote invocation directory already exists")
    invocation_dir.mkdir(parents=True, exist_ok=False)
    result.mkdir(exist_ok=False)
    admission = {
        "schema": "priority25.f3_terminal_support_admission.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "status": "ADMITTED",
        "arm": args.arm,
        "operation": args.operation,
        "invocation": args.invocation,
        "unit": unit,
        "plan_sha256": gate.PLAN_SHA256,
        "native_attempt_sha256": fresh_attempt_sha,
        "native_attempt": native,
        "sequence": sequence,
        "current_step": fresh_current,
        "stage1_pair": pair,
        "wrapper_manifest_sha256": args.manifest_sha256,
        "wrapper_review_sha256": args.review_sha256,
        "runtime_files": manifest["runtime_files"],
        "checkpoint_support_manifest_sha256": SUPPORT_MANIFEST_SHA,
        "checkpoint_support_review_sha256": SUPPORT_REVIEW_SHA,
        "resources": resource,
        "parent_memory_events_before": memory_events,
        "physical_scratch": scratch,
        "free_bytes": free_bytes,
        "physical_scratch_free_bytes": scratch_free_bytes,
        "collision_paths_checked_absent": [
            str(result_path),
            *([str(snapshot_path)] if args.operation == "collector" else []),
        ],
        "existing_collection_inputs_verified": (
            [str(collect_paths(args.arm)["receipt"]), str(collect_paths(args.arm)["snapshot"])]
            if args.operation == "verifier"
            else []
        ),
        "command": command,
    }
    admission_raw = write_json(result / "ADMISSION_V1.json", admission)
    started = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    after = (
        dict(
            line.split("=", 1)
            for line in run(
                ["systemctl", "--user", "show", unit, "-p", "Id,LoadState,ActiveState,SubState"]
            ).stdout.splitlines()
        )
        if started.returncode == 0
        else {}
    )
    service_started = (
        after.get("Id") == unit
        and after.get("LoadState") == "loaded"
        and after.get("ActiveState") in {"active", "activating"}
    )
    launch_record = {
        "schema": "priority25.f3_terminal_support_launch.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "unit": unit,
        "invocation": args.invocation,
        "admission_sha256": hashlib.sha256(admission_raw).hexdigest(),
        "returncode": started.returncode,
        "stdout": started.stdout,
        "stderr": started.stderr,
        "post_launch_properties": after,
        "service_started": service_started,
    }
    launch_raw = write_json(result / "LAUNCH_V1.json", launch_record)
    return {
        "status": "LAUNCHED" if service_started else "LAUNCH_UNCONFIRMED",
        "remote_record_dir": str(result),
        "unit": unit,
        "service_started": service_started,
        "admission": base64.b64encode(admission_raw).decode(),
        "launch": base64.b64encode(launch_raw).decode(),
    }


def capture(args: argparse.Namespace) -> dict[str, object]:
    """Capture raw output of exactly our exited support unit; never stop it here."""
    verify_packet(args)
    record = operation_record_dir(args.invocation, args.arm, args.operation)
    unit = f"computer-vision-homework-r01-f3-{args.arm}-terminal-{args.operation}-{args.invocation}.service"
    for name in ("ADMISSION_V1.json", "LAUNCH_V1.json"):
        path = record / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
            raise gate.GateError(f"missing support evidence: {name}")
    capture_path = record / "CAPTURE_V1.json"
    existing = read_existing_capture(capture_path, unit)
    if existing is not None:
        return existing
    props = dict(
        line.split("=", 1)
        for line in run(
            ["systemctl", "--user", "show", unit, "-p", SYSTEMD_FIELDS]
        ).stdout.splitlines()
    )
    if props.get("Id") != unit or props.get("SubState") not in {"exited", "failed"}:
        return {"status": "WAITING_SUPPORT_EXIT", "unit": unit, "properties": props}
    output_path, _ = output_paths(args.arm, args.operation)
    raw_files = {
        "ADMISSION_V1.json": record / "ADMISSION_V1.json",
        "LAUNCH_V1.json": record / "LAUNCH_V1.json",
    }
    missing_paths: list[str] = []
    if output_path.is_file() and not output_path.is_symlink():
        raw_files[output_path.name] = output_path
    else:
        missing_paths.append(str(output_path))
    packed: dict[str, object] = {}
    for name, path in raw_files.items():
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
            raise gate.GateError(f"capture input missing, symlinked, or too large: {path}")
        raw = path.read_bytes()
        packed[name] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "base64": base64.b64encode(raw).decode(),
        }
    parent = Path("/sys/fs/cgroup") / run(
        ["systemctl", "--user", "show", "wnn-campaign.slice", "-p", "ControlGroup", "--value"]
    ).stdout.strip().lstrip("/")
    populated = []
    for event in parent.rglob("cgroup.events"):
        values = dict(line.split() for line in event.read_text().splitlines())
        if values.get("populated") == "1":
            populated.append(event.parent.relative_to(parent).as_posix())
    journal = run(["journalctl", "--user", "-u", unit, "-n", "80", "--no-pager"]).stdout
    if len(journal.encode()) > 500_000:
        journal = journal[-500_000:]
    admission = json.loads((record / "ADMISSION_V1.json").read_bytes())
    evidence = {
        "schema": "priority25.f3_terminal_support_capture.v1",
        "captured_utc": dt.datetime.now(dt.UTC).isoformat(),
        "status": "TERMINAL_CAPTURE",
        "unit": unit,
        "properties": props,
        "raw_files": packed,
        "missing_raw_paths": missing_paths,
        "journal_tail": journal,
        "parent_memory_events_before": admission.get("parent_memory_events_before"),
        "parent_memory_events": (parent / "memory.events").read_text(),
        "populated_descendant_paths_before_stop": sorted(populated),
        "unit_success": props.get("SubState") == "exited"
        and props.get("Result") == "success"
        and props.get("ExecMainStatus") == "0",
        "stop_after_local_durable_capture": True,
    }
    write_json(capture_path, evidence)
    return evidence


def stop(args: argparse.Namespace) -> dict[str, object]:
    """Stop only this unit after its remote capture record is already durable."""
    verify_packet(args)
    record = operation_record_dir(args.invocation, args.arm, args.operation)
    capture_path = record / "CAPTURE_V1.json"
    if capture_path.is_symlink() or not capture_path.is_file():
        raise gate.GateError("remote capture record is not durable")
    if capture_path.stat().st_size > 10_000_000:
        raise gate.GateError("remote capture record exceeds its size bound")
    capture_record = json.loads(capture_path.read_bytes())
    if (
        capture_record.get("stop_after_local_durable_capture") is not True
        or capture_record.get("status") != "TERMINAL_CAPTURE"
        or capture_record.get("unit")
        != f"computer-vision-homework-r01-f3-{args.arm}-terminal-{args.operation}-{args.invocation}.service"
    ):
        raise gate.GateError("remote capture does not authorize post-capture unit stop")
    unit = f"computer-vision-homework-r01-f3-{args.arm}-terminal-{args.operation}-{args.invocation}.service"
    if os.path.lexists(record / "STOP_V1.json"):
        raise gate.GateError("remote stop record already exists; do not repeat stop")
    props = dict(
        line.split("=", 1)
        for line in run(
            ["systemctl", "--user", "show", unit, "-p", "Id,SubState,ExecMainStatus"]
        ).stdout.splitlines()
    )
    if props.get("Id") != unit or props.get("SubState") not in {"exited", "failed"}:
        raise gate.GateError("refusing to stop a different or live support unit")
    stopped = subprocess.run(
        ["systemctl", "--user", "stop", unit],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    record_value = {
        "schema": "priority25.f3_terminal_support_stop.v1",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "unit": unit,
        "pre_stop": props,
        "capture_sha256": digest(capture_path),
        "after_durable_capture": True,
        "returncode": stopped.returncode,
        "stdout": stopped.stdout,
        "stderr": stopped.stderr,
        "capture_unit_success": capture_record.get("unit_success"),
    }
    write_json(record / "STOP_V1.json", record_value)
    return record_value


def main() -> None:
    """Run one explicit root-gated collector or verifier operation on B."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "capture", "stop"))
    parser.add_argument("--arm", choices=("wide", "deep"), required=True)
    parser.add_argument("--operation", choices=("collector", "verifier"), required=True)
    parser.add_argument("--invocation", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--remote-sha256", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{7,63}", args.invocation):
        raise gate.GateError("invocation is not a safe lowercase slug")
    value = {"launch": launch, "capture": capture, "stop": stop}[args.mode](args)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
