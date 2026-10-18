#!/usr/bin/env python3
"""Collect one stable F1/seed1001 checkpoint boundary without fitting."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from f1_checkpoint_common import (
    ARTIFACTS,
    REMOTE_ROOT,
    arm_spec,
    expected_counts,
    mapping,
    read_json,
    require,
    sha256_file,
    validate_attempt,
    validate_history,
    validate_optional_epoch0_binding,
    validate_parent_headroom,
    validate_plan,
    validate_systemd_runtime_limit,
)


def _stats(path: Path) -> dict[str, int]:
    """Record a source generation around a bounded copy."""
    value = path.stat(follow_symlinks=False)
    return {"inode": value.st_ino, "size": value.st_size, "mtime_ns": value.st_mtime_ns}


def reserve_archive(path: Path) -> None:
    """Reserve an archive directory atomically without replacing prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)


def _paths(base: Path, spec: dict[str, object]) -> tuple[Path, Path, dict[str, Path]]:
    """Resolve the exact source and archive paths for one declared boundary."""
    run = str(spec["run"])
    source = base / "data/artifacts/runs" / run
    if spec["phase"] == "epoch0":
        archive = base / "checkpoint_snapshots" / run / "epoch0"
    elif spec["phase"] == "stage1":
        archive = base / "checkpoint_snapshots" / run / "stage1_epoch1"
    else:
        archive = base / "checkpoint_snapshots" / run / "epoch5_terminal"
    files = {name: source / name for name in cast(tuple[str, ...], spec["artifacts"])}
    attempt = str(spec["attempt"])
    files["attempt.json"] = base / "runtime/attempts" / f"{attempt}.json"
    files["training.log"] = base / "runtime/logs" / f"{attempt}.log"
    return source, archive, files


def _live_resource_envelope() -> dict[str, object]:
    """Read parent headroom and this service's cgroup CPU/memory ceilings."""
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
    require(quota > 0 and period > 0 and quota / period <= 0.25, "collector CPU cgroup limit")
    require(0 < memory_max <= 512_000_000 and swap_max == 0, "collector memory cgroup limit")
    service_name = service_path.rstrip("/").split("/")[-1]
    require(service_name.endswith(".service"), "collector must run in a systemd service")
    runtime = subprocess.check_output(
        ["systemctl", "--user", "show", service_name, "--property=RuntimeMaxUSec", "--value"],
        text=True,
    ).strip()
    validate_systemd_runtime_limit(runtime)
    parent_line = subprocess.check_output(
        ["systemctl", "--user", "show", "wnn-campaign.slice", "-p", "ControlGroup", "--value"],
        text=True,
    ).strip()
    parent = Path("/sys/fs/cgroup") / parent_line.lstrip("/")
    current = int((parent / "memory.current").read_text())
    maximum = int((parent / "memory.max").read_text())
    parent_swap = int((parent / "memory.swap.max").read_text())
    parent_path = parent_line.rstrip("/")
    require(service_path.startswith(parent_path + "/"), "collector is outside campaign slice")
    require(parent_swap == 0, "campaign slice swap must be disabled")
    validate_parent_headroom(current, min(maximum, 17_000_000_000))
    return {
        "service": service_name,
        "parent_memory_current_bytes": current,
        "parent_memory_max_bytes": maximum,
        "parent_memory_admission_limit_bytes": min(maximum, 17_000_000_000),
        "collector_cpu_max": (service / "cpu.max").read_text().strip(),
        "collector_memory_max_bytes": memory_max,
        "collector_memory_swap_max_bytes": swap_max,
        "collector_runtime_seconds": 120,
    }


def collect_snapshot(
    base: Path,
    arm: str,
    phase: str,
    plan_path: Path,
    output_path: Path,
    *,
    epoch0_receipt: Path | None = None,
    stage1_snapshot: Path | None = None,
    resource_envelope: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Copy/hash one F1 boundary and refuse unstable, stale or mismatched inputs.

    Args:
        base: Existing campaign directory on physical Beleriand storage.
        arm: ``wide`` or ``deep`` from the frozen F1 pair plan.
        phase: ``epoch0``, ``stage1`` or ``terminal`` checkpoint boundary.
        plan_path: Immutable F1 plan staged at its declared path.
        output_path: Create-only local collection receipt path.
        epoch0_receipt: Optional earlier F1 epoch0 receipt; if supplied, its model
            SHA must match the epoch0 checkpoint included in the stage1 snapshot.
        stage1_snapshot: Exact same-arm stage1 archive required for terminal.
        resource_envelope: Observed live systemd/cgroup admission. Required so
            the receipt records enforced outer limits rather than assertions.

    Raises:
        FileExistsError: A destination or receipt already exists.
        ValueError: Producer, attempt, ancestry, size or source generation differs.
    """
    require(base.resolve() == REMOTE_ROOT, "campaign root")
    require(resource_envelope is not None, "live support resource envelope is required")
    require(not output_path.exists(), "collection receipt already exists")
    plan = validate_plan(plan_path)
    spec = arm_spec(plan, arm, phase)
    if phase == "terminal":
        require(stage1_snapshot is not None, "terminal stage1 snapshot is required")
    source, destination, paths = _paths(base, spec)
    require(not destination.exists(), "checkpoint archive already exists")
    before = {name: _stats(path) for name, path in paths.items()}
    require(sum(row["size"] for row in before.values()) <= 100_000_000, "archive size bound")
    require(all(row["size"] > 0 for row in before.values()), "empty producer artifact")
    require(
        all(path.is_file() and not path.is_symlink() for path in paths.values()),
        "producer file type",
    )
    attempt = validate_attempt(paths["attempt.json"], plan, spec)
    provenance = read_json(paths["provenance.json"])
    expected_epochs = int(spec["completed_epochs"])
    require(
        provenance.get("cumulative_updates") == expected_counts(expected_epochs),
        "producer cumulative update counts",
    )
    if phase != "epoch0":
        history = json.loads(paths["history.json"].read_text())
        validate_history(history, expected_epochs)
    parent: dict[str, object] | None = None
    if phase == "stage1":
        previous = read_json(epoch0_receipt) if epoch0_receipt is not None else None
        validate_optional_epoch0_binding(
            previous,
            arm,
            str(spec["stage1_run"]),
            sha256_file(paths["model_ep0.pt"]),
        )
    elif phase == "terminal":
        parent_path = cast(Path, stage1_snapshot) / "training_state.pt"
        parent_provenance = read_json(cast(Path, stage1_snapshot) / "provenance.json")
        parent_sha = sha256_file(parent_path)
        expected_resume = {
            "path": str(
                base / "data/artifacts/runs" / str(spec["stage1_run"]) / "training_state.pt"
            ),
            "sha256": parent_sha,
            "next_epoch": 2,
        }
        require(provenance.get("resumed_from") == expected_resume, "terminal resume parent")
        require(
            parent_provenance.get("cumulative_updates") == expected_counts(2),
            "stage1 parent counters",
        )
        parent = expected_resume

    total_before = sum(row["size"] for row in before.values())
    require(total_before <= 100_000_000, "archive exceeds 100 MB bound")
    reserve_archive(destination)
    copied: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        target = destination / name
        with path.open("rb") as source_stream, target.open("xb") as target_stream:
            shutil.copyfileobj(source_stream, target_stream, length=1 << 20)
        shutil.copystat(path, target, follow_symlinks=False)
        copied[name] = {
            "path": str(target),
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
        require(copied[name]["sha256"] == sha256_file(path), f"source/copy hash {name}")
    after = {name: _stats(path) for name, path in paths.items()}
    require(before == after, "producer generation changed during copy")
    for name in cast(tuple[str, ...], spec["artifacts"]):
        if name.startswith("model_ep") or name == "model.pt":
            expected = mapping(provenance["checkpoint_sha256"], "producer checkpoint hashes")[name]
            require(copied[name]["sha256"] == expected, f"producer model hash {name}")
    final_artifacts = {
        name: {**copied[name], "path": str(destination / name)}
        for name in cast(tuple[str, ...], spec["artifacts"])
    }
    attempt_evidence = {
        **copied["attempt.json"],
        **{key: attempt[key] for key in ("id", "status", "exit_status")},
    }
    receipt = {
        "schema": "priority25.f1_checkpoint_collection.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "arm": arm,
        "phase": phase,
        "run": spec["run"],
        "attempt": spec["attempt"],
        "source_directory": str(source),
        "snapshot_directory": str(destination),
        "plan_sha256": sha256_file(plan_path),
        "attempt_record_sha256": attempt["record_sha256"],
        "source_stats_before": before,
        "source_stats_after": after,
        "artifacts": final_artifacts,
        "attempt_evidence": attempt_evidence,
        "log_evidence": copied["training.log"],
        "source_snapshot_hash_match": True,
        "producer_model_hash_matches": True,
        "resumed_from": parent,
        "collector_sha256": sha256_file(Path(__file__)),
        "collector_dependencies": {
            "f1_checkpoint_common.py": sha256_file(
                Path(__file__).with_name("f1_checkpoint_common.py")
            )
        },
        "limits": {
            "cpu_quota": "25%",
            "memory_bytes": 512_000_000,
            "swap_bytes": 0,
            "runtime_seconds": 120,
            "gpu": False,
        },
        "resource_envelope": dict(cast(Mapping[str, object], resource_envelope)),
        "scientific_status": "F1 checkpoint archive only; independent structural verification and quality grade pending",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    """Collect one declared F1 boundary on B; do not launch a fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("wide", "deep"), required=True)
    parser.add_argument("--phase", choices=tuple(ARTIFACTS), required=True)
    parser.add_argument("--campaign-root", type=Path, default=REMOTE_ROOT)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch0-receipt", type=Path)
    parser.add_argument("--stage1-snapshot", type=Path)
    args = parser.parse_args()
    envelope = _live_resource_envelope()
    receipt = collect_snapshot(
        args.campaign_root,
        args.arm,
        args.phase,
        args.plan,
        args.output,
        epoch0_receipt=args.epoch0_receipt,
        stage1_snapshot=args.stage1_snapshot,
        resource_envelope=envelope,
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
