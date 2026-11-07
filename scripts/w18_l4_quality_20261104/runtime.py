"""Bounded local runtime checks for one W18 L4 cohort."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import cast

from scripts.w18_l4_quality_20261104.contract import CohortPanel, ContractError, read_object
from wnn.multistage_search_runtime import process_identity


def cell_paths(panel: CohortPanel, cell: dict[str, object]) -> dict[str, Path]:
    """Derive the one-time output paths for a frozen fit cell."""
    paths = cast(dict[str, object], panel.value["paths"])
    runtime, project = Path(str(paths["runtime_root"])), Path(str(paths["project_root"]))
    cell_id, run_name = str(cell["cell_id"]), str(cell["run_name"])
    return {
        "profile_log": runtime / "logs" / "profiles" / f"{cell_id}.log",
        "profile_receipt": runtime / "receipts" / "profiles" / f"{cell_id}.json",
        "profile_request": runtime / "requests" / "profiles" / f"{cell_id}.json",
        "profile_run_dir": project / "artifacts" / "runs" / f"{run_name}_profile_r1",
        "fit_log": runtime / "logs" / "fits" / f"{cell_id}.log",
        "fit_receipt": runtime / "receipts" / "fits" / f"{cell_id}.json",
        "run_dir": project / "artifacts" / "runs" / run_name,
        "prediction": project / "cache" / f"pred_{run_name}_valid.npy",
        "export_log": runtime / "logs" / "exports" / f"{cell_id}.log",
        "export_receipt": runtime / "receipts" / "exports" / f"{cell_id}.json",
    }


def runtime_bytes(panel: CohortPanel) -> int:
    """Count regular new runtime, checkpoint and prediction files, ignoring links."""
    paths = cast(dict[str, object], panel.value["paths"])
    roots = [Path(str(paths["runtime_root"]))]
    for raw_cell in panel.value["cells"]:
        cell = cell_paths(panel, cast(dict[str, object], raw_cell))
        roots.extend(
            (
                cell["profile_run_dir"],
                cell["run_dir"],
                cell["prediction"],
                cell["prediction"].with_name(cell["prediction"].name + ".partial"),
                cell["prediction"].with_suffix(".json"),
            )
        )
    total = 0
    seen: set[tuple[int, int]] = set()
    for root in roots:
        candidates = root.rglob("*") if root.is_dir() and not root.is_symlink() else (root,)
        for item in candidates:
            try:
                if not item.is_symlink() and item.is_file():
                    stat = item.stat()
                    identity = (stat.st_dev, stat.st_ino)
                    if identity not in seen:
                        total += stat.st_size
                        seen.add(identity)
            except FileNotFoundError:
                continue
    return total


def gpu_snapshot(gpu_uuid: str) -> dict[str, object]:
    """Read the selected device identity and headroom without inspecting foreign PIDs."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--id=" + gpu_uuid,
            "--query-gpu=uuid,name,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows = [row.strip().split(",") for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError(f"cannot identify the assigned GPU {gpu_uuid}")
    uuid, name, free_mib, total_mib, utilization = (item.strip() for item in rows[0])
    if uuid != gpu_uuid or "L4" not in name:
        raise RuntimeError(f"assigned device is not the admitted L4: {uuid} {name}")
    return {
        "uuid": uuid,
        "name": name,
        "free_mib": int(free_mib),
        "total_mib": int(total_mib),
        "utilization_percent": int(utilization),
    }


def check_budget(panel: CohortPanel) -> dict[str, object]:
    """Enforce the frozen new-output cap and physical free-space reserve."""
    budget, paths = panel.value["budget"], panel.value["paths"]
    disk = shutil.disk_usage(Path(str(paths["disk_mount"])))
    written = runtime_bytes(panel)
    if disk.free < int(budget["minimum_free_bytes"]):
        raise RuntimeError("data2 free space crossed the frozen reserve")
    if written > int(budget["max_output_bytes"]):
        raise RuntimeError("new runtime output crossed the frozen byte cap")
    return {"disk_free_bytes": disk.free, "output_bytes": written}


def state_path(panel: CohortPanel) -> Path:
    """Return the durable finite-wave state path."""
    paths = cast(dict[str, object], panel.value["paths"])
    return Path(str(paths["runtime_root"])) / "COHORT_STATE_R1.json"


def load_or_start_state(panel: CohortPanel, phase: str) -> dict[str, object]:
    """Create the first-profile deadline once and enforce strict phase ordering."""
    path = state_path(panel)
    if path.is_file():
        state = read_object(path)
        if state.get("panel_sha256") != panel.sha256:
            raise RuntimeError("cohort state belongs to a different frozen panel")
        expected = {"fit": "profile_complete", "export": "fit_complete"}.get(phase)
        if phase == "profile" or state.get("phase") != expected:
            raise RuntimeError(f"phase {phase!r} cannot follow cohort state {state.get('phase')!r}")
        return state
    if phase != "profile":
        raise RuntimeError("profile phase must create the original cohort deadline first")
    budget = cast(dict[str, object], panel.value["budget"])
    started = time.time()
    state: dict[str, object] = {
        "schema": "wnn.w18-l4-cohort-state.v1",
        "panel_sha256": panel.sha256,
        "first_profile_started_epoch": started,
        "hard_deadline_epoch": started + int(budget["max_wave_seconds"]),
        "collection_reserve_seconds": int(budget["collection_reserve_seconds"]),
        "profile_attempts": 0,
        "scientific_fit_attempts": 0,
        "phase": "profile",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(state, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return state


def remaining_seconds(state: dict[str, object]) -> int:
    """Return time before the reserved collection window begins."""
    remaining = int(float(state["hard_deadline_epoch"]) - time.time() - int(state["collection_reserve_seconds"]))
    if remaining <= 0:
        raise TimeoutError("only the reserved collection interval remains")
    return remaining


def environment(panel: CohortPanel, cell: dict[str, object]) -> dict[str, str]:
    """Build the selected-GPU, bounded-thread legacy W18 environment."""
    from wnn.multistage_search_runtime import base_environment

    paths = cast(dict[str, object], panel.value["paths"])
    threads = int(cell["cpu_threads"])
    env = base_environment(str(cell["gpu_uuid"]), Path(str(paths["source_root"])))
    env.update(
        {
            "WNN_PROJECT_ROOT": str(paths["project_root"]),
            "WNN_THREADS": str(threads),
            "WNN_LOADER_WORKERS": str(min(2, max(1, threads // 2))),
            "WNN_LOADER_RAM_GB": "8",
            "PYTHONUNBUFFERED": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "0",
        }
    )
    return env


def command_prefix(cell: dict[str, object], command: list[str]) -> list[str]:
    """Pin an argv vector to its disjoint declared CPU set when supplied."""
    affinity = cell.get("cpu_affinity")
    if affinity is None:
        return command
    if hasattr(os, "sched_getaffinity") and not set(cast(list[int], affinity)).issubset(
        os.sched_getaffinity(0)
    ):
        raise RuntimeError("frozen CPU affinity is outside the controller's allowed CPU set")
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset is required for the frozen CPU affinity assignment")
    return [taskset, "-c", ",".join(str(cpu) for cpu in cast(list[int], affinity)), *command]


def owned_rss_bytes(panel: CohortPanel, phase: str) -> int:
    """Sum RSS only for process groups recorded by this cohort's receipts."""
    paths = cast(dict[str, object], panel.value["paths"])
    phase_dir = Path(str(paths["runtime_root"])) / "receipts" / f"{phase}s"
    groups: set[int] = set()
    if phase_dir.is_dir():
        for path in phase_dir.glob("*.json"):
            if path.is_symlink():
                continue
            try:
                child = read_object(path).get("child")
            except (OSError, ContractError):
                continue
            if isinstance(child, dict) and type(child.get("pgid")) is int:
                groups.add(cast(int, child["pgid"]))
    total = 0
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return total
    for process in proc_root.iterdir():
        if not process.name.isdigit():
            continue
        identity = process_identity(int(process.name))
        if identity is None or identity.get("pgid") not in groups:
            continue
        try:
            for line in (process / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError):
            continue
    return total
