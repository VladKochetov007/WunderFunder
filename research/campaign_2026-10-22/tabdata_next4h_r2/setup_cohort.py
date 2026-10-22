"""Freeze mounted inputs, finite GPU allocation and every executable file."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from delivery import PACKET, ROOT, STATE, UNIT, read, sha, write


def link(path: Path, target: Path) -> None:
    """Create only a new owned link, rejecting accidental path replacement."""
    if path.exists() or path.is_symlink():
        if not path.is_symlink() or path.resolve() != target.resolve():
            raise ValueError(f"Unexpected existing path: {path}")
    else:
        path.symlink_to(target, target_is_directory=target.is_dir())


def main() -> None:
    """Prepare read-only input mounts and seal one nonrenewable cohort."""
    if (PACKET / "PLAN.json").exists() or (STATE / "START.json").exists():
        raise FileExistsError("Existing frozen allocation must not be replaced")
    if "29 passed" not in (STATE / "numeric-tests-rerun.log").read_text():
        raise ValueError("Numeric receiver tests have not passed")
    project = STATE / "nll_project"
    (project / "cache").mkdir(parents=True, exist_ok=True)
    (project / "artifacts/runs").mkdir(parents=True, exist_ok=True)
    link(project / "datasets", ROOT / "data/datasets")
    link(project / "splits", ROOT / "splits")
    link(project / "src", PACKET / "sources/nll/src")
    link(project / "scripts", PACKET / "sources/nll/scripts")
    for source in (ROOT / "data/cache").glob("*.npz"):
        link(project / "cache" / source.name, source)
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).splitlines()
    devices = [[cell.strip() for cell in row.split(",")] for row in rows]
    if len(devices) != 8 or any("B300" not in row[2] for row in devices):
        raise ValueError("Expected eight authorized B300 GPUs")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        text=True,
        timeout=10,
    )
    if processes.strip():
        raise ValueError("Node availability changed; reassess foreign workload before allocation")
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 24:
        raise ValueError("Insufficient CPU lanes for bounded independent data loading")
    jobs = []
    for fold, seed in ((0, 1001), (2, 1002)):
        for arm in ("raw_events", "uniform16", "quantile16"):
            gpu = 2 + len(jobs)
            jobs.append(
                {
                    "id": f"numeric_f{fold}_{arm}",
                    "family": "numeric",
                    "fold": fold,
                    "seed": seed,
                    "arm": arm,
                    "gpu_index": gpu,
                }
            )
    for fold, seed, gpu in ((0, 1201, 0), (2, 1202, 1)):
        for arm in ("wmse", "nll"):
            jobs.append(
                {
                    "id": f"nll_f{fold}_s{seed}_{arm}",
                    "family": "nll",
                    "fold": fold,
                    "seed": seed,
                    "arm": arm,
                    "gpu_index": gpu,
                }
            )
    for index, job in enumerate(jobs):
        job["gpu_uuid"] = next(row[1] for row in devices if int(row[0]) == job["gpu_index"])
        job["cpus"] = available[index * 2 : index * 2 + 2]
    files = {
        str(path): sha(path)
        for path in PACKET.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".json", ".toml", ".lock"}
        and "__pycache__" not in path.parts
        and path.name != "PLAN.json"
    }
    for fold in (0, 2):
        path = ROOT / "splits" / f"fold{fold}.json"
        files[str(path)] = sha(path)
    write(
        PACKET / "PLAN.json",
        {
            "schema": "tabdata-next4h-r2.v1",
            "unit": UNIT,
            "max_wall_seconds": 14400,
            "runtime_max_seconds": 13500,
            "stop_grace_seconds": 120,
            "finalizer_timeout_seconds": 110,
            "conservative_all_in_upper_seconds": 13740,
            "fit_cutoff_seconds": 11100,
            "export_cutoff_seconds": 12900,
            "grade_cutoff_seconds": 13140,
            "epochs": 3,
            "fixed_epoch": 2,
            "partial_progress_allowed": True,
            "partial_is_quality_endpoint": False,
            "jobs": jobs,
            "files": files,
            "inputs": read(STATE / "INPUTS.json"),
            "baseline_gpu_rows": devices,
            "mask_readout": "not admitted: exact parent assets absent; zero substitution",
            "source_commits": {
                "numeric": "20faaa1abddb79f342a3b66ba0e5623a15c65a25",
                "nll": "6577b88e",
                "mask": "eceb95c0536bf03ff260379e8f86a52cb4f5d56b",
            },
            "new_fits": 10,
            "new_teacher_or_parent_fits": 0,
            "scientific_precision": "strict FP32",
            "evaluation": "unchanged official clip2 pooled WP; full exposed evaluation components",
            "authority": "User: run next experiments on eight B300s, several epochs, useful concurrency; strict four-hour wall and cleanup/publication retained",
        },
    )
    print("SEALED_PLAN", sha(PACKET / "PLAN.json"), "JOBS", len(jobs))


if __name__ == "__main__":
    main()
