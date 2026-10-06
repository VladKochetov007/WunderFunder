from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from derive_source import digest, stage
from host_guard import gpu_sample, supervised, write_json

CAMPAIGN = "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4"
FIT_REQUIRED_OUTPUTS = (
    "model_ep3.pt",
    "model.pt",
    "provenance.json",
    "history.json",
    "training_state.pt",
)


def output_run_name(width: int, kind: str) -> str:
    """Give this immutable packet revision a unique run identity."""
    if kind not in {"probe", "fit"}:
        raise ValueError("run kind must be probe or fit")
    return f"c20261006night_w{width}_adamw_s3_f1_serialv3r1_{kind}"


def collect_fit_artifacts(root: Path, fit_name: str) -> list[dict[str, object]]:
    """Require and hash the exact run's checkpoints, state, provenance and history."""
    run_directory = root / "artifacts/runs" / fit_name
    if not run_directory.is_dir():
        raise FileNotFoundError(f"fresh fit output directory is missing: {run_directory}")
    missing = [name for name in FIT_REQUIRED_OUTPUTS if not (run_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"fit output files are missing: {', '.join(missing)}")
    members = sorted(run_directory.iterdir())
    invalid = [path.name for path in members if path.is_symlink() or not path.is_file()]
    if invalid:
        raise RuntimeError(f"fit output directory has unmanifested non-files: {', '.join(invalid)}")
    records = []
    for path in members:
        before = path.stat()
        value = digest(path)
        after = path.stat()
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise RuntimeError(f"fit output changed while hashing: {path}")
        records.append({"path": str(path), "sha256": value, "bytes": after.st_size})
    return records


def verify_inputs(root: Path, manifest: Path, output: Path) -> None:
    """Verify actual mounted bytes once, recording exact stat identities.

    A prior receipt or a same-name file is never accepted instead of hashing.
    Source members and external mount descriptors use the same byte checks.
    Only verified read-only files receive cache-release advice.
    """
    value = json.loads(manifest.read_text())
    rows = []
    for index, row in enumerate(value["files"]):
        source = root / row["path"]
        before = source.stat()
        if before.st_size != row["size"]:
            raise ValueError(f"input size differs: {row['path']}")
        observed = digest(source)
        after = source.stat()
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        current = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if observed != row["sha256"] or identity != current:
            raise ValueError(f"input identity differs: {row['path']}")
        rows.append({**row, "stat_identity": identity})
        if hasattr(os, "posix_fadvise"):
            descriptor = os.open(source, os.O_RDONLY)
            try:
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(descriptor)
        write_json(
            output / "verification_progress.json",
            {
                "verified_files": index + 1,
                "total_files": len(value["files"]),
                "last_path": row["path"],
                "time": time.time(),
            },
        )
    write_json(output / "verified_inputs.json", rows)


def check_probe(log_path: Path) -> dict[str, object]:
    """Require the original full-shape one-batch finite update gate."""
    records = [
        json.loads(line[4:])
        for line in log_path.read_text().splitlines()
        if line.startswith("FIT ")
    ]
    if len(records) != 1:
        raise ValueError("probe did not emit exactly one FIT record")
    event = records[0]
    counts = event.get("update_diagnostics", {}).get("epoch", {})
    if event.get("epoch") != 0 or event.get("evaluation") != "withheld":
        raise ValueError("probe epoch or evaluation policy changed")
    if counts.get("completed_batches") != 1 or counts.get("optimizer_completed_updates", 0) < 1:
        raise ValueError("no completed full-shape probe update")
    if counts.get("amp_skipped_updates") != 0:
        raise ValueError("probe skipped updates")
    if counts.get("ema_updates") != counts.get("optimizer_completed_updates"):
        raise ValueError("probe EMA update count differs")
    return counts


def worker_plan(root: Path, width: int, probe: bool) -> tuple[list[str], str]:
    """Load exact historical scientific argv, changing output identity only."""
    kind = "probe" if probe else "fit"
    old = f"c20261004_w{width}_adamw_s3_f1_t4v4_{kind}"
    new = output_run_name(width, kind)
    spec = json.loads((root / f"kaggle_specs/{old}.json").read_text())
    argv = spec["argv"]
    if "--plan" in argv:
        old_plan = root / argv[argv.index("--plan") + 1]
        plan = json.loads(old_plan.read_text())
        # Exact token replacement preserves data/split/schedule/optimizer flags.
        for step in plan["steps"]:
            step["argv"] = [item.replace(old, new) for item in step["argv"]]
        plan["report"] = f"continuation_width/{new}_steps.json"
        destination = root / f"continuation_width/{new}_plan.json"
        write_json(destination, plan)
        return [sys.executable, "scripts/kaggle_sequence.py", "--plan", str(destination)], new
    return [sys.executable, *[new if item == old else item for item in argv[1:]]], new


def run(root: Path, runtime: Path, width: int, manifest: Path) -> None:
    """Run one width per notebook; keep matched pair scoring external and frozen."""
    output = Path(f"/kaggle/working/width_{width}_serial_v3r1")
    output.mkdir(exist_ok=False)
    for kind in ("probe", "fit"):
        run_directory = root / "artifacts/runs" / output_run_name(width, kind)
        if run_directory.exists():
            raise FileExistsError(f"recovery run identity already exists: {run_directory}")
    verify_inputs(root, manifest, output)
    receipt = stage(root, manifest, runtime)
    write_json(output / "source_derivation.json", receipt)
    devices = gpu_sample()
    if len(devices) not in (1, 2) or any("T4" not in str(device["name"]) for device in devices):
        raise ValueError("requires measured Tesla T4 backend")
    if any(int(device["used_bytes"]) > 128 * 1024**2 for device in devices):
        raise ValueError("unexpected preexisting GPU usage")
    device = devices[0]
    total = int(device["total_bytes"])
    allocator = min(11 * 1024**3, int(total * 0.80))
    stop = min(13 * 1024**3, int(total * 0.90))
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(device["uuid"]),
            "WNN_GPU_MEMORY_BYTES": str(allocator),
            "WNN_LOADER_RAM_GB": "1",
            "WNN_LOADER_WORKERS": "0",
            "MALLOC_ARENA_MAX": "2",
            "ARROW_DEFAULT_MEMORY_POOL": "system",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(root / "src"),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "WNN_PORTABLE_DATA_MANIFEST": f"{CAMPAIGN}/portable_inputs_f1.json",
        }
    )
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "POLARS_MAX_THREADS",
        "ARROW_NUM_THREADS",
        "WNN_THREADS",
    ):
        env[name] = "2"
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    write_json(
        output / "admission_design.json",
        {
            "width": width,
            "scientific_concurrency": 1,
            "devices": devices,
            "allocator_bytes": allocator,
            "observed_gpu_stop_bytes": stop,
            "all_allocated_devices_charged": True,
            "host_target_fraction": 0.70,
            "host_stop_fraction": 0.85,
            "rss_stop_bytes": 8 * 1024**3,
        },
    )
    env["WNN_INTEGRATION_DEVICE"] = "cuda"
    supervised(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_trainer_resume_integration.py",
            "tests/test_training_state.py",
        ],
        root,
        output / "cuda_preflight",
        env,
        timeout=900,
        rss_limit=8 * 1024**3,
        gpu_uuid=str(device["uuid"]),
        gpu_stop=stop,
    )
    for probe in (True, False):
        argv, name = worker_plan(root, width, probe)
        result = supervised(
            argv,
            root,
            output / name,
            env,
            timeout=1800 if probe else 12600,
            rss_limit=8 * 1024**3,
            gpu_uuid=str(device["uuid"]),
            gpu_stop=stop,
        )
        if probe:
            write_json(output / "probe_update_gate.json", check_probe(output / name / "worker.log"))
        write_json(output / f"{name}_receipt.json", result)
    fit_name = output_run_name(width, "fit")
    required = [
        root / f"artifacts/runs/{fit_name}/model_ep3.pt",
        root / f"cache/pred_{fit_name}_valid.npy",
        root / f"cache/pred_{fit_name}_valid.json",
    ]
    if any(not path.is_file() for path in required):
        raise RuntimeError("fit or complete prediction export is missing")
    artifacts = collect_fit_artifacts(root, fit_name)
    for path in required[1:]:
        artifacts.append({"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size})
    write_json(output / "artifact_manifest.json", artifacts)
    write_json(
        output / "completion.json",
        {
            "status": "complete",
            "width": width,
            "scientific_claim": "none; requires frozen matched paired scoring",
        },
    )


def main() -> None:
    """Enter only from the reviewed private kernel bootstrap."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, choices=[256, 384], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    run(
        Path("/kaggle/working/experiment"),
        Path(__file__).resolve().parent,
        args.width,
        args.manifest,
    )


if __name__ == "__main__":
    main()
