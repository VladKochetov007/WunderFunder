from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from derive_source import digest, stage
from smallcap_host_guard import gpu_sample, supervised, write_json

CAMPAIGN = "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4"
PAIR_NAME = "c20261006night_smallcap_w256w384_adamw_s3_f1"
WIDTHS = (256, 384)
BASE_BATCH = 32
PREREG_BATCHES = (16, 8)
# Reserve four provider CPUs for this job, leaving the fifth aggregate CPU for
# concurrent local monitoring/research. The child uses three trainer threads;
# the supervisor, watchdog, and telemetry polling share the remaining budget.
CPU_BUDGET = 4
WORKER_THREADS = 3
GPU_ALLOCATOR_BYTES = 2_500_000_000
GPU_AGGREGATE_STOP_BYTES = 2_800_000_000
GPU_HARD_LIMIT_BYTES = 3_000_000_000
GPU_PRELAUNCH_BYTES = 400_000_000
FIT_TIMEOUT_SECONDS = 13_000
PROBE_TIMEOUT_SECONDS = 1_800
RSS_STOP_BYTES = 10 * 1024**3
FIT_REQUIRED_OUTPUTS = (
    "model_ep3.pt",
    "model.pt",
    "provenance.json",
    "history.json",
    "training_state.pt",
)


def output_run_name(batch_size: int, width: int, kind: str) -> str:
    """Bind each fresh probe and fit to this small-cap preregistration."""
    if batch_size not in PREREG_BATCHES or width not in WIDTHS or kind not in {"probe", "fit"}:
        raise ValueError("unsupported width or run kind")
    return f"{PAIR_NAME}_b{batch_size}_w{width}_{kind}"


def stable_file_record(path: Path) -> dict[str, object]:
    """Hash one finished artifact while rejecting concurrent modification."""
    before = path.stat()
    value = digest(path)
    after = path.stat()
    before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_id != after_id:
        raise RuntimeError(f"artifact changed while hashing: {path}")
    return {"path": str(path), "sha256": value, "bytes": after.st_size}


def collect_fit_artifacts(root: Path, fit_name: str) -> list[dict[str, object]]:
    """Require a complete fresh fit directory and hash every top-level file."""
    run_directory = root / "artifacts/runs" / fit_name
    if not run_directory.is_dir():
        raise FileNotFoundError(f"fresh fit output directory is missing: {run_directory}")
    missing = [name for name in FIT_REQUIRED_OUTPUTS if not (run_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"fit output files are missing: {', '.join(missing)}")
    members = sorted(run_directory.iterdir())
    invalid = [path.name for path in members if path.is_symlink() or not path.is_file()]
    if invalid:
        raise RuntimeError(f"unmanifested fit output type: {', '.join(invalid)}")
    return [stable_file_record(path) for path in members]


def verify_inputs(root: Path, manifest_path: Path, output: Path) -> None:
    """Verify actual mounted source/data bytes against the preserved manifest."""
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for index, row in enumerate(manifest["files"]):
        source = root / row["path"]
        before = source.stat()
        if before.st_size != row["size"]:
            raise ValueError(f"input size differs: {row['path']}")
        observed = digest(source)
        after = source.stat()
        before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if observed != row["sha256"] or before_id != after_id:
            raise ValueError(f"input identity differs: {row['path']}")
        rows.append({**row, "stat_identity": before_id})
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
                "total_files": len(manifest["files"]),
                "last_path": row["path"],
                "time": time.time(),
            },
        )
    write_json(output / "verified_inputs.json", rows)


def check_probe(log_path: Path) -> dict[str, object]:
    """Require exactly one finite, completed production-shape optimizer update."""
    events = [
        json.loads(line[4:])
        for line in log_path.read_text().splitlines()
        if line.startswith("FIT ")
    ]
    if len(events) != 1:
        raise ValueError("probe did not emit exactly one FIT record")
    event = events[0]
    counts = event.get("update_diagnostics", {}).get("epoch", {})
    if event.get("epoch") != 0 or event.get("evaluation") != "withheld":
        raise ValueError("probe epoch or evaluation policy changed")
    if counts.get("completed_batches") != 1:
        raise ValueError("probe did not process exactly one production batch")
    if counts.get("optimizer_attempts", 0) < 1 or counts.get("optimizer_completed_updates", 0) < 1:
        raise ValueError("probe did not complete a finite optimizer update")
    if counts.get("amp_skipped_updates") != 0:
        raise ValueError("probe had a skipped mixed-precision update")
    if counts.get("ema_updates") != counts.get("optimizer_completed_updates"):
        raise ValueError("probe EMA and optimizer update counts differ")
    return counts


def worker_plan(
    root: Path, width: int, kind: str, batch_size: int, plan_directory: Path
) -> tuple[list[str], str]:
    """Copy the reviewed recipe, changing only output names and preregistered batch."""
    if width not in WIDTHS or kind not in {"probe", "fit"} or batch_size not in PREREG_BATCHES:
        raise ValueError("unsupported width or run kind")
    old = f"c20261004_w{width}_adamw_s3_f1_t4v4_{kind}"
    new = output_run_name(batch_size, width, kind)
    spec = json.loads((root / f"kaggle_specs/{old}.json").read_text())
    argv = list(spec["argv"])
    if "--plan" not in argv:
        raise ValueError("pinned GPU spec no longer references a sequence plan")
    original_plan = root / argv[argv.index("--plan") + 1]
    plan = json.loads(original_plan.read_text())
    changed_batches = 0
    for step in plan["steps"]:
        step_argv = [item.replace(old, new) for item in step["argv"]]
        for index, token in enumerate(step_argv[:-1]):
            if token == "--batch":
                if step_argv[index + 1] != str(BASE_BATCH):
                    raise ValueError("pinned source batch differs from preregistered B32 base")
                step_argv[index + 1] = str(batch_size)
                changed_batches += 1
        step["argv"] = step_argv
    if changed_batches != 1:
        raise ValueError(f"expected one B32->{batch_size} change, observed {changed_batches}")
    plan["report"] = (
        "research/campaign_2026-10-06/continuation_night/smallcap_pair_v1/"
        f"{new}_steps.json"
    )
    plan_path = plan_directory / f"{new}_plan.json"
    write_json(plan_path, plan)
    argv[argv.index("--plan") + 1] = str(plan_path.relative_to(root))
    return [sys.executable, *argv[1:]], new


def profile_policy(devices: list[dict[str, object]]) -> tuple[list[str], dict[str, object]]:
    """Validate the full allocated inventory and select one training T4."""
    if len(devices) not in (1, 2):
        raise ValueError("expected one or two allocated Tesla T4 devices")
    if any("T4" not in str(device["name"]) for device in devices):
        raise ValueError("allocated accelerator is not a Tesla T4")
    total_used = sum(int(device["used_bytes"]) for device in devices)
    if total_used >= GPU_PRELAUNCH_BYTES:
        raise ValueError("aggregate preexisting GPU use exceeds the launch reserve")
    selected = str(devices[0]["uuid"])
    return [str(device["uuid"]) for device in devices], {
        "selected_training_uuid": selected,
        "all_allocated_devices": devices,
        "aggregate_prelaunch_used_bytes": total_used,
        "aggregate_prelaunch_limit_bytes": GPU_PRELAUNCH_BYTES,
        "allocator_limit_bytes": GPU_ALLOCATOR_BYTES,
        "aggregate_stop_bytes": GPU_AGGREGATE_STOP_BYTES,
        "selected_device_visible_to_training": True,
        "unselected_allocated_devices_visible_to_training": False,
    }


def profiles_ready(profiled_widths: list[int]) -> bool:
    """Open the fit gate only after both preregistered width probes pass."""
    return profiled_widths == list(WIDTHS)


def is_vram_failure(output: Path, error: Exception) -> tuple[bool, str]:
    """Allow the preregistered B8 fallback only for a clear VRAM failure."""
    result_path = output / "result.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    reason = str(result.get("reason", ""))
    if reason == "aggregate_allocated_gpu_stop":
        return True, reason
    log_path = output / "worker.log"
    log = log_path.read_text(errors="replace") if log_path.is_file() else ""
    lower = log.lower()
    markers = (
        "cuda out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
    )
    for marker in markers:
        if marker in lower:
            return True, marker
    return False, f"non_vram_failure:{type(error).__name__}"


def prepare_environment(root: Path, selected_uuid: str, worker_cpu_count: int) -> dict[str, str]:
    """Set one visible GPU, B16-friendly loader bounds, and capped CPU pools."""
    threads = min(WORKER_THREADS, worker_cpu_count)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": selected_uuid,
            "WNN_GPU_MEMORY_BYTES": str(GPU_ALLOCATOR_BYTES),
            "WNN_LOADER_RAM_GB": "1",
            "WNN_LOADER_WORKERS": "0",
            "WNN_THREADS": str(threads),
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
    ):
        environment[name] = str(threads)
    return environment


def run(root: Path, runtime: Path, manifest_path: Path) -> None:
    """Profile both matched widths before starting either full training run."""
    output = Path("/kaggle/working/smallcap_pair_20261006_v1")
    output.mkdir(exist_ok=False)
    run_names = {
        (batch_size, width, kind): output_run_name(batch_size, width, kind)
        for batch_size in PREREG_BATCHES
        for width in WIDTHS
        for kind in ("probe", "fit")
    }
    existing = [
        str(root / "artifacts/runs" / name)
        for name in run_names.values()
        if (root / "artifacts/runs" / name).exists()
    ]
    if existing:
        raise FileExistsError(f"fresh small-cap run identity already exists: {existing}")
    evidence: dict[str, object] = {
        "status": "running",
        "protocol": "serial W256/W384 probes at B16; only a classified VRAM failure permits common B8 reprobes",
        "required_widths": list(WIDTHS),
        "baseline": "new preregistered common B16 baseline, with B8 as the sole frozen memory fallback; never compare to B32 scores",
        "candidate_common_batches": list(PREREG_BATCHES),
        "selected_batch_size": None,
        "changed_from_prior_recipe": ["batch size 32 to 16, with B8 only on a classified VRAM failure", "one visible GPU; widths run serially"],
        "cpu_budget_including_guard": CPU_BUDGET,
        "worker_thread_limit": WORKER_THREADS,
        "allocator_limit_bytes": GPU_ALLOCATOR_BYTES,
        "aggregate_gpu_stop_bytes": GPU_AGGREGATE_STOP_BYTES,
        "profile_passed_widths": [],
        "profiled_batch_pairs": [],
        "fits_started": False,
    }
    write_json(output / "design.json", evidence)
    try:
        verify_inputs(root, manifest_path, output)
        receipt = stage(root, manifest_path, runtime)
        write_json(output / "source_derivation.json", receipt)

        allowed = sorted(os.sched_getaffinity(0))
        if not allowed:
            raise RuntimeError("no CPU affinity is available")
        active_cpu_count = min(CPU_BUDGET, len(allowed))
        guard_cpu = allowed[0]
        worker_cpus = allowed[1:active_cpu_count]
        if not worker_cpus:
            worker_cpus = [guard_cpu]
        os.sched_setaffinity(0, {guard_cpu})
        evidence["active_cpu_count"] = active_cpu_count
        evidence["guard_cpu_affinity"] = [guard_cpu]
        evidence["trainer_cpu_affinity"] = worker_cpus
        evidence["trainer_threads"] = min(WORKER_THREADS, len(worker_cpus))

        devices = gpu_sample()
        allocated_uuids, gpu_policy = profile_policy(devices)
        selected_uuid = str(devices[0]["uuid"])
        environment = prepare_environment(root, selected_uuid, len(worker_cpus))
        evidence["gpu_policy"] = gpu_policy
        evidence["allocated_gpu_uuids_charged"] = allocated_uuids
        evidence["training_visible_gpu_count"] = 1
        write_json(output / "design.json", evidence)

        # Generate every plan before the first resource probe. The checks below
        # enforce one B32->B16 change per width and keep each width's remaining
        # scientific argv equal to its pinned source recipe.
        plan_directory = root / "research/campaign_2026-10-06/continuation_night/smallcap_pair_v1/plans"
        plan_directory.mkdir(parents=True, exist_ok=False)
        plans = {
            (batch_size, width, kind): worker_plan(root, width, kind, batch_size, plan_directory)
            for batch_size in PREREG_BATCHES
            for width in WIDTHS
            for kind in ("probe", "fit")
        }
        write_json(
            output / "generated_plan_manifest.json",
            [
                stable_file_record(plan_directory / f"{name}_plan.json")
                for _key, (_argv, name) in plans.items()
            ],
        )

        selected_batch = None
        for batch_size in PREREG_BATCHES:
            batch_profiles = []
            vram_fallback_allowed = False
            for width in WIDTHS:
                argv, name = plans[(batch_size, width, "probe")]
                phase_output = output / name
                try:
                    result = supervised(
                        argv,
                        root,
                        phase_output,
                        environment,
                        timeout=PROBE_TIMEOUT_SECONDS,
                        rss_limit=RSS_STOP_BYTES,
                        reserve=2 * 1024**3,
                        gpu_uuids=allocated_uuids,
                        gpu_aggregate_stop=GPU_AGGREGATE_STOP_BYTES,
                        gpu_poll_seconds=1.0,
                        worker_affinity=worker_cpus,
                    )
                    counts = check_probe(phase_output / "worker.log")
                    peak = int(result["peak_gpu_aggregate_bytes"])
                    if peak <= 0 or peak >= GPU_AGGREGATE_STOP_BYTES:
                        raise RuntimeError(f"W{width} probe lacks a passing aggregate VRAM profile")
                    record = {
                        "batch_size": batch_size,
                        "width": width,
                        "run_name": name,
                        "status": "passed",
                        "single_batch_update": counts,
                        "resource_result": result,
                        "aggregate_peak_gpu_bytes": peak,
                        "aggregate_stop_bytes": GPU_AGGREGATE_STOP_BYTES,
                    }
                except Exception as error:
                    is_vram, cause = is_vram_failure(phase_output, error)
                    record = {
                        "batch_size": batch_size,
                        "width": width,
                        "run_name": name,
                        "status": "vram_blocked" if is_vram else "failed_closed",
                        "fallback_cause": cause,
                        "error": repr(error),
                    }
                    write_json(output / f"b{batch_size}_w{width}_resource_profile.json", record)
                    batch_profiles.append(record)
                    if batch_size == 16 and is_vram:
                        vram_fallback_allowed = True
                        break
                    raise
                write_json(output / f"b{batch_size}_w{width}_resource_profile.json", record)
                batch_profiles.append(record)

            evidence["profiled_batch_pairs"].append(
                {
                    "batch_size": batch_size,
                    "profiles": batch_profiles,
                    "passed": profiles_ready(
                        [int(item["width"]) for item in batch_profiles if item["status"] == "passed"]
                    ),
                }
            )
            write_json(output / "design.json", evidence)
            if profiles_ready(
                [int(item["width"]) for item in batch_profiles if item["status"] == "passed"]
            ):
                selected_batch = batch_size
                evidence["profile_passed_widths"] = list(WIDTHS)
                break
            if batch_size == 16 and vram_fallback_allowed:
                evidence["fallback"] = {
                    "trigger": "B16 resource profile classified as a VRAM failure",
                    "action": "discard the B16 pair; profile W256 and W384 at common B8 before any fit",
                }
                write_json(output / "design.json", evidence)
                continue
            raise RuntimeError("both width resource profiles must pass at one common batch size")

        if selected_batch is None:
            raise RuntimeError("no preregistered common batch passed both resource profiles")
        evidence["selected_batch_size"] = selected_batch
        evidence["profile_gate"] = f"passed_for_both_widths_at_b{selected_batch}"
        evidence["fits_started"] = True
        write_json(output / "design.json", evidence)

        for width in WIDTHS:
            argv, fit_name = plans[(selected_batch, width, "fit")]
            result = supervised(
                argv,
                root,
                output / fit_name,
                environment,
                timeout=FIT_TIMEOUT_SECONDS,
                rss_limit=RSS_STOP_BYTES,
                reserve=2 * 1024**3,
                gpu_uuids=allocated_uuids,
                gpu_aggregate_stop=GPU_AGGREGATE_STOP_BYTES,
                gpu_poll_seconds=1.0,
                worker_affinity=worker_cpus,
            )
            required = [
                root / f"artifacts/runs/{fit_name}/model_ep3.pt",
                root / f"cache/pred_{fit_name}_valid.npy",
                root / f"cache/pred_{fit_name}_valid.json",
            ]
            if any(not path.is_file() for path in required):
                raise RuntimeError(f"W{width} fit or full validation prediction is missing")
            artifacts = collect_fit_artifacts(root, fit_name)
            artifacts.extend(stable_file_record(path) for path in required[1:])
            write_json(output / f"w{width}_artifact_manifest.json", artifacts)
            write_json(output / f"w{width}_fit_receipt.json", result)

        evidence["status"] = "complete"
        evidence["scientific_claim"] = "none; score both new B16 predictors on identical fold-1 rows"
        write_json(output / "design.json", evidence)
        write_json(
            output / "completion.json",
            {
                "status": "complete",
                "fit_names": [run_names[(selected_batch, width, "fit")] for width in WIDTHS],
                "selected_common_batch_size": selected_batch,
                "score_comparison_to_B32": "prohibited; selected B16/B8 is a newly preregistered baseline",
                "next_step": "root-owned paired score using the frozen complete-fold-1 evaluator",
            },
        )
    except BaseException as error:
        evidence["status"] = "failed_closed"
        evidence["error"] = repr(error)
        evidence["fits_started"] = bool(evidence.get("fits_started", False))
        evidence["failed_at"] = time.time()
        write_json(output / "design.json", evidence)
        if not evidence["fits_started"]:
            write_json(
                output / "fit_gate.json",
                {
                    "status": "closed",
                    "reason": "resource or input profile failed before either full fit",
                    "profile_passed_widths": evidence["profile_passed_widths"],
                    "error": repr(error),
                },
            )
        raise


def main() -> None:
    """Enter from the immutable private Kaggle packet after source verification."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    run(
        Path("/kaggle/working/experiment"),
        Path("/kaggle/working/continuation_runtime"),
        args.manifest,
    )


if __name__ == "__main__":
    main()
