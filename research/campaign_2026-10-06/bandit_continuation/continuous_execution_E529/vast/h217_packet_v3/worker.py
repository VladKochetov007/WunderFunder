from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from pathlib import Path

ARMS = ("tcn", "gru")
PROFILE_SECONDS = 900
FIT_SECONDS = 7200
VALID_EXPORT_SECONDS = 1800
ARM_WORST_CASE_SECONDS = FIT_SECONDS + VALID_EXPORT_SECONDS
AMP_INITIAL_SCALE = 4096.0


def digest(path: Path) -> str:
    """Hash a file incrementally so large prediction exports stay out of RAM."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically persist a bounded worker receipt."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def stop(child: subprocess.Popen[bytes]) -> None:
    """Terminate only the process group created for one phase."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def run_phase(
    argv: list[str],
    name: str,
    timeout_seconds: int,
    project: Path,
    output: Path,
    deadline: float,
) -> dict[str, object]:
    """Run one phase with worker RSS, host RAM, VRAM, and deadline guards."""
    import psutil

    original_argv = list(argv)
    if len(argv) > 1 and argv[1].startswith("scripts/"):
        wrapper = (
            "import sys,runpy,pyarrow;pyarrow.set_cpu_count(1);"
            "pyarrow.set_io_thread_count(1);sys.argv=sys.argv[1:];"
            "sys.path.insert(0,'scripts');runpy.run_path(sys.argv[0],run_name='__main__')"
        )
        argv = [argv[0], "-c", wrapper, *argv[1:]]

    started = time.monotonic()
    phase_deadline = min(deadline, started + timeout_seconds)
    ram_cap = int(os.environ["WNN_WORKER_RAM_BYTES"])
    vram_cap = int(os.environ["WNN_WORKER_VRAM_BYTES"])
    peak_rss = 0
    peak_vram = 0
    child: subprocess.Popen[bytes] | None = None
    error_text: str | None = None
    log_path = output / f"{name}.log"
    try:
        with log_path.open("wb") as log:
            child = subprocess.Popen(
                argv,
                cwd=project,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            while child.poll() is None:
                if time.monotonic() >= phase_deadline:
                    raise TimeoutError(f"{name} exceeded its phase or cohort deadline")
                try:
                    processes = [psutil.Process(child.pid)]
                    processes.extend(processes[0].children(recursive=True))
                    rss = sum(p.memory_info().rss for p in processes if p.is_running())
                except psutil.NoSuchProcess:
                    rss = 0
                gpu = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                vram = sum(int(value) * 1024**2 for value in gpu.stdout.splitlines())
                peak_rss = max(peak_rss, rss)
                peak_vram = max(peak_vram, vram)
                heartbeat = {
                    "phase": name,
                    "elapsed_seconds": time.monotonic() - started,
                    "rss_bytes": rss,
                    "peak_rss_bytes": peak_rss,
                    "vram_bytes": vram,
                    "peak_vram_bytes": peak_vram,
                    "host_available_bytes": psutil.virtual_memory().available,
                }
                write_json(output / "heartbeat.json", heartbeat)
                if (
                    rss > ram_cap
                    or heartbeat["host_available_bytes"] < 2_000_000_000
                    or vram > vram_cap
                ):
                    raise MemoryError(f"resource guard stopped {name}: {heartbeat}")
                time.sleep(2)
    except Exception as error:  # Preserve phase errors so another arm can still run.
        error_text = f"{type(error).__name__}: {error}"
    finally:
        if child is not None and child.poll() is None:
            stop(child)

    exit_code = None if child is None else child.returncode
    if exit_code not in (0, None) and error_text is None:
        error_text = f"phase exited {exit_code}; see {log_path.name}"
    if child is None and error_text is None:
        error_text = "phase process did not start"
    return {
        "phase": name,
        "argv": original_argv,
        "subprocess_argv": argv,
        "exit_code": exit_code,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss,
        "peak_vram_bytes": peak_vram,
        "log_sha256": digest(log_path),
        "error": error_text,
    }


def profile_gate(path: Path, fit: dict[str, object]) -> dict[str, object]:
    """Require exact finite AMP evidence and estimate fit time in outer batches."""
    rows = path.read_text().splitlines()
    config = [json.loads(row) for row in rows if row.startswith("{") and "amp_initial_scale" in row]
    evidence = [
        json.loads(row.split(" ", 1)[1])
        for row in rows
        if row.startswith("PROFILE_UPDATE_EVIDENCE ")
    ]
    timings = [json.loads(row.split(" ", 1)[1]) for row in rows if row.startswith("PROFILE ")]
    if len(config) != 1 or len(evidence) != 1 or len(timings) != 1:
        raise ValueError("profile needs one config, timing, and update-evidence receipt")
    if (
        config[0].get("precision") != "fp16"
        or float(config[0].get("amp_initial_scale", -1)) != AMP_INITIAL_SCALE
    ):
        raise ValueError("profile did not use the declared FP16 AMP scale 4096")
    item = evidence[0]
    required_true = ("loss_finite", "gradient_finite", "counts_consistent", "valid_for_throughput")
    if any(item.get(key) is not True for key in required_true):
        raise ValueError("profile failed strict finite-update evidence")
    if item.get("amp_skipped_updates") != 0 or item.get("optimizer_completed_updates") != item.get(
        "optimizer_attempts"
    ):
        raise ValueError("profile contains a skipped optimizer update")
    if float(item.get("scaler_initial_scale", -1)) != AMP_INITIAL_SCALE:
        raise ValueError("effective GradScaler initial scale differs from the declared 4096")
    step_seconds = float(timings[0]["step_wall"]) / 1000.0
    if not math.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("profile outer-batch time is invalid")
    estimate = math.ceil(11940 / int(fit["batch_size"])) * 4 * step_seconds * 1.5 + 600
    if estimate > FIT_SECONDS:
        raise TimeoutError(f"projected four-epoch fit/export exceeds fit budget: {estimate:.1f}s")
    return {
        "status": "passed",
        "update_evidence": item,
        "outer_batch_step_wall_ms": timings[0]["step_wall"],
        "timing_unit": "one outer DataLoader batch, including its internal chunk loop",
        "projected_fit_and_export_seconds": round(estimate, 1),
    }


def validate_recipe(recipe: dict[str, object], runtime: dict[str, object]) -> None:
    """Fail closed on drift from the reviewed H217 V3 two-arm recipe."""
    profiles = {str(row["arm"]): row for row in recipe["profile_sequence"]}
    fits = {str(row["arm"]): row for row in recipe["fit_branches"]["batch8"]}
    if (
        set(profiles) != set(ARMS)
        or set(fits) != set(ARMS)
        or len(profiles) != len(recipe["profile_sequence"])
        or len(fits) != len(recipe["fit_branches"]["batch8"])
    ):
        raise ValueError("recipe must contain exactly one TCN and one GRU B8 branch")

    def option(argv: list[str], key: str) -> str | None:
        try:
            return argv[argv.index(key) + 1]
        except (ValueError, IndexError):
            return None

    for arm in ARMS:
        for phase, branch in (("profile", profiles[arm]), ("fit", fits[arm])):
            argv = list(branch["argv"])
            expected = {
                "--seed": "7",
                "--batch": "8",
                "--epochs": "4",
                "--checkpoint-epoch": "3",
                "--checkpoint-policy": "fixed",
                "--valid-fold": "0",
                "--train-source": "train+valid",
                "--train-seqs": "10607",
                "--precision": "fp16",
                "--amp-initial-scale": str(int(AMP_INITIAL_SCALE)),
                "--loss": "wmse",
                "--loss-norm": "chunk",
                "--optim": "adamw",
                "--sched": "onecycle",
                "--dropout": "0",
                "--chunk": "2000",
                "--lr": "0.001",
                "--wd": "0.01",
            }
            if any(option(argv, key) != value for key, value in expected.items()):
                raise ValueError(f"{arm} {phase} argv differs from frozen V3 settings")
            if "--raw-only" not in argv or option(argv, "--arch") != (
                "tcn" if arm == "tcn" else "gru"
            ):
                raise ValueError(f"{arm} {phase} architecture or raw-only input differs")
            if option(argv, "--name") != branch.get("run_name"):
                raise ValueError(f"{arm} {phase} run name differs from command")
            expected_architecture = (
                {"--hidden": "128", "--layers": "12", "--tcn-mix": "1"}
                if arm == "tcn"
                else {"--hidden": "256", "--layers": "2", "--cell": "gru"}
            )
            if any(option(argv, key) != value for key, value in expected_architecture.items()):
                raise ValueError(f"{arm} {phase} architecture dimensions differ")
            if option(argv, "--init") != "" or option(argv, "--distill") != "":
                raise ValueError(f"{arm} {phase} must train from scratch without a teacher")
            if phase == "profile" and (
                option(argv, "--profile-warmup") != "3" or option(argv, "--profile-steps") != "20"
            ):
                raise ValueError(f"{arm} profile timing budget differs from the frozen profile")
        export = fits[arm].get("prediction_export", {})
        if (
            export.get("epoch") != 3
            or export.get("split") != "valid"
            or export.get("inference_precision") != "fp32"
        ):
            raise ValueError(f"{arm} export must be valid-only fixed-epoch-3 FP32")
    if int(runtime["cohort_max_runtime_seconds"]) != 21600:
        raise ValueError("cohort deadline differs from the reviewed 21600-second ceiling")
    phase_sum = 2 * int(runtime["profile_timeout_seconds"]) + 2 * (
        int(runtime["fit_timeout_seconds"]) + int(runtime["valid_export_timeout_seconds"])
    )
    reserves = int(runtime["setup_timeout_seconds"]) + int(
        runtime["terminal_packaging_reserve_seconds"]
    )
    if phase_sum + reserves > int(runtime["cohort_max_runtime_seconds"]):
        raise ValueError("phase budgets plus setup/package reserves exceed the worker lifetime")


def validate_prediction(
    project: Path, export: dict[str, object], checkpoint: Path
) -> dict[str, object]:
    """Verify fixed-epoch FP32 validation output with a bounded memmap scan."""
    import numpy as np

    cache = project / "cache"
    name = str(export["output_name"])
    prediction = cache / f"pred_{name}_valid.npy"
    sidecar = prediction.with_suffix(".json")
    if not prediction.is_file() or not sidecar.is_file():
        raise FileNotFoundError("fixed-epoch valid prediction or sidecar is missing")
    metadata = json.loads(sidecar.read_text())
    expected_checkpoint_hash = digest(checkpoint)
    if metadata.get("epoch") != 3 or metadata.get("inference_precision") != "fp32":
        raise ValueError("prediction metadata does not bind fixed epoch 3 FP32 inference")
    if metadata.get("checkpoint_sha256") != expected_checkpoint_hash:
        raise ValueError("prediction sidecar checkpoint hash differs from delivered checkpoint")
    expected_groups = np.load(cache / "valid_targets.npz")["row_group"]
    actual_groups = np.asarray(metadata.get("row_groups", []), dtype=np.int64)
    if not np.array_equal(actual_groups, expected_groups):
        raise ValueError("prediction row groups differ from the valid target membership")
    values = np.load(prediction, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(expected_groups), 20_000, 2)
    if values.shape != expected_shape or values.dtype != np.float32:
        raise ValueError(f"prediction shape/dtype mismatch: {values.shape} {values.dtype}")
    for start in range(0, len(values), 16):
        if not np.isfinite(values[start : start + 16]).all():
            raise ValueError("prediction contains a nonfinite value")
    prediction_hash = digest(prediction)
    if metadata.get("prediction_sha256") != prediction_hash:
        raise ValueError("prediction sidecar hash mismatch")
    return {
        "path": str(prediction.relative_to(project)),
        "sidecar": str(sidecar.relative_to(project)),
        "prediction_sha256": prediction_hash,
        "checkpoint_sha256": expected_checkpoint_hash,
        "row_group_count": len(expected_groups),
        "row_groups_sha256": hashlib.sha256(actual_groups.astype("<i8").tobytes()).hexdigest(),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite": True,
        "inference_precision": "fp32",
        "epoch": 3,
    }


PhaseRunner = Callable[[list[str], str, int, Path, Path, float], dict[str, object]]
ProfileGate = Callable[[Path, dict[str, object]], dict[str, object]]


def execute_arms(
    recipe: dict[str, object],
    project: Path,
    output: Path,
    deadline: float,
    result: dict[str, object],
    phase_runner: PhaseRunner,
    gate_runner: ProfileGate,
) -> None:
    """Run both profiles, then each eligible fit/export without cross-arm aborts."""
    profiles = {str(row["arm"]): row for row in recipe["profile_sequence"]}
    fits = {str(row["arm"]): row for row in recipe["fit_branches"]["batch8"]}
    arms: dict[str, dict[str, object]] = {
        arm: {
            "profile_status": "pending",
            "fit_status": "blocked_profile",
            "export_status": "not_run",
        }
        for arm in ARMS
    }
    result["arms"] = arms

    for arm in ARMS:
        profile = profiles.get(arm)
        fit = fits.get(arm)
        if profile is None or fit is None:
            arms[arm]["profile_status"] = "invalid_recipe"
            arms[arm]["error"] = "profile or fit branch is missing"
            write_json(output / "result.json", result)
            continue
        try:
            phase = phase_runner(
                [sys.executable, *profile["argv"]],
                f"profile_{arm}",
                PROFILE_SECONDS,
                project,
                output,
                deadline,
            )
            arms[arm]["profile_phase"] = phase
            if phase.get("exit_code") != 0 or phase.get("error"):
                arms[arm]["profile_status"] = "failed"
                arms[arm]["error"] = phase.get("error") or f"profile exit {phase.get('exit_code')}"
            else:
                gate = gate_runner(output / f"profile_{arm}.log", fit)
                arms[arm]["profile_gate"] = gate
                arms[arm]["profile_status"] = "passed"
                arms[arm]["fit_status"] = "pending"
        except Exception as error:
            arms[arm]["profile_status"] = "failed"
            arms[arm]["error"] = f"{type(error).__name__}: {error}"
        write_json(output / "result.json", result)

    eligible = [arm for arm in ARMS if arms[arm]["profile_status"] == "passed"]
    for index, arm in enumerate(eligible):
        fit = fits[arm]
        remaining = max(0.0, deadline - time.monotonic())
        reserve = ARM_WORST_CASE_SECONDS * (len(eligible) - index - 1)
        if remaining < ARM_WORST_CASE_SECONDS + reserve:
            arms[arm]["fit_status"] = "deferred_cohort_time_reserve"
            arms[arm]["error"] = "insufficient deadline to preserve the independent next-arm budget"
            write_json(output / "result.json", result)
            continue
        try:
            fit_phase = phase_runner(
                [sys.executable, *fit["argv"]],
                f"fit_{arm}",
                FIT_SECONDS,
                project,
                output,
                deadline,
            )
            arms[arm]["fit_phase"] = fit_phase
            if fit_phase.get("exit_code") != 0 or fit_phase.get("error"):
                arms[arm]["fit_status"] = "failed"
                arms[arm]["error"] = (
                    fit_phase.get("error") or f"fit exit {fit_phase.get('exit_code')}"
                )
                write_json(output / "result.json", result)
                continue
            checkpoint = project / str(fit["output_path"]) / "model_ep3.pt"
            if not checkpoint.is_file():
                arms[arm]["fit_status"] = "failed_missing_checkpoint"
                arms[arm]["error"] = (
                    "fit returned success without the declared fixed epoch-3 checkpoint"
                )
                write_json(output / "result.json", result)
                continue
            arms[arm]["checkpoint_sha256"] = digest(checkpoint)
            arms[arm]["fit_status"] = "passed"
            write_json(output / "result.json", result)
            export = fit["prediction_export"]
            export_argv = [
                sys.executable,
                "scripts/predict_run.py",
                str(fit["run_name"]),
                "valid",
                "--epoch",
                "3",
                "--output-name",
                str(export["output_name"]),
                "--inference-precision",
                "fp32",
            ]
            export_phase = phase_runner(
                export_argv,
                f"export_{arm}_valid",
                VALID_EXPORT_SECONDS,
                project,
                output,
                deadline,
            )
            arms[arm]["export_phase"] = export_phase
            if export_phase.get("exit_code") != 0 or export_phase.get("error"):
                arms[arm]["export_status"] = "failed"
                arms[arm]["error"] = (
                    export_phase.get("error") or f"export exit {export_phase.get('exit_code')}"
                )
                write_json(output / "result.json", result)
                continue
            arms[arm]["prediction"] = validate_prediction(project, export, checkpoint)
            arms[arm]["export_status"] = "passed"
            arms[arm]["delivery_status"] = "delivered_ungraded"
        except Exception as error:
            arms[arm]["error"] = f"{type(error).__name__}: {error}"
            if arms[arm]["fit_status"] == "pending":
                arms[arm]["fit_status"] = "failed"
            elif arms[arm]["fit_status"] == "passed":
                arms[arm]["export_status"] = "failed"
        write_json(output / "result.json", result)


def main() -> int:
    """Run the admitted H217 V3 cohort and retain per-arm terminal evidence."""
    packet = Path(__file__).resolve().parent
    project = packet / "project"
    output = Path(os.environ["WNN_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    deadline: float | None = None
    result: dict[str, object] = {
        "status": "starting",
        "attempt_id": os.environ.get("WNN_ATTEMPT_ID"),
        "packet_sha256": os.environ.get("WNN_PACKET_SHA256"),
        "scientific_score": None,
        "grade_performed": False,
    }
    try:
        runtime = json.loads((packet / "runtime_resources.json").read_text())
        for environment, field in {
            "WNN_THREADS": "cpu_threads",
            "WNN_LOADER_WORKERS": "loader_workers",
            "WNN_WORKER_RAM_BYTES": "worker_memory_max_bytes",
            "WNN_WORKER_VRAM_BYTES": "observed_total_vram_max_bytes",
            "WNN_GPU_MEMORY_BYTES": "cuda_allocator_max_bytes",
        }.items():
            if int(os.environ[environment]) != runtime[field]:
                raise ValueError(f"runtime resource mismatch: {environment}")
        if int(os.environ["WNN_MAX_RUNTIME_SECONDS"]) != int(runtime["cohort_max_runtime_seconds"]):
            raise ValueError("worker runtime differs from the frozen 21600-second cohort cap")
        deadline = (
            time.monotonic()
            + int(runtime["cohort_max_runtime_seconds"])
            - int(runtime["setup_timeout_seconds"])
            - int(runtime["terminal_packaging_reserve_seconds"])
        )
        write_json(output / "runtime_resources.json", runtime)
        recipe = json.loads((packet / "recipe.json").read_text())
        validate_recipe(recipe, runtime)
        os.environ["PYTHONPATH"] = str(project / "src")
        os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(
            project / "research/campaign_2026-10-04/portable_inputs.json"
        )
        inputs = json.loads(os.environ["WNN_INPUT_MANIFEST"])
        for item in inputs:
            source = Path(os.environ["WNN_INPUT_DIR"]) / item["path"]
            destination = project / item["path"]
            if not source.is_file() or source.stat().st_size != item["size_bytes"]:
                raise ValueError(f"missing or truncated staged input: {item['path']}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source)
        import torch

        if torch.__version__ != "2.14.0+cu130" or not torch.cuda.is_available():
            raise RuntimeError(f"unexpected Torch/device environment: {torch.__version__}")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("the packet declares exactly one GPU")
        if time.monotonic() >= deadline:
            raise TimeoutError("cohort deadline expired during input preflight")
        result["environment"] = {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "input_manifest": inputs,
        }
        write_json(output / "environment.json", result["environment"])
        result["runtime_resources"] = runtime
        result["status"] = "running_profiles"
        write_json(output / "result.json", result)
        if deadline is None:
            raise RuntimeError("cohort deadline was not established")
        execute_arms(recipe, project, output, deadline, result, run_phase, profile_gate)
        arms = result["arms"]
        delivered = [
            arm for arm in arms.values() if arm.get("delivery_status") == "delivered_ungraded"
        ]
        if len(delivered) == len(ARMS):
            result["status"] = "delivered_ungraded"
        elif delivered:
            result["status"] = "partial_delivery_ungraded"
        else:
            result["status"] = "failed"
        result["worker_finished_at_unix_seconds"] = time.time()
    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        write_json(output / "result.json", result)
        artifact_path = output / "artifacts.tar"
        with tarfile.open(artifact_path, "w") as archive:
            for path in sorted(output.iterdir()):
                if path.is_file() and path.name not in {
                    "artifacts.tar",
                    "artifact_manifest.json",
                    "worker.log",
                }:
                    archive.add(path, arcname=f"diagnostics/{path.name}", recursive=False)
            for directory in (project / "artifacts/runs", project / "cache"):
                if directory.exists():
                    for path in sorted(directory.rglob("*")):
                        if (
                            path.is_file()
                            and not path.is_symlink()
                            and (directory.name == "runs" or path.name.startswith("pred_"))
                        ):
                            archive.add(
                                path, arcname=str(path.relative_to(project)), recursive=False
                            )
    return 0 if result["status"] in {"delivered_ungraded", "partial_delivery_ungraded"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
