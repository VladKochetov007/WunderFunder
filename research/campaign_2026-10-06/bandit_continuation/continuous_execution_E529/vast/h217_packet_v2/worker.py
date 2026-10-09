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
from pathlib import Path


def digest(path: Path) -> str:
    """Hash an artifact without materializing it in RAM."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Persist bounded worker state atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def stop(child: subprocess.Popen[bytes]) -> None:
    """Terminate only this worker's child process group."""
    if child.poll() is not None:
        return
    os.killpg(child.pid, signal.SIGTERM)
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
    """Execute one recipe phase with RAM, VRAM and wall-time guards."""
    import psutil

    if len(argv) > 1 and argv[1].startswith("scripts/"):
        # Limit Arrow in the child before loader forks; scientific argv is unchanged.
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
    log_path = output / f"{name}.log"
    with log_path.open("wb") as log:
        child = subprocess.Popen(argv, cwd=project, stdout=log, stderr=log, start_new_session=True)
        try:
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
                vram = sum(int(x) * 1024**2 for x in gpu.stdout.splitlines())
                peak_rss = max(peak_rss, rss)
                peak_vram = max(peak_vram, vram)
                available = psutil.virtual_memory().available
                heartbeat = {
                    "phase": name,
                    "elapsed_seconds": time.monotonic() - started,
                    "rss_bytes": rss,
                    "peak_rss_bytes": peak_rss,
                    "vram_bytes": vram,
                    "peak_vram_bytes": peak_vram,
                    "host_available_bytes": available,
                }
                write_json(output / "heartbeat.json", heartbeat)
                if rss > ram_cap or available < 2_000_000_000 or vram > vram_cap:
                    raise MemoryError(f"resource guard stopped {name}: {heartbeat}")
                time.sleep(2)
            if child.returncode:
                raise RuntimeError(f"{name} exited {child.returncode}; see {log_path.name}")
        finally:
            stop(child)
    return {
        "phase": name,
        "argv": argv,
        "exit_code": child.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss,
        "peak_vram_bytes": peak_vram,
        "log_sha256": digest(log_path),
    }


def profile_gate(path: Path, fit: dict[str, object], remaining: float) -> float:
    """Require complete finite updates and time for the declared full fit/export."""
    rows = path.read_text().splitlines()
    evidence = [
        json.loads(r.split(" ", 1)[1]) for r in rows if r.startswith("PROFILE_UPDATE_EVIDENCE ")
    ]
    timings = [json.loads(r.split(" ", 1)[1]) for r in rows if r.startswith("PROFILE ")]
    if len(evidence) != 1 or not evidence[0].get("valid_for_throughput") or len(timings) != 1:
        raise ValueError("profile lacks one finite completed-update timing receipt")
    step_seconds = float(timings[0]["step_wall"]) / 1000
    if not math.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("invalid observed profile step time")
    # Declared training membership is 10,535 train + 1,405 valid; B8, four epochs.
    estimate = math.ceil(11940 / int(fit["batch_size"])) * 4 * step_seconds * 1.5 + 600
    if estimate > 7200 or estimate + 3600 > remaining:
        raise TimeoutError(f"full fit plus two exports does not fit allocation: {estimate:.1f}s")
    return estimate


def main() -> int:
    """Own the H217 profile→fit→fixed-epoch export cohort and failure artifacts."""
    packet = Path(__file__).resolve().parent
    project = packet / "project"
    output = Path(os.environ["WNN_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + int(os.environ["WNN_MAX_RUNTIME_SECONDS"]) - 90
    result: dict[str, object] = {"status": "starting", "phases": [], "scientific_score": None}
    try:
        runtime_resources = json.loads((packet / "runtime_resources.json").read_text())
        for variable, field in {
            "WNN_THREADS": "cpu_threads",
            "WNN_LOADER_WORKERS": "loader_workers",
            "WNN_WORKER_RAM_BYTES": "worker_memory_max_bytes",
            "WNN_WORKER_VRAM_BYTES": "observed_total_vram_max_bytes",
            "WNN_GPU_MEMORY_BYTES": "cuda_allocator_max_bytes",
        }.items():
            if int(os.environ[variable]) != runtime_resources[field]:
                raise ValueError(f"runtime resource drift: {variable}")
        result["runtime_resources"] = runtime_resources
        write_json(output / "runtime_resources.json", runtime_resources)
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
            # Controller verified the uploaded bytes before atomic promotion.
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source)
        import torch

        if torch.__version__ != "2.14.0+cu130" or not torch.cuda.is_available():
            raise RuntimeError(f"unexpected Torch/device environment: {torch.__version__}")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("this packet admits exactly one GPU")
        write_json(
            output / "environment.json",
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "python": sys.version,
                "input_manifest": inputs,
            },
        )
        recipe = json.loads((packet / "recipe.json").read_text())
        phases = result["phases"]
        if not isinstance(phases, list):
            raise TypeError("phase ledger malformed")
        for index, profile in enumerate(recipe["profile_sequence"]):
            name = f"profile_{index}"
            phases.append(
                run_phase([sys.executable, *profile["argv"]], name, 900, project, output, deadline)
            )
            write_json(output / "result.json", result)
        for index, fit in enumerate(recipe["fit_branches"]["batch8"]):
            estimate = profile_gate(
                output / f"profile_{index}.log", fit, deadline - time.monotonic()
            )
            fit_name = f"fit_{index}"
            phases.append(
                run_phase([sys.executable, *fit["argv"]], fit_name, 7200, project, output, deadline)
            )
            result[f"fit_{index}_projected_seconds"] = estimate
            checkpoint = project / fit["output_path"] / "model_ep3.pt"
            if not checkpoint.is_file():
                raise ValueError("fixed epoch-3 checkpoint was not delivered")
            for split in ("valid", "train"):
                argv = [
                    sys.executable,
                    "scripts/predict_run.py",
                    fit["run_name"],
                    split,
                    "--epoch",
                    "3",
                ]
                phases.append(
                    run_phase(argv, f"export_{index}_{split}", 1800, project, output, deadline)
                )
            write_json(output / "result.json", result)
        result["status"] = "delivered_ungraded"
    except Exception as error:
        # Failure is preserved for autonomous diagnosis; no successful fit is fabricated.
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        write_json(output / "result.json", result)
        with tarfile.open(output / "artifacts.tar", "w") as archive:
            for path in sorted(output.iterdir()):
                if path.is_file() and path.name not in {
                    "artifacts.tar",
                    "artifact_manifest.json",
                    "worker.log",
                }:
                    archive.add(path, arcname="diagnostics/" + path.name, recursive=False)
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
        # The collector manifest is written by run.sh after stdout/stderr have closed.
    return 0 if result["status"] == "delivered_ungraded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
