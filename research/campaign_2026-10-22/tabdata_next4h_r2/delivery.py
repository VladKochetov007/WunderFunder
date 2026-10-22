"""Finite per-cell delivery; the systemd cgroup owns every child process."""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

ROOT = Path("/data/epanov_s/tabdata")
PACKET = ROOT / "runtime/next4h-20261022-r2"
STATE = ROOT / "state/next4h-20261022-r2"
PYTHON = ROOT / ".venv/bin/python"
UNIT = "tabdata-next4h-20261022-r2.service"


def read(path: Path) -> Any:
    """Read a bounded local receipt."""
    return json.loads(path.read_text())


def write(path: Path, value: Any) -> None:
    """Replace one owned JSON receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path: Path) -> str:
    """Hash an artifact in bounded blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def environment(job: dict[str, Any] | None = None) -> dict[str, str]:
    """Isolate devices and import the exact committed experiment source."""
    source = PACKET / "sources" / ("numeric" if job is None else job["family"])
    result = dict(
        os.environ,
        PYTHONPATH=str(source / "src"),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        PYTHONHASHSEED="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        CUDA_VISIBLE_DEVICES="",
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        WNN_PROJECT_ROOT=str(STATE / "nll_project"),
    )
    if job is not None:
        result["CUDA_VISIBLE_DEVICES"] = job["gpu_uuid"]
    return result


def phase(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Path,
    deadline: float,
    cpus: list[int] | None = None,
) -> dict[str, Any]:
    """Enforce a common absolute cutoff on an entire owned process group."""
    started = time.time()
    if started >= deadline:
        raise TimeoutError("Common phase deadline passed before admission")
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = command if cpus is None else ["taskset", "-c", ",".join(map(str, cpus)), *command]
    with log.open("x") as stream:
        child = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = child.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                child.wait(timeout=8)
            # A parent can exit before a signal-ignoring descendant. Escalate the
            # owned group even when waiting for the direct child already returned.
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=3)
            settled = time.monotonic() + 3
            while time.monotonic() < settled:
                active = []
                for status in Path("/proc").glob("[0-9]*/stat"):
                    try:
                        fields = status.read_text().rsplit(")", 1)[1].split()
                    except (FileNotFoundError, ProcessLookupError, PermissionError):
                        continue
                    if int(fields[2]) == child.pid and fields[0] != "Z":
                        active.append(status.parent.name)
                if not active:
                    break
                time.sleep(0.05)
            raise TimeoutError(f"Owned group stopped at cutoff: {log.name}") from None
    result = {
        "argv": argv,
        "cwd": str(cwd),
        "exit_code": code,
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        "log_sha256": sha(log),
    }
    write(log.with_suffix(".receipt.json"), result)
    if code:
        raise RuntimeError(f"Phase exit {code}: {log}")
    return result


def numeric_command(job: dict[str, Any], stage: str) -> list[str]:
    """Use the R2 runner for prepare/profile/fit/export and unchanged grading."""
    argv = [
        str(PYTHON),
        "-u",
        "scripts/run_numeric_experiment.py",
        stage,
        "--revision",
        "r2_e3",
        "--fold",
        str(job["fold"]),
        "--arm",
        job["arm"],
        "--data-root",
        str(ROOT / "data"),
        "--split-manifest",
        str(ROOT / "splits" / f"fold{job['fold']}.json"),
        "--output-root",
        str(STATE / "numeric"),
    ]
    if stage == "fit":
        argv += [
            "--profile-receipt",
            str(
                STATE / "numeric/profiles_r2_e3" / f"f{job['fold']}_{job['arm']}" / "approved.json"
            ),
        ]
    return argv


def nll_command(job: dict[str, Any], stage: str) -> list[str]:
    """Keep the cold-label recipe fixed; profiles use separate run names."""
    split = ROOT / "splits" / f"fold{job['fold']}.json"
    if stage == "evaluate":
        return [
            str(PYTHON),
            "-u",
            "scripts/predict_run.py",
            job["id"],
            "valid",
            "--epoch",
            "2",
            "--campaign-split-manifest",
            str(split),
            "--campaign-full-valid-feature-only",
            "--device",
            "cuda",
            "--inference-precision",
            "fp32",
            "--loader-workers",
            "0",
        ]
    argv = [
        str(PYTHON),
        "-u",
        "scripts/train_rnn.py",
        "--name",
        job["id"] + ("_profile" if stage == "profile" else ""),
        "--split-manifest",
        str(split),
        "--valid-fold",
        str(job["fold"]),
        "--eval-source",
        "vfold",
        "--train-source",
        "train+valid",
        "--seed",
        str(job["seed"]),
        "--train-seqs",
        "10607",
        "--arch",
        "gru",
        "--cell",
        "gru",
        "--enc",
        "linear",
        "--hidden",
        "256",
        "--layers",
        "2",
        "--head-layers",
        "1",
        "--batch",
        "32",
        "--chunk",
        "2000",
        "--epochs",
        "3",
        "--checkpoint-policy",
        "fixed",
        "--checkpoint-epoch",
        "2",
        "--no-eval-during-fit",
        "--lr",
        ".001",
        "--wd",
        ".01",
        "--beta2",
        ".999",
        "--sched",
        "onecycle",
        "--clip",
        "1.0",
        "--precision",
        "fp32",
        "--mean-variance-head",
        "1",
        "--mean-variance-objective",
        job["arm"],
        "--grad-stats",
        "--loader-workers",
        "0",
    ]
    if job["arm"] == "nll":
        argv += ["--mean-variance-target-source", "labels"]
    if stage == "profile":
        argv += ["--profile-warmup", "2", "--profile-steps", "4"]
        argv = [str(PYTHON), "-u", str(PACKET / "nll_profile.py"), *argv[2:]]
    return argv


def run_job_phase(job: dict[str, Any], stage: str, deadline: float) -> dict[str, Any]:
    """Run one frozen stage and retain its command, timing and process outcome."""
    builder = numeric_command if job["family"] == "numeric" else nll_command
    return phase(
        builder(job, stage),
        cwd=PACKET / "sources" / job["family"],
        env=environment(job),
        log=STATE / "logs" / job["id"] / f"{stage}.log",
        deadline=deadline,
        cpus=job["cpus"],
    )


def approve_profile(job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Inspect exact-config finite evidence and headroom before science admission."""
    if job["family"] == "numeric":
        folder = STATE / "numeric/profiles_r2_e3" / f"f{job['fold']}_{job['arm']}"
        receipt = read(folder / "profile_receipt.json")
        if (
            receipt["status"] != "profile_complete"
            or receipt["updates"] != 4
            or receipt["errors"]
            or receipt["error_count"]
            or receipt["scientific_checkpoint_written"]
        ):
            raise ValueError("Numeric profile failed exact four-update contract")
        for memory in receipt["memory_by_rank"]:
            if memory["peak_reserved_bytes"] >= 0.9 * memory["device_total_memory_bytes"]:
                raise ValueError("Less than ten percent per-process VRAM headroom")
        approved = dict(receipt, approved=True)
        # Canonical fit admission performs the full current identity validation again.
        write(folder / "approved.json", approved)
        result["forecast_fit_seconds_raw"] = 88638 * receipt["elapsed_seconds"] / 4
        result["profile_receipt_sha256"] = sha(folder / "profile_receipt.json")
        result["memory"] = {
            "peak_reserved_bytes": receipt["peak_reserved_bytes"],
            "total_bytes": receipt["memory_by_rank"][0]["device_total_memory_bytes"],
        }
    else:
        lines = (STATE / "logs" / job["id"] / "profile.log").read_text().splitlines()
        timings = [json.loads(s.split(" ", 1)[1]) for s in lines if s.startswith("PROFILE ")]
        evidence = [
            json.loads(s.split(" ", 1)[1])
            for s in lines
            if s.startswith("PROFILE_UPDATE_EVIDENCE ")
        ]
        if not timings or not evidence or evidence[-1].get("valid_for_throughput") is not True:
            raise ValueError("NLL fit-path profile lacks valid finite update evidence")
        if not math.isfinite(timings[-1]["step_wall"]) or timings[-1]["step_wall"] <= 0:
            raise ValueError("Invalid NLL profile timing")
        result["timing"] = timings[-1]
        result["update_evidence"] = evidence[-1]
        resources = [
            json.loads(s.split(" ", 1)[1]) for s in lines if s.startswith("RESOURCE_PROFILE ")
        ]
        if not resources or resources[-1]["gpu_uuid"] != job["gpu_uuid"]:
            raise ValueError("Missing source-bound NLL live allocator peak")
        memory = resources[-1]
        if (
            not 0
            < memory["peak_allocated_bytes"]
            <= memory["peak_reserved_bytes"]
            < 0.9 * memory["total_bytes"]
        ):
            raise ValueError("NLL allocator peak or headroom failed")
        result["memory"] = memory
        result["forecast_fit_seconds_raw"] = 3 * 374 * timings[-1]["step_wall"] / 1000
    observed = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            job["gpu_uuid"],
            "--query-gpu=uuid,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    fields = [v.strip() for v in observed.strip().split(",")]
    if fields[0] != job["gpu_uuid"] or float(fields[1]) < 0.1 * float(fields[2]):
        raise ValueError("Wrong GPU or less than ten percent actual device headroom")
    result["gpu_observation"] = observed.strip()
    result["forecast_is_projection"] = True
    write(STATE / "profiles" / (job["id"] + ".json"), result)
    return result
