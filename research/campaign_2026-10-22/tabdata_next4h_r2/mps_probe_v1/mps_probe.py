"""Bounded direct-versus-private-MPS profiles for a matched cold GRU pair."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent import futures
from pathlib import Path
from typing import Any

ROOT = Path("/data/epanov_s/tabdata")
BASE_PACKET = ROOT / "runtime/next4h-20261022-r2"
BASE_STATE = ROOT / "state/next4h-20261022-r2"
STATE = ROOT / "state/tabdata-mps-probe-20261022"
UNIT = "tabdata-mps-probe-20261022.service"
MPS_CONTROL = Path("/usr/bin/nvidia-cuda-mps-control")
sys.path.insert(0, str(BASE_PACKET))

import delivery


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def cgroup(pid: int) -> str:
    return Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")


def owned_cuda_clients() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader",
        ],
        text=True,
        timeout=10,
    )
    owned = []
    for line in output.splitlines():
        try:
            pid = int(line.split(",", 1)[0].strip())
            group = cgroup(pid)
        except (ValueError, OSError):
            continue
        if UNIT in group:
            owned.append({"pid": pid, "row": line.strip(), "cgroup": group})
    return owned


def device_map(plan: dict[str, Any]) -> dict[int, str]:
    devices: dict[int, str] = {}
    for job in plan["jobs"]:
        index, uuid = int(job["gpu_index"]), str(job["gpu_uuid"])
        if index in devices and devices[index] != uuid:
            raise ValueError(f"base plan disagrees on GPU {index} UUID")
        devices[index] = uuid
    if set(devices) != set(range(8)):
        raise ValueError("base plan does not identify all eight allocated GPUs")
    return devices


def verify_base_files(plan: dict[str, Any]) -> None:
    for filename, expected in plan.get("files", {}).items():
        if delivery.sha(Path(filename)) != expected:
            raise ValueError(f"base cohort source/input hash mismatch: {filename}")


def make_jobs(gpu_uuid: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    direct, mps = [], []
    for arm_index, arm in enumerate(("wmse", "nll")):
        common = {
            "family": "nll",
            "fold": 0,
            "seed": 1601,
            "arm": arm,
            "gpu_index": 0,
            "gpu_uuid": gpu_uuid,
            "cpus": list(range(112 + arm_index * 4, 116 + arm_index * 4)),
        }
        direct.append({**common, "id": f"mpsprobe_direct_f0_s1601_{arm}"})
        mps.append({**common, "id": f"mpsprobe_mps_f0_s1601_{arm}"})
    return direct, mps


def configure_delivery(pipe: Path, logdir: Path) -> None:
    delivery.PACKET = BASE_PACKET
    delivery.STATE = STATE
    delivery.UNIT = UNIT
    original_environment = delivery.environment

    def probe_environment(job: dict[str, Any] | None = None) -> dict[str, str]:
        env = original_environment(job)
        env["WNN_PROJECT_ROOT"] = str(BASE_STATE / "nll_project")
        for key in (
            "CUDA_MPS_PIPE_DIRECTORY",
            "CUDA_MPS_LOG_DIRECTORY",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT",
        ):
            env.pop(key, None)
        if pipe.exists():
            env["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe)
            env["CUDA_MPS_LOG_DIRECTORY"] = str(logdir)
        return env

    delivery.environment = probe_environment


def profile(job: dict[str, Any], deadline: float) -> dict[str, Any]:
    phase = delivery.run_job_phase(job, "profile", deadline)
    receipt = delivery.approve_profile(job, phase)
    argv = phase["argv"]
    expected = {
        "--seed": str(job["seed"]),
        "--precision": "fp32",
        "--epochs": "3",
        "--checkpoint-epoch": "2",
        "--arch": "gru",
        "--hidden": "256",
        "--layers": "2",
        "--batch": "32",
        "--chunk": "2000",
        "--mean-variance-head": "1",
        "--mean-variance-objective": job["arm"],
        "--loss": "wmse",
        "--loss-norm": "chunk",
        "--lr": ".001",
        "--wd": ".01",
        "--sched": "onecycle",
        "--clip": "1.0",
        "--aux-struct": "0.3",
        "--aux-next": "0.3",
        "--profile-warmup": "2",
        "--profile-steps": "4",
    }
    for flag, value in expected.items():
        if flag not in argv or argv[argv.index(flag) + 1] != value:
            raise ValueError(f"profile command drift for {job['id']}: {flag}")
    if "--grad-stats" not in argv or "--no-eval-during-fit" not in argv:
        raise ValueError(
            "profile requires canonical gradient evidence and no fit evaluation"
        )
    if job["arm"] == "nll":
        target = "--mean-variance-target-source"
        if target not in argv or argv[argv.index(target) + 1] != "labels":
            raise ValueError("NLL profile target source drifted from cold labels")
    elif "--mean-variance-target-source" in argv:
        raise ValueError("WMSE control must retain the canonical non-NLL target route")
    evidence = receipt["update_evidence"]
    attempts = int(evidence["optimizer_attempts"])
    if not (
        evidence.get("valid_for_throughput") is True
        and evidence.get("counts_consistent") is True
        and attempts > 0
        and evidence.get("optimizer_completed_updates") == attempts
        and evidence.get("amp_skipped_updates") == 0
        and evidence.get("loss_checks") == attempts
        and evidence.get("loss_finite") is True
        and evidence.get("grad_scaler_enabled") is False
        and evidence.get("gradient_finite") is True
        and evidence.get("gradient_stats_steps") == attempts
        and evidence.get("gradient_stats_nonfinite_steps") == 0
    ):
        raise ValueError(
            f"profile did not pass strict finite loss/gradient checks: {job['id']}"
        )
    receipt.pop("forecast_fit_seconds_raw", None)
    receipt["profile_only_no_fit_forecast"] = True
    write(STATE / "profiles" / f"{job['id']}.json", receipt)
    return receipt


def run_pair(jobs: list[dict[str, Any]], deadline: float) -> dict[str, Any]:
    results, errors = {}, {}
    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        running = {pool.submit(profile, job, deadline): job for job in jobs}
        for future in futures.as_completed(running):
            job = running[future]
            try:
                results[job["arm"]] = future.result()
            except Exception as error:  # noqa: BLE001 - preserve either arm's bounded profile failure
                errors[job["arm"]] = repr(error)
    return {
        "results": results,
        "errors": errors,
        "complete": len(results) == 2 and not errors,
    }


def gpu_snapshot(uuid: str) -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            uuid,
            "--query-gpu=uuid,utilization.gpu,memory.used,memory.free,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()


def mps_environment(pipe: Path, logdir: Path, visible: str) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
    env.pop("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT", None)
    env.update(
        {
            "CUDA_MPS_PIPE_DIRECTORY": str(pipe),
            "CUDA_MPS_LOG_DIRECTORY": str(logdir),
            "CUDA_VISIBLE_DEVICES": visible,
        }
    )
    return env


def existing_mps_processes() -> list[dict[str, Any]]:
    """Record MPS processes without connecting to, reconfiguring or signaling them."""
    result = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            pid = int(path.parent.name)
            args = (
                path.read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
            )
            if "nvidia-cuda-mps-control" in args or "nvidia-cuda-mps-server" in args:
                result.append({"pid": pid, "cmdline": args, "cgroup": cgroup(pid)})
        except (OSError, ValueError):
            continue
    return result


def control(
    pipe: Path, logdir: Path, visible: str, command: str, timeout: float = 4
) -> str:
    result = subprocess.run(
        [str(MPS_CONTROL)],
        input=command + "\n",
        text=True,
        capture_output=True,
        env=mps_environment(pipe, logdir, visible),
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"MPS control command {command!r} exited {result.returncode}: {result.stderr[-1000:]}"
        )
    return result.stdout.strip()


def mps_server_receipt(pipe: Path, logdir: Path, visible: str) -> dict[str, Any]:
    listing = control(pipe, logdir, visible, "get_server_list")
    servers = []
    for token in listing.split():
        if not token.isdecimal():
            continue
        pid = int(token)
        try:
            group = cgroup(pid)
            cmdline = (
                Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            )
        except OSError:
            continue
        servers.append({"pid": pid, "cgroup": group, "cmdline": cmdline})
    if any(UNIT not in server["cgroup"] for server in servers):
        raise RuntimeError(
            "private MPS control pipe exposed a server outside the probe cgroup"
        )
    return {"server_list": listing, "servers": servers, "observed_unix": time.time()}


def start_mps(
    pipe: Path, logdir: Path, visible: str, deadline: float
) -> subprocess.Popen[bytes]:
    if not MPS_CONTROL.is_file():
        raise FileNotFoundError(f"MPS control binary absent: {MPS_CONTROL}")
    if pipe.exists() or logdir.exists():
        raise FileExistsError(
            "refusing to reuse an existing private MPS pipe or log directory"
        )
    pipe.mkdir(mode=0o700, parents=True)
    logdir.mkdir(mode=0o700, parents=True)
    os.chmod(pipe, 0o700)
    os.chmod(logdir, 0o700)
    env = mps_environment(pipe, logdir, visible)
    process = None
    try:
        with (STATE / "mps_foreground.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                [str(MPS_CONTROL), "-f"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        process_group = cgroup(process.pid)
        if UNIT not in process_group:
            process.terminate()
            process.wait(timeout=3)
            raise RuntimeError(
                "foreground MPS process escaped the dedicated service cgroup"
            )
        write(
            STATE / "MPS_PROCESS.json",
            {
                "pid": process.pid,
                "cgroup": process_group,
                "command": [str(MPS_CONTROL), "-f"],
            },
        )
        # Empty server list is valid before any MPS client connects; private pipe must answer.
        last_error = ""
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"foreground MPS exited early with status {process.returncode}"
                )
            try:
                listing = control(pipe, logdir, visible, "get_server_list", timeout=3)
                write(
                    STATE / "MPS_READY.json",
                    {"server_list_before_clients": listing, "time_unix": time.time()},
                )
                return process
            except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                last_error = repr(error)
                time.sleep(0.5)
        raise TimeoutError(
            f"private MPS control pipe did not become ready: {last_error}"
        )
    except Exception:
        if process is not None and process.poll() is None:
            try:
                control(pipe, logdir, visible, "quit", timeout=2)
            except Exception as error:  # noqa: BLE001 - record failed startup cleanup
                write(STATE / "MPS_STARTUP_QUIT_ERROR.json", {"error": repr(error)})
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        raise


def cleanup_mps(
    process: subprocess.Popen[bytes] | None,
    pipe: Path,
    logdir: Path,
    visible: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"foreground_pid": process.pid if process else None}
    if process is not None and process.poll() is None:
        try:
            result["quit_response"] = control(pipe, logdir, visible, "quit", timeout=2)
        except Exception as error:  # noqa: BLE001 - retain bounded shutdown failure
            result["quit_error"] = repr(error)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # This is the exact foreground process we started, never a discovered host PID.
            process.terminate()
            result["fallback_signal"] = "SIGTERM exact foreground PID"
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                result["fallback_escalation"] = "SIGKILL exact foreground PID"
                process.wait(timeout=2)
    clients = []
    expires = time.monotonic() + 6
    while time.monotonic() < expires:
        clients = owned_cuda_clients()
        if not clients:
            break
        time.sleep(0.5)
    result["remaining_owned_cuda_clients"] = clients
    result["foreground_returncode"] = process.returncode if process else None
    result["finished_unix"] = time.time()
    if (
        pipe.exists()
        and not pipe.is_symlink()
        and pipe.parent == STATE
        and pipe.name == "mpspipe"
    ):
        shutil.rmtree(pipe)
        result["private_pipe_removed"] = True
    else:
        result["private_pipe_removed"] = not pipe.exists()
    return result


def main() -> None:
    if os.environ.get("TABDATA_SERVICE_UNIT") != UNIT:
        raise ValueError("probe service identity environment is missing or mismatched")
    if UNIT not in Path("/proc/self/cgroup").read_text(encoding="utf-8"):
        raise ValueError("probe must run inside its dedicated systemd cgroup")
    start = time.time()
    absolute_deadline = start + 300
    profile_deadline = start + 260
    STATE.mkdir(parents=True, exist_ok=True)
    owner_fd = os.open(
        STATE / "OWNER.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
    )
    with os.fdopen(owner_fd, "w", encoding="utf-8") as owner:
        owner.write(f"pid={os.getpid()}\nunit={UNIT}\n")
    base_plan_path = BASE_PACKET / "PLAN.json"
    expected_plan_sha = os.environ.get("TABDATA_BASE_PLAN_SHA256", "").lower()
    if (
        len(expected_plan_sha) != 64
        or delivery.sha(base_plan_path).lower() != expected_plan_sha
    ):
        raise ValueError(
            "active base plan is not bound to the externally supplied hash"
        )
    base_plan = read(base_plan_path)
    verify_base_files(base_plan)
    device_uuids = device_map(base_plan)
    direct_jobs, mps_jobs = make_jobs(device_uuids[0])
    all_jobs = direct_jobs + mps_jobs
    project_runs = BASE_STATE / "nll_project/artifacts/runs"
    for job in all_jobs:
        for suffix in ("", "_profile"):
            if (project_runs / f"{job['id']}{suffix}").exists():
                raise FileExistsError(
                    f"nonunique probe run identity already exists: {job['id']}{suffix}"
                )

    pipe, logdir = STATE / "mpspipe", STATE / "mpslog"
    if pipe.exists() or logdir.exists():
        raise FileExistsError("probe-private MPS pipe/log path already exists")
    visible = ",".join(device_uuids[index] for index in range(8))
    configure_delivery(pipe, logdir)
    write(
        STATE / "START.json",
        {
            "start_unix": start,
            "absolute_deadline_unix": absolute_deadline,
            "profile_deadline_unix": profile_deadline,
            "unit": UNIT,
            "base_plan_sha256": expected_plan_sha,
            "scope": "two matched strict-FP32 profile pairs only; no scientific fits",
        },
    )
    process: subprocess.Popen[bytes] | None = None
    result: dict[str, Any] = {
        "direct": None,
        "mps": None,
        "gpu_before": gpu_snapshot(device_uuids[0]),
        "other_mps_process_snapshot": existing_mps_processes(),
    }
    try:
        direct = run_pair(direct_jobs, profile_deadline)
        result["direct"] = direct
        result["gpu_after_direct"] = gpu_snapshot(device_uuids[0])
        if not direct["complete"]:
            raise RuntimeError(
                f"direct baseline profiles incomplete: {direct['errors']}"
            )

        result["private_pipe_created"] = False
        process = start_mps(
            pipe, logdir, visible, min(profile_deadline, time.time() + 25)
        )
        result["private_pipe_created"] = True
        result["mps_server_initial"] = mps_server_receipt(pipe, logdir, visible)
        result["gpu_before_mps"] = gpu_snapshot(device_uuids[0])
        mps = run_pair(mps_jobs, profile_deadline)
        result["mps"] = mps
        result["mps_server_after_profiles"] = mps_server_receipt(pipe, logdir, visible)
        result["gpu_after_mps"] = gpu_snapshot(device_uuids[0])
        if not mps["complete"]:
            raise RuntimeError(f"MPS profiles incomplete: {mps['errors']}")
        servers = result["mps_server_after_profiles"]["servers"]
        if not servers:
            raise RuntimeError(
                "MPS profiles completed but no private server PID was observable"
            )
        comparison = {}
        for arm in ("wmse", "nll"):
            direct_ms = float(direct["results"][arm]["timing"]["step_wall"])
            mps_ms = float(mps["results"][arm]["timing"]["step_wall"])
            comparison[arm] = {
                "direct_step_wall_ms": direct_ms,
                "mps_step_wall_ms": mps_ms,
                "mps_throughput_ratio": direct_ms / mps_ms,
                "direct_elapsed_seconds": direct["results"][arm]["elapsed_seconds"],
                "mps_elapsed_seconds": mps["results"][arm]["elapsed_seconds"],
                "direct_peak_reserved_bytes": direct["results"][arm]["memory"][
                    "peak_reserved_bytes"
                ],
                "mps_peak_reserved_bytes": mps["results"][arm]["memory"][
                    "peak_reserved_bytes"
                ],
            }
        result["matched_comparison"] = comparison
        result["status"] = "profile_only_complete"
    except Exception as error:  # noqa: BLE001 - always write a terminal probe receipt
        result["status"] = "failed_or_unsupported"
        result["error"] = repr(error)
    finally:
        result["cleanup"] = cleanup_mps(process, pipe, logdir, visible)
        result["cleanup"]["remaining_owned_mps_processes"] = [
            process_info
            for process_info in existing_mps_processes()
            if UNIT in process_info["cgroup"]
        ]
        result["total_elapsed_seconds"] = time.time() - start
        result["remaining_to_deadline_seconds"] = max(
            0.0, absolute_deadline - time.time()
        )
        write(
            STATE
            / (
                "RESULT.json"
                if result.get("status") == "profile_only_complete"
                else "FAILURE.json"
            ),
            result,
        )
    if result.get("cleanup", {}).get("remaining_owned_cuda_clients"):
        raise RuntimeError("probe cgroup still owns CUDA clients after MPS cleanup")
    if result.get("cleanup", {}).get("remaining_owned_mps_processes"):
        raise RuntimeError("probe cgroup still owns MPS processes after shutdown")
    if result.get("status") != "profile_only_complete":
        raise RuntimeError(result.get("error", "MPS probe did not complete"))
    print(
        json.dumps(
            {
                "status": result["status"],
                "matched_comparison": result["matched_comparison"],
                "total_elapsed_seconds": result["total_elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
