from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EXPECTED_JOBS = {"ctl_f0", "reg_f0", "ctl_f2", "reg_f2"}
PINNED_SOURCE_COMMIT = "0a70b4ab96c1810484544f382df4bcf0c08e8fd4"
MAX_HOURS = 24.0
DEFAULT_EVALUATION_SECONDS = 900


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _append_event(path: Path, value: dict[str, object]) -> None:
    encoded = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("ab") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _path(value: object, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"plan is missing path {label!r}")
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _load_plan(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("jobs"), list):
        raise ValueError("plan must contain a jobs array")
    if value.get("source_commit") != PINNED_SOURCE_COMMIT:
        raise ValueError("plan source_commit does not match the pinned canonical source")
    base = path.resolve().parent
    for field in ("source_root", "data_root", "run_root"):
        value[field] = str(_path(value.get(field), base, field))
    run_root = Path(value["run_root"])
    source_root = Path(value["source_root"])
    if not (source_root / "src" / "wnn").is_dir():
        raise ValueError("source_root must contain the pinned source's src/wnn tree")
    jobs: dict[str, dict[str, Any]] = {}
    all_gpus: list[str] = []
    for raw in value["jobs"]:
        if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
            raise ValueError("each plan job must have a name")
        name = raw["name"]
        if name in jobs:
            raise ValueError(f"duplicate job name: {name}")
        job = dict(raw)
        job["run_root"] = str(run_root)
        for field in ("config", "split_manifest", "profile_receipt"):
            job[field] = str(_path(job.get(field), base, f"{name}.{field}"))
        uuids = job.get("gpu_uuids")
        cpus = job.get("cpus")
        if (
            not isinstance(uuids, list)
            or not uuids
            or any(not isinstance(item, str) or not item for item in uuids)
        ):
            raise ValueError(f"{name} needs at least one assigned GPU UUID")
        if (
            not isinstance(cpus, list)
            or not cpus
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
        ):
            raise ValueError(f"{name} needs a nonempty CPU affinity list")
        all_gpus.extend(item.lower().removeprefix("gpu-") for item in uuids)
        profile = json.loads(Path(job["profile_receipt"]).read_text(encoding="utf-8"))
        identity = profile.get("identity") if isinstance(profile, dict) else None
        if not isinstance(identity, dict):
            raise ValueError(f"{name} profile receipt has no run identity")
        job["profile_identity"] = identity
        jobs[name] = job
    if set(jobs) != EXPECTED_JOBS:
        raise ValueError(f"plan must contain exactly {sorted(EXPECTED_JOBS)}")
    if len(all_gpus) != len(set(all_gpus)):
        raise ValueError("GPU assignments must be disjoint across the four jobs")
    return value, jobs


def _epoch_snapshots(job: dict[str, Any], source_root: Path) -> list[dict[str, Any]]:
    source_path = str(source_root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from wnn.training.sequence_epoch_index import load_epoch_index

    fit = Path(job["run_root"]) / job["name"] / "fit"
    snapshots = fit / "epoch_indexes"
    if snapshots.is_symlink() or (snapshots.exists() and not snapshots.is_dir()):
        raise ValueError(f"{job['name']} epoch_indexes must be a real directory")
    result: list[dict[str, Any]] = []
    if not snapshots.exists():
        return result
    for path in sorted(snapshots.glob("epoch-*.json")):
        index = load_epoch_index(path, job["profile_identity"])
        entry = index["entries"][-1]
        result.append(
            {
                "path": path.resolve(),
                "completed_epochs": index["completed_epochs"],
                "index_sha256": index["index_sha256"],
                "checkpoint_path": str((fit / str(entry["checkpoint_path"])).resolve()),
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "global_step": entry["global_step"],
            }
        )
    return result


def _record_existing(job: dict[str, Any], epoch: dict[str, Any], state: dict[str, Any]) -> bool:
    destination = (
        Path(job["run_root"])
        / job["name"]
        / "validation"
        / f"epoch-{epoch['completed_epochs']:06d}"
    )
    if destination.is_symlink():
        state["jobs"][job["name"]]["epochs"][str(epoch["completed_epochs"])] = {
            "status": "failed_preserved",
            "reason": "evaluation output path is a symlink",
            "epoch_index_sha256": epoch["index_sha256"],
            "checkpoint_sha256": epoch["checkpoint_sha256"],
        }
        return True
    if not destination.exists():
        return False
    result_path = destination / "RESULT.json"
    status = "failed_preserved"
    details: dict[str, object] = {"reason": "existing output has no terminal RESULT.json"}
    if result_path.is_file() and not result_path.is_symlink():
        try:
            prior = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior = {}
        if (
            prior.get("job") == job["name"]
            and prior.get("completed_epochs") == epoch["completed_epochs"]
            and prior.get("epoch_index_sha256") == epoch["index_sha256"]
            and prior.get("checkpoint_sha256") == epoch["checkpoint_sha256"]
        ):
            status = "complete" if prior.get("status") == "complete" else "failed_preserved"
            details = {
                "result_path": str(result_path.resolve()),
                "result_status": prior.get("status"),
            }
    state["jobs"][job["name"]]["epochs"][str(epoch["completed_epochs"])] = {
        "status": status,
        "epoch_index_sha256": epoch["index_sha256"],
        "checkpoint_sha256": epoch["checkpoint_sha256"],
        **details,
    }
    return True


def _start_evaluation(
    plan_path: Path,
    plan: dict[str, Any],
    job: dict[str, Any],
    epoch: dict[str, Any],
    batch_size: int,
    max_seconds: int,
) -> dict[str, Any]:
    source_root = Path(plan["source_root"])
    run_root = Path(plan["run_root"])
    destination = run_root / job["name"] / "validation" / f"epoch-{epoch['completed_epochs']:06d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"evaluation output already exists: {destination}")
    destination.mkdir()
    (destination / "worker.log").touch(exist_ok=False)
    expected_gpu = job["gpu_uuids"][0]
    command = [
        sys.executable,
        str(Path(__file__).resolve().with_name("evaluate.py")),
        "--plan",
        str(plan_path.resolve()),
        "--job",
        job["name"],
        "--epoch-index",
        str(epoch["path"]),
        "--epoch-index-sha256",
        str(epoch["index_sha256"]),
        "--output",
        str(destination.resolve()),
        "--batch-size",
        str(batch_size),
        "--max-gpu-gib",
        "12",
        "--max-seconds",
        str(max_seconds),
        "--expected-gpu-uuid",
        expected_gpu,
    ]
    environment = os.environ.copy()
    source_path = str(source_root / "src")
    runtime_path = str(Path(__file__).resolve().parent)
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (runtime_path, source_path, environment.get("PYTHONPATH", "")) if item
    )
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": expected_gpu,
            "CUBLAS_EMULATE_SINGLE_PRECISION": "1",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCH_LOGS": "graph_breaks,recompiles",
        }
    )
    cache = run_root / ".validation-cache" / job["name"]
    scratch = run_root / ".validation-tmp" / job["name"]
    cache.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache)
    environment["TMPDIR"] = str(scratch)
    log = (destination / "worker.log").open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            command,
            cwd=source_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log.close()
        raise
    return {
        "process": process,
        "log": log,
        "job": job,
        "epoch": epoch,
        "destination": destination,
        "started_monotonic": time.monotonic(),
        "pid": process.pid,
        "command": command,
    }


def _stop_owned(active: dict[str, dict[str, Any]]) -> None:
    for item in active.values():
        process = item["process"]
        if process.poll() is not None:
            continue
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        finally:
            item["log"].close()


def _job_done(job: dict[str, Any], state: dict[str, Any], epochs: list[dict[str, Any]]) -> bool:
    required = int(json.loads(Path(job["config"]).read_text(encoding="utf-8"))["epochs"])
    statuses = state["jobs"][job["name"]]["epochs"]
    return len(epochs) >= required and all(
        str(number) in statuses and statuses[str(number)]["status"] != "running"
        for number in range(1, required + 1)
    )


def run_monitor(args: argparse.Namespace) -> int:
    plan_path = args.plan.resolve()
    plan, jobs = _load_plan(plan_path)
    output = args.output.resolve()
    if output.is_symlink():
        raise ValueError("monitor output cannot be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / "MONITOR.json"
    if state_path.exists() or state_path.is_symlink():
        raise FileExistsError(
            f"monitor state already exists; refusing a second owner: {state_path}"
        )
    events_path = output / "events.jsonl"
    plan_bytes = plan_path.read_bytes()
    state: dict[str, Any] = {
        "schema": "wnn.compact-transformer-monitor.v1",
        "status": "running",
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "started_utc": _utc_now(),
        "deadline_seconds": args.max_hours * 3600,
        "jobs": {name: {"epochs": {}, "status": "running"} for name in sorted(jobs)},
    }
    _atomic_json(state_path, state)
    _append_event(events_path, {"time_utc": _utc_now(), "event": "monitor_started"})

    source_root = Path(plan["source_root"])
    sys.path.insert(0, str(source_root / "src"))
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    active: dict[str, dict[str, Any]] = {}
    try:
        while time.monotonic() < deadline:
            indexed: dict[str, list[dict[str, Any]]] = {}
            for name, job in jobs.items():
                if state["jobs"][name]["status"] != "running":
                    indexed[name] = []
                    continue
                try:
                    indexed[name] = _epoch_snapshots(job, source_root)
                except Exception as error:
                    indexed[name] = []
                    state["jobs"][name]["status"] = "invalid_epoch_index"
                    state["jobs"][name]["error"] = f"{type(error).__name__}: {error}"
                    _append_event(
                        events_path,
                        {
                            "time_utc": _utc_now(),
                            "event": "epoch_index_rejected",
                            "job": name,
                            "error": state["jobs"][name]["error"],
                        },
                    )
            for name, job in jobs.items():
                for epoch in indexed[name]:
                    key = str(epoch["completed_epochs"])
                    if key in state["jobs"][name]["epochs"] or name in active:
                        continue
                    if _record_existing(job, epoch, state):
                        _append_event(
                            events_path,
                            {
                                "time_utc": _utc_now(),
                                "event": "existing_output_preserved",
                                "job": name,
                                "epoch": epoch["completed_epochs"],
                            },
                        )
                        continue
                    try:
                        item = _start_evaluation(
                            plan_path, plan, job, epoch, args.batch_size, args.evaluation_seconds
                        )
                    except Exception as error:
                        state["jobs"][name]["epochs"][key] = {
                            "status": "admission_failed",
                            "reason": f"{type(error).__name__}: {error}",
                            "epoch_index_sha256": epoch["index_sha256"],
                        }
                        continue
                    active[name] = item
                    state["jobs"][name]["epochs"][key] = {
                        "status": "running",
                        "pid": item["pid"],
                        "started_utc": _utc_now(),
                        "epoch_index_sha256": epoch["index_sha256"],
                        "checkpoint_sha256": epoch["checkpoint_sha256"],
                        "command": item["command"],
                    }
                    _append_event(
                        events_path,
                        {
                            "time_utc": _utc_now(),
                            "event": "evaluation_started",
                            "job": name,
                            "epoch": epoch["completed_epochs"],
                            "pid": item["pid"],
                        },
                    )

            for name, item in list(active.items()):
                process = item["process"]
                timed_out = (
                    time.monotonic() - item["started_monotonic"] > args.evaluation_seconds + 30
                )
                if process.poll() is None and not timed_out:
                    continue
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                item["log"].close()
                result_path = item["destination"] / "RESULT.json"
                try:
                    result = (
                        json.loads(result_path.read_text(encoding="utf-8"))
                        if result_path.is_file()
                        else {}
                    )
                except (OSError, json.JSONDecodeError):
                    result = {}
                epoch = item["epoch"]
                good = (
                    process.returncode == 0
                    and result.get("status") == "complete"
                    and result.get("epoch_index_sha256") == epoch["index_sha256"]
                    and result.get("checkpoint_sha256") == epoch["checkpoint_sha256"]
                )
                state["jobs"][name]["epochs"][str(epoch["completed_epochs"])] = {
                    "status": "complete" if good else "failed_preserved",
                    "exit_code": process.returncode,
                    "timed_out": timed_out,
                    "result_path": str(result_path),
                    "epoch_index_sha256": epoch["index_sha256"],
                    "checkpoint_sha256": epoch["checkpoint_sha256"],
                }
                _append_event(
                    events_path,
                    {
                        "time_utc": _utc_now(),
                        "event": "evaluation_finished",
                        "job": name,
                        "epoch": epoch["completed_epochs"],
                        "status": state["jobs"][name]["epochs"][str(epoch["completed_epochs"])][
                            "status"
                        ],
                        "exit_code": process.returncode,
                    },
                )
                del active[name]

            for name, job in jobs.items():
                if state["jobs"][name]["status"] == "running" and _job_done(
                    job, state, indexed[name]
                ):
                    epoch_statuses = [
                        record["status"] for record in state["jobs"][name]["epochs"].values()
                    ]
                    state["jobs"][name]["status"] = (
                        "complete"
                        if all(status == "complete" for status in epoch_statuses)
                        else "partial"
                    )
            if all(item["status"] != "running" for item in state["jobs"].values()) and not active:
                state["status"] = (
                    "complete"
                    if all(item["status"] == "complete" for item in state["jobs"].values())
                    else "partial"
                )
                break
            _atomic_json(state_path, state)
            time.sleep(args.poll_seconds)
        else:
            state["status"] = "partial_deadline"
        if active:
            _stop_owned(active)
            for name, item in active.items():
                epoch = item["epoch"]
                state["jobs"][name]["epochs"][str(epoch["completed_epochs"])] = {
                    "status": "terminated_at_monitor_deadline",
                    "epoch_index_sha256": epoch["index_sha256"],
                    "checkpoint_sha256": epoch["checkpoint_sha256"],
                    "pid": item["pid"],
                }
            active.clear()
        if state["status"] == "running":
            state["status"] = "partial"
        for item in state["jobs"].values():
            if item["status"] == "running":
                item["status"] = "partial"
        state["finished_utc"] = _utc_now()
        _atomic_json(state_path, state)
        _append_event(
            events_path,
            {"time_utc": _utc_now(), "event": "monitor_finished", "status": state["status"]},
        )
        return 0 if state["status"] == "complete" else 2
    finally:
        _stop_owned(active)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Observe immutable safe-epoch indexes and evaluate each once."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--batch-size", type=int, choices=(4, 8), default=4)
    parser.add_argument("--evaluation-seconds", type=int, default=DEFAULT_EVALUATION_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if not 0 < args.max_hours <= MAX_HOURS:
        parser.error("--max-hours must be in (0, 24]")
    if not 1 <= args.evaluation_seconds <= DEFAULT_EVALUATION_SECONDS:
        parser.error("--evaluation-seconds must be in [1, 900]")
    if not 0.1 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be in [0.1, 60]")
    raise SystemExit(run_monitor(args))


if __name__ == "__main__":
    main()
