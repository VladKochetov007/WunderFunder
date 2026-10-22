"""Run sixteen bounded H512 cold-label NLL/WMSE fits under the parent node deadline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Any

ROOT = Path("/data/epanov_s/tabdata")
BASE_PACKET = ROOT / "runtime/next4h-20261022-r2"
BASE_STATE = ROOT / "state/next4h-20261022-nllfixed"
PACKET = ROOT / "runtime/next4h-20261022-nllfixed-h512"
SIDE_PLAN = PACKET / "SIDEPLAN.json"
STATE = ROOT / "state/next4h-20261022-nllfixed-h512"
UNIT = "tabdata-next4h-20261022-nllfixed-h512.service"
PROFILE_CUTOFF = 5400
FIT_CUTOFF = 11100
EVAL_CUTOFF = 12900
GRADE_CUTOFF = 13140

sys.path.insert(0, str(BASE_PACKET))
sys.path.insert(0, str(PACKET))

import delivery  # noqa: E402


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def bind_delivery() -> None:
    """Reuse the sealed operator while redirecting every owned receipt to side state."""
    delivery.PACKET = BASE_PACKET
    delivery.STATE = STATE
    delivery.UNIT = UNIT


def job_key(job: dict[str, Any]) -> tuple[int, int]:
    return int(job["fold"]), int(job["seed"])


def expected_jobs() -> dict[tuple[int, int, str], tuple[int, list[int]]]:
    expected: dict[tuple[int, int, str], tuple[int, list[int]]] = {}
    for slot, (fold, seed, gpu) in enumerate(
        [
            (0, 1201, 0),
            (2, 1202, 1),
            (0, 1301, 2),
            (0, 1302, 3),
            (0, 1303, 4),
            (2, 1401, 5),
            (2, 1402, 6),
            (2, 1403, 7),
        ]
    ):
        for arm_index, arm in enumerate(("wmse", "nll")):
            cpu_start = 128 + slot * 16 + arm_index * 8
            expected[(fold, seed, arm)] = (gpu, list(range(cpu_start, cpu_start + 8)))
    return expected


def verify(base_plan: dict[str, Any], plan: dict[str, Any]) -> None:
    """Bind both source cohorts and reject drift in the finite side allocation."""
    expected_base = os.environ["TABDATA_BASE_PLAN_SHA256"].lower()
    if delivery.sha(BASE_PACKET / "PLAN.json").lower() != expected_base:
        raise ValueError("base cohort PLAN differs from separately supplied SHA256")
    if plan.get("base_plan_sha256", "").lower() != expected_base:
        raise ValueError("side plan is not bound to the active base plan")
    expected_side = os.environ["TABDATA_WIDE_PLAN_SHA256"].lower()
    if delivery.sha(SIDE_PLAN).lower() != expected_side:
        raise ValueError("side allocation differs from separately sealed plan hash")
    base_start = read(BASE_STATE / "START.json")
    if (
        abs(float(plan.get("original_start_unix", -1)) - float(base_start["start_unix"])) > 1e-6
        or abs(float(plan.get("deadline_unix", -1)) - float(base_start["deadline_unix"])) > 1e-6
        or plan.get("unit") != UNIT
    ):
        raise ValueError("side plan changes the original deadline or service identity")
    for filename, digest in plan.get("files", {}).items():
        if delivery.sha(Path(filename)) != digest:
            raise ValueError(f"side packet file hash mismatch: {filename}")

    # Existing verifier checks the source packet and data mounts without launching work.
    old_hash = os.environ.get("TABDATA_PLAN_SHA256")
    os.environ["TABDATA_PLAN_SHA256"] = expected_base
    try:
        import cohort

        cohort.verify(base_plan)
    finally:
        if old_hash is None:
            os.environ.pop("TABDATA_PLAN_SHA256", None)
        else:
            os.environ["TABDATA_PLAN_SHA256"] = old_hash

    if plan.get("hidden") != 512:
        raise ValueError("Wide recipe requires hidden=512")
    expected = expected_jobs()
    visible_cpus = os.sched_getaffinity(0)
    devices = {job["gpu_index"]: job["gpu_uuid"] for job in base_plan["jobs"]}
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 16:
        raise ValueError("wide plan must contain exactly sixteen new jobs")
    seen: set[tuple[int, int, str]] = set()
    ids: set[str] = set()
    for job in jobs:
        key = (int(job["fold"]), int(job["seed"]), str(job["arm"]))
        if key not in expected or key in seen:
            raise ValueError(f"unexpected or duplicate wide job: {key}")
        gpu, cpus = expected[key]
        expected_id = f"fixedh512_f{key[0]}_s{key[1]}_{key[2]}"
        if (
            job.get("family") != "nll"
            or job.get("id") != expected_id
            or int(job.get("gpu_index", -1)) != gpu
            or list(map(int, job.get("cpus", []))) != cpus
            or not isinstance(job.get("gpu_uuid"), str)
            or not job["gpu_uuid"].startswith("GPU-")
            or job["gpu_uuid"] != devices[gpu]
            or not set(cpus).issubset(visible_cpus)
        ):
            raise ValueError(f"wide job violates frozen GPU/CPU/source allocation: {key}")
        seen.add(key)
        ids.add(expected_id)
    if seen != set(expected):
        raise ValueError("wide plan omits a declared matched arm")


def require_fresh_outputs(jobs: list[dict[str, Any]]) -> None:
    """Share verified project inputs while refusing every existing side run/output."""
    project = BASE_STATE / "nll_project"
    for job in jobs:
        for name in (job["id"], job["id"] + "_profile"):
            if (project / "artifacts/runs" / name).exists():
                raise FileExistsError("Existing wide run: " + name)
            if list((project / "cache").glob("pred_" + name + "*")):
                raise FileExistsError("Existing wide prediction: " + name)
        if (STATE / "logs" / job["id"]).exists():
            raise FileExistsError("Existing wide log: " + job["id"])


def configure_delivery() -> None:
    bind_delivery()
    original_environment = delivery.environment

    def side_environment(job: dict[str, Any] | None = None) -> dict[str, str]:
        env = original_environment(job)
        env["WNN_PROJECT_ROOT"] = str(BASE_STATE / "nll_project")
        # The eight pinned CPUs per worker include the recipe's four data-loader workers.
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        return env

    delivery.environment = side_environment
    original_command = delivery.nll_command

    def wide_command(job: dict[str, Any], stage: str) -> list[str]:
        command = original_command(job, stage)
        if stage != "evaluate":
            command[command.index("--hidden") + 1] = "512"
        return command

    delivery.nll_command = wide_command


def publish(final: bool = False) -> None:
    """Make a bounded progress update; publication errors remain local evidence."""
    stamp = str(time.time_ns())
    deadline = min(read(STATE / "START.json")["start_unix"] + GRADE_CUTOFF, time.time() + 85)
    try:
        delivery.phase(
            [
                str(delivery.PYTHON),
                str(PACKET / "wide_publish.py"),
                *(["--final"] if final else []),
            ],
            cwd=ROOT,
            env=delivery.environment(),
            log=STATE / "publication_logs" / f"{stamp}.log",
            deadline=deadline,
        )
    except Exception as error:
        write(STATE / "publication_logs" / f"{stamp}.failure.json", {"error": repr(error)})


def run_profile(job: dict[str, Any], deadline: float) -> dict[str, Any]:
    result = delivery.run_job_phase(job, "profile", deadline)
    approved = delivery.approve_profile(job, result)
    write(STATE / "profiles" / f"{job['id']}.json", approved)
    return approved


def profile_pair(jobs: list[dict[str, Any]], deadline: float) -> dict[str, Any]:
    """Profile each paired arm serially on its GPU, then reserve aggregate headroom."""
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for job in sorted(jobs, key=lambda item: item["arm"]):
        try:
            results[job["id"]] = run_profile(job, deadline)
        except Exception as error:
            errors[job["id"]] = repr(error)
            break
    if errors or len(results) != 2:
        return {"passed": False, "profiles": results, "errors": errors}

    gpu_uuid = jobs[0]["gpu_uuid"]
    observation = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            gpu_uuid,
            "--query-gpu=uuid,memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    fields = [value.strip() for value in observation.split(",")]
    if fields[0] != gpu_uuid:
        return {"passed": False, "profiles": results, "errors": {"gpu": "UUID mismatch"}}
    used, free, total = map(float, fields[1:])
    reserved = sum(float(results[j["id"]]["memory"]["peak_reserved_bytes"]) for j in jobs)
    # Profile processes have exited; current used memory captures co-resident base fits.
    used_bytes = used * 1024 * 1024
    free_bytes = free * 1024 * 1024
    total_bytes = total * 1024 * 1024
    # Both width controllers admit at most 32 GiB per pair; reserve their sum.
    aggregate_reserve = 64 * 1024**3
    if reserved > 32 * 1024**3 or used_bytes + aggregate_reserve >= 0.90 * total_bytes or aggregate_reserve > free_bytes:
        return {
            "passed": False,
            "profiles": results,
            "errors": {
                "aggregate_memory": "paired measured peaks plus observed residents fail 90%/free-memory gate"
            },
            "gpu_observation": observation,
            "paired_peak_reserved_bytes": reserved,
        }
    return {
        "passed": True,
        "profiles": results,
        "gpu_observation": observation,
        "paired_peak_reserved_bytes": reserved,
        "co_resident_memory_used_bytes": used_bytes,
    }


def fit(job: dict[str, Any], start: float) -> dict[str, Any]:
    record: dict[str, Any] = {"job": job, "phase": "fitting", "started_unix": time.time()}
    path = STATE / "jobs" / f"{job['id']}.json"
    write(path, record)
    try:
        record["fit"] = delivery.run_job_phase(job, "fit", start + FIT_CUTOFF)
        record["phase"] = "fit_complete"
        write(path, record)
        record["evaluation"] = delivery.run_job_phase(job, "evaluate", start + EVAL_CUTOFF)
        record["phase"] = "evaluated"
    except Exception as error:
        record["phase"] = "failed_or_partial"
        record["error"] = repr(error)
    record["finished_unix"] = time.time()
    write(path, record)
    return record


def grade_pairs(jobs: list[dict[str, Any]], start: float) -> None:
    by_pair: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for job in jobs:
        path = STATE / "jobs" / f"{job['id']}.json"
        if path.exists():
            record = read(path)
            by_pair.setdefault(job_key(job), {})[job["arm"]] = record
    for (fold, seed), arms in sorted(by_pair.items()):
        output = STATE / "grades" / f"pair_f{fold}_s{seed}.json"
        failure = STATE / "grades" / f"pair_f{fold}_s{seed}.failure.json"
        if output.exists() or failure.exists() or set(arms) != {"nll", "wmse"}:
            continue
        if any(arms[arm].get("phase") != "evaluated" for arm in ("nll", "wmse")):
            continue
        nll = f"fixedh512_f{fold}_s{seed}_nll@ep2"
        wmse = f"fixedh512_f{fold}_s{seed}_wmse@ep2"
        stamp = str(time.time_ns())
        try:
            delivery.phase(
                [
                    str(delivery.PYTHON),
                    str(BASE_PACKET / "nll_grade.py"),
                    "--project-root",
                    str(BASE_STATE / "nll_project"),
                    "--split-manifest",
                    str(ROOT / "splits" / f"fold{fold}.json"),
                    "--candidate",
                    nll,
                    "--control",
                    wmse,
                    "--output",
                    str(output),
                    "--plan",
                    str(SIDE_PLAN),
                    "--plan-sha256",
                    os.environ["TABDATA_WIDE_PLAN_SHA256"],
                ],
                cwd=PACKET,
                env=delivery.environment(),
                log=STATE / "grade_logs" / f"f{fold}_s{seed}_{stamp}.log",
                deadline=start + GRADE_CUTOFF,
            )
        except Exception as error:
            write(STATE / "grades" / f"pair_f{fold}_s{seed}.failure.json", {"error": repr(error)})
        publish()


def monitor(stop: threading.Event) -> None:
    next_publish = time.time() + 3600
    while not stop.wait(30):
        try:
            snapshot = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            with (STATE / "gpu_samples.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"time_unix": time.time(), "gpu_csv": snapshot}) + "\n")
            if time.time() >= next_publish:
                publish()
                next_publish = time.time() + 3600
        except Exception as error:
            write(STATE / "MONITOR_ERROR.json", {"error": repr(error), "time_unix": time.time()})


def main() -> None:
    base_plan = read(BASE_PACKET / "PLAN.json")
    plan = read(SIDE_PLAN)
    verify(base_plan, plan)
    require_fresh_outputs(plan["jobs"])
    configure_delivery()
    start_record = read(BASE_STATE / "START.json")
    start = float(start_record["start_unix"])
    if os.environ.get("TABDATA_SERVICE_UNIT") != UNIT:
        raise ValueError("wide service identity environment is missing or mismatched")
    cgroup = Path("/proc/self/cgroup").read_text(encoding="utf-8")
    if UNIT not in cgroup:
        raise ValueError("wide process is not inside its dedicated systemd unit")
    STATE.mkdir(parents=True, exist_ok=True)
    try:
        owner_fd = os.open(STATE / "OWNER.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("wide one-shot has already been claimed") from error
    with os.fdopen(owner_fd, "w", encoding="utf-8") as owner:
        owner.write(f"pid={os.getpid()}\nunit={UNIT}\n")
    if time.time() >= start + FIT_CUTOFF:
        raise TimeoutError("Scaleout arrived after the common fit-admission cutoff")
    if (STATE / "START.json").exists():
        raise FileExistsError("Scaleout state already contains a start receipt")
    write(
        STATE / "START.json",
        {
            "start_unix": start,
            "sidecar_started_unix": time.time(),
            "deadline_unix": float(start_record["deadline_unix"]),
            "service": UNIT,
            "base_plan_sha256": os.environ["TABDATA_BASE_PLAN_SHA256"],
            "wide_plan_sha256": os.environ["TABDATA_WIDE_PLAN_SHA256"],
            "partial_endpoints_allowed": True,
        },
    )
    write(
        STATE / "PLAN_BINDING.json",
        {
            "base_plan": os.environ["TABDATA_BASE_PLAN_SHA256"],
            "wide_plan": os.environ["TABDATA_WIDE_PLAN_SHA256"],
        },
    )
    stop = threading.Event()
    monitor_thread = threading.Thread(target=monitor, args=(stop,), daemon=True)
    monitor_thread.start()
    jobs = plan["jobs"]
    publish()

    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for job in jobs:
        groups.setdefault(job_key(job), []).append(job)
    profile_results: dict[tuple[int, int], dict[str, Any]] = {}
    profile_deadline = start + PROFILE_CUTOFF
    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        running = {
            pool.submit(profile_pair, pair, profile_deadline): key for key, pair in groups.items()
        }
        for completed in futures.as_completed(running):
            key = running[completed]
            try:
                profile_results[key] = completed.result()
            except Exception as error:
                profile_results[key] = {"passed": False, "errors": {"worker": repr(error)}}
    write(
        STATE / "PROFILES.json",
        {f"f{fold}_s{seed}": value for (fold, seed), value in sorted(profile_results.items())},
    )
    admitted_pairs = [key for key, result in profile_results.items() if result.get("passed")]
    admitted_ids = [job["id"] for job in jobs if job_key(job) in admitted_pairs]
    write(
        STATE / "ADMISSION.json",
        {
            "admitted_pairs": [f"f{fold}_s{seed}" for fold, seed in sorted(admitted_pairs)],
            "admitted_jobs": admitted_ids,
            "failed_pairs": {
                f"f{f}_s{s}": v for (f, s), v in profile_results.items() if not v.get("passed")
            },
            "fit_cutoff_unix": start + FIT_CUTOFF,
        },
    )
    publish()
    admitted = [job for job in jobs if job["id"] in set(admitted_ids)]
    if admitted:
        with futures.ThreadPoolExecutor(max_workers=len(admitted)) as pool:
            running = {pool.submit(fit, job, start): job for job in admitted}
            for completed in futures.as_completed(running):
                completed.result()
                # This grades/publishes a pair as soon as both arms have completed.
                grade_pairs(admitted, start)
    stop.set()
    monitor_thread.join(timeout=5)
    write(
        STATE / "COMPLETE.json", {"finished_unix": time.time(), "jobs": [j["id"] for j in admitted]}
    )
    publish()


if __name__ == "__main__":
    main()
