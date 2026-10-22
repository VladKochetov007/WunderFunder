"""A finite ten-fit cohort under a nonrenewing four-hour node deadline."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from concurrent import futures
from pathlib import Path
from typing import Any

from delivery import (
    PACKET,
    PYTHON,
    ROOT,
    STATE,
    UNIT,
    approve_profile,
    environment,
    phase,
    read,
    run_job_phase,
    sha,
    write,
)


def verify(plan: dict[str, Any]) -> None:
    """Reject code, input or finite-allocation drift before any GPU use."""
    if sha(PACKET / "PLAN.json") != os.environ["TABDATA_PLAN_SHA256"]:
        raise ValueError("Allocation differs from the independently sealed launch hash")
    expected = set()
    for fold, seed, offset in ((0, 1001, 2), (2, 1002, 5)):
        for index, arm in enumerate(("raw_events", "uniform16", "quantile16")):
            expected.add(("numeric", fold, seed, arm, offset + index))
    for fold, seed, gpu in ((0, 1201, 0), (2, 1202, 1)):
        for arm in ("wmse", "nll"):
            expected.add(("nll", fold, seed, arm, gpu))
    actual = {(j["family"], j["fold"], j["seed"], j["arm"], j["gpu_index"]) for j in plan["jobs"]}
    if actual != expected or len({j["id"] for j in plan["jobs"]}) != 10 or len(plan["files"]) < 100:
        raise ValueError("Unexpected job panel or incomplete executable identity")
    project = STATE / "nll_project"
    for name, target in (
        ("src", PACKET / "sources/nll/src"),
        ("scripts", PACKET / "sources/nll/scripts"),
        ("datasets", ROOT / "data/datasets"),
    ):
        if not (project / name).is_symlink() or (project / name).resolve() != target.resolve():
            raise ValueError("NLL explicit input/source mount mismatch: " + name)
    for filename in ("train_targets.npz", "valid_targets.npz", "chains.npz", "chain_folds.npz"):
        if (project / "cache" / filename).resolve() != (ROOT / "data/cache" / filename).resolve():
            raise ValueError("NLL cache mount mismatch")
    for name, digest in plan["files"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Sealed packet changed: " + name)
    if len(plan["jobs"]) != 10 or plan["max_wall_seconds"] != 14400:
        raise ValueError("Incorrect finite cohort authority")


def publish(final: bool = False) -> None:
    """Bound periodic publication; preserve failures instead of retrying writes."""
    command = [str(PYTHON), str(PACKET / "publish.py")]
    if final:
        command += ["--final"]
    stamp = str(time.time_ns())
    try:
        phase(
            command,
            cwd=ROOT,
            env=environment(),
            log=STATE / "publication_logs" / (stamp + ".log"),
            deadline=min(read(STATE / "START.json")["start_unix"] + 13400, time.time() + 100),
        )
    except Exception as error:
        write(
            STATE / "publication_logs" / (stamp + ".failure.json"),
            {"error": repr(error)},
        )


def monitor(stop: threading.Event) -> None:
    """Sample all GPU lanes and refresh GitHub hourly without training admission."""
    next_publish = time.time() + 3600
    while not stop.wait(15):
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            with (STATE / "gpu_samples.jsonl").open("a") as stream:
                stream.write(json.dumps({"time_unix": time.time(), "gpu_csv": result}) + "\n")
            if time.time() >= next_publish:
                publish()
                next_publish = time.time() + 3600
        except Exception as error:
            write(
                STATE / "MONITOR_ERROR.json",
                {"error": repr(error), "time_unix": time.time()},
            )


def prepare(plan: dict[str, Any], start: float) -> None:
    """Calibrate numeric features on fitting components only, once per fold."""
    from delivery import numeric_command

    selected = [
        next(j for j in plan["jobs"] if j["family"] == "numeric" and j["fold"] == fold)
        for fold in (0, 2)
    ]

    def worker(job: dict[str, Any]) -> None:
        phase(
            numeric_command(job, "prepare"),
            cwd=PACKET / "sources/numeric",
            env=environment(),
            log=STATE / "logs" / f"prepare_f{job['fold']}.log",
            deadline=start + 1800,
            cpus=job["cpus"],
        )

    with futures.ThreadPoolExecutor(max_workers=2) as pool:
        for _result in pool.map(worker, selected):
            pass


def deliver(job: dict[str, Any], start: float) -> dict[str, Any]:
    """Fit once and evaluate the fixed endpoint; retain partial runs at cutoff."""
    record = {"job": job, "phase": "fitting", "started_unix": time.time()}
    location = STATE / "jobs" / (job["id"] + ".json")
    write(location, record)
    try:
        record["fit"] = run_job_phase(job, "fit", start + 11100)
        record["phase"] = "fit_complete"
        write(location, record)
        record["evaluation"] = run_job_phase(job, "evaluate", start + 12900)
        record["phase"] = "evaluated"
    except Exception as error:
        record["phase"] = "failed_or_partial"
        record["error"] = repr(error)
    record["finished_unix"] = time.time()
    write(location, record)
    return record


def grade(start: float) -> None:
    """Grade only complete comparisons using frozen official evaluation."""
    config = (
        PACKET
        / "sources/numeric/research/experiment_drafts/typed-bin-feature-encoding/r2_e3/configs"
    )
    numeric = [read(p) for p in (STATE / "jobs").glob("numeric_*.json")]
    if len(numeric) == 6 and all(r["phase"] == "evaluated" for r in numeric):
        try:
            phase(
                [
                    str(PYTHON),
                    "scripts/score_numeric_panel.py",
                    "--runs",
                    str(STATE / "numeric/runs_r2_e3"),
                    "--configs",
                    str(config),
                    "--output",
                    str(STATE / "numeric/panel_r2_e3.json"),
                ],
                cwd=PACKET / "sources/numeric",
                env=environment(),
                log=STATE / "logs/numeric_panel.log",
                deadline=start + 13140,
            )
        except Exception as error:
            write(STATE / "NUMERIC_GRADE_FAILURE.json", {"error": repr(error)})
    for fold in (0, 2):
        jobs = [read(p) for p in (STATE / "jobs").glob(f"nll_f{fold}_*.json")]
        if len(jobs) == 2 and all(r["phase"] == "evaluated" for r in jobs):
            by_arm = {r["job"]["arm"]: r["job"]["id"] for r in jobs}
            try:
                phase(
                    [
                        str(PYTHON),
                        str(PACKET / "nll_grade.py"),
                        "--project-root",
                        str(STATE / "nll_project"),
                        "--split-manifest",
                        str(ROOT / "splits" / f"fold{fold}.json"),
                        "--candidate",
                        by_arm["nll"] + "@ep2",
                        "--control",
                        by_arm["wmse"] + "@ep2",
                        "--output",
                        str(STATE / f"nll_grade_f{fold}.json"),
                        "--plan",
                        str(PACKET / "PLAN.json"),
                        "--plan-sha256",
                        os.environ["TABDATA_PLAN_SHA256"],
                    ],
                    cwd=PACKET / "sources/nll",
                    env=dict(environment(), PYTHONPATH=str(PACKET / "sources/nll/src")),
                    log=STATE / "logs" / f"nll_grade_f{fold}.log",
                    deadline=start + 13140,
                )
            except Exception as error:
                write(STATE / f"NLL_GRADE_FAILURE_F{fold}.json", {"error": repr(error)})


def main() -> None:
    """Own profiles, ten fits, grading and an independently enforced terminal stop."""
    plan = read(PACKET / "PLAN.json")
    verify(plan)
    start = time.time()
    if (STATE / "START.json").exists():
        raise FileExistsError("No implicit cohort restart or renewed deadline")
    write(
        STATE / "START.json",
        {
            "start_unix": start,
            "deadline_unix": start + 14400,
            "service": UNIT,
            "first_phase": "fitting_only_numeric_preparation",
        },
    )
    stop = threading.Event()
    watcher = threading.Thread(target=monitor, args=(stop,), daemon=True)
    watcher.start()
    publish()
    try:
        prepare(plan, start)
        profiles, errors = {}, {}
        with futures.ThreadPoolExecutor(max_workers=10) as pool:
            running = {
                pool.submit(run_job_phase, job, "profile", start + 3600): job
                for job in plan["jobs"]
            }
            for completed in futures.as_completed(running):
                job = running[completed]
                try:
                    profiles[job["id"]] = approve_profile(job, completed.result())
                except Exception as error:
                    errors[job["id"]] = repr(error)
        totals = {}
        for job in plan["jobs"]:
            if job["id"] in profiles:
                memory = profiles[job["id"]]["memory"]
                totals[job["gpu_uuid"]] = (
                    totals.get(job["gpu_uuid"], 0) + memory["peak_reserved_bytes"]
                )
        for job in plan["jobs"]:
            if (
                job["id"] in profiles
                and totals[job["gpu_uuid"]] >= 0.9 * profiles[job["id"]]["memory"]["total_bytes"]
            ):
                errors[job["id"]] = (
                    "Aggregate co-resident measured peaks exceed ninety percent VRAM"
                )
                del profiles[job["id"]]
        write(
            STATE / "PROFILES.json",
            {"results": profiles, "failures": errors, "co_resident_reserved_peak_sums": totals},
        )
        admitted = [
            j
            for j in plan["jobs"]
            if all(k["id"] in profiles for k in plan["jobs"] if k["family"] == j["family"])
        ]
        write(
            STATE / "ADMISSION.json",
            {
                "admitted": [j["id"] for j in admitted],
                "profile_failures": errors,
                "partial_endpoints_allowed": True,
                "partial_is_quality_endpoint": False,
                "time_unix": time.time(),
            },
        )
        publish()
        results = []
        with futures.ThreadPoolExecutor(max_workers=10) as pool:
            for result in futures.as_completed(
                [pool.submit(deliver, job, start) for job in admitted]
            ):
                results.append(result.result())
                write(
                    STATE / "PROGRESS.json",
                    {"results": results, "time_unix": time.time()},
                )
                publish()
        grade(start)
        verify(plan)
        write(
            STATE / "COMPLETE.json",
            {
                "stage": "finite_cohort_terminal",
                "results": results,
                "profile_failures": errors,
                "elapsed_seconds": time.time() - start,
            },
        )
    except Exception as error:
        write(STATE / "FAILURE.json", {"error": repr(error), "time_unix": time.time()})
        raise
    finally:
        stop.set()
        watcher.join(timeout=20)


if __name__ == "__main__":
    main()
