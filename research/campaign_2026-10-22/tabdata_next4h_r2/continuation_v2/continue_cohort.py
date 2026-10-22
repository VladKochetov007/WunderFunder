"""Independent family readiness, retaining the original watchdog and CPU work."""

from __future__ import annotations

import concurrent.futures as futures
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from cohort import deliver, grade, monitor, publish, verify
from delivery import PACKET, STATE, UNIT, approve_profile, read, run_job_phase, write

LOCK = threading.Lock()
GRADING = threading.Lock()
PROFILES: dict[str, Any] = {}
ERRORS: dict[str, str] = {}
ADMITTED: dict[str, list[str]] = {}


def wait_for_calibration(job: dict[str, Any], start: float) -> None:
    """Reuse healthy in-flight calibration without changing its source or RNG."""
    path = STATE / "numeric/preprocessing_r2_e3" / f"f{job['fold']}.json"
    while not path.exists():
        if time.time() > start + 1800:
            raise TimeoutError("Original full-fold calibration cutoff reached")
        time.sleep(2)
    value = read(path)
    expected = next(row for row in read(STATE / "INPUTS.json") if row["fold"] == job["fold"])
    if value["split_sha256"] != expected["split_sha256"]:
        raise ValueError("Calibration source fold differs")
    if len(value["fit_members"]) != expected["fit_train"] + expected["fit_valid"]:
        raise ValueError("Calibration fitting membership is incomplete")
    # The unchanged canonical profile/fit artifact loader verifies all arrays,
    # exact fitting members, source-code hashes and the deterministic artifact hash.


def profile(job: dict[str, Any], start: float) -> dict[str, Any]:
    """Admit numeric profiles only after their own fold's artifact is complete."""
    if job["family"] == "numeric":
        wait_for_calibration(job, start)
    return approve_profile(job, run_job_phase(job, "profile", start + 3600))


def family(name: str, jobs: list[dict[str, Any]], start: float) -> None:
    """Deliver one fixed family without waiting for another family's preparation."""
    passed, failed = {}, {}
    with futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        pending = {pool.submit(profile, job, start): job for job in jobs}
        for done in futures.as_completed(pending):
            job = pending[done]
            try:
                passed[job["id"]] = done.result()
            except Exception as error:
                failed[job["id"]] = repr(error)
    totals = {}
    for job in jobs:
        if job["id"] in passed:
            totals[job["gpu_uuid"]] = (
                totals.get(job["gpu_uuid"], 0) + passed[job["id"]]["memory"]["peak_reserved_bytes"]
            )
    for job in jobs:
        if (
            job["id"] in passed
            and totals[job["gpu_uuid"]] >= 0.9 * passed[job["id"]]["memory"]["total_bytes"]
        ):
            failed[job["id"]] = "Co-resident measured allocator peaks exceed ninety percent VRAM"
    admitted = [] if failed else jobs
    with LOCK:
        PROFILES.update(passed)
        ERRORS.update(failed)
        ADMITTED[name] = [j["id"] for j in admitted]
        write(STATE / "PROFILES.json", {"results": PROFILES, "failures": ERRORS})
        write(
            STATE / "ADMISSION.json",
            {
                "families": ADMITTED,
                "admitted": [j for ids in ADMITTED.values() for j in ids],
                "profile_failures": ERRORS,
                "original_deadline_unix": start + 14400,
                "partial_endpoints_allowed": True,
                "partial_is_quality_endpoint": False,
            },
        )
    publish()
    results = []
    with futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for done in futures.as_completed([pool.submit(deliver, job, start) for job in admitted]):
            results.append(done.result())
            write(
                STATE / f"PROGRESS_{name}.json", {"results": results, "observed_unix": time.time()}
            )
            with GRADING:
                grade(start)
            publish()
    write(STATE / f"COMPLETE_{name}.json", {"results": results, "profile_failures": failed})


def unit_properties() -> dict[str, str]:
    """Read the original independently enforced unit identity and timer."""
    names = (
        "InvocationID",
        "ActiveEnterTimestampMonotonic",
        "RuntimeMaxUSec",
        "TimeoutStopUSec",
        "KillMode",
        "Restart",
        "MainPID",
    )
    output = subprocess.check_output(
        ["systemctl", "--user", "show", UNIT, *["-p" + n for n in names]], text=True, timeout=10
    )
    return dict(line.split("=", 1) for line in output.splitlines())


def watchdog(stop: threading.Event, start: float, expected: dict[str, str]) -> None:
    """Keep the original watchdog immutable and request stop on coordinator drift."""
    while not stop.wait(5):
        try:
            actual = unit_properties()
            status = Path("/proc/840559/status").read_text()
            invalid = actual != expected or "\nState:\tT" not in status
        except (OSError, subprocess.SubprocessError):
            invalid = True
        if invalid or time.time() >= start + 13500:
            write(
                STATE / "CONTINUATION_WATCHDOG_STOP.json",
                {
                    "time_unix": time.time(),
                    "reason": "original timer/coordinator changed or original budget cutoff reached",
                },
            )
            subprocess.run(
                ["systemctl", "--user", "stop", "--no-block", UNIT], timeout=10, check=True
            )
            return


def execute() -> None:
    """Join the existing owned cgroup; never renew its runtime or wall deadline."""
    cg = subprocess.check_output(
        ["systemctl", "--user", "show", UNIT, "-p", "ControlGroup", "--value"], text=True
    ).strip()
    assert cg.endswith("/" + UNIT)
    Path("/sys/fs/cgroup" + cg + "/cgroup.procs").write_text(str(os.getpid()))
    assert "/" + UNIT in Path(f"/proc/{os.getpid()}/cgroup").read_text()
    old_main = int(
        subprocess.check_output(
            ["systemctl", "--user", "show", UNIT, "-p", "MainPID", "--value"], text=True
        )
    )
    assert old_main == 840559
    assert "\nState:\tT" in Path(f"/proc/{old_main}/status").read_text()
    start = read(STATE / "START.json")["start_unix"]
    assert start == 1792693855.4927282 and time.time() < start + 11100
    plan = read(PACKET / "PLAN.json")
    verify(plan)
    properties = unit_properties()
    if properties != plan["original_unit_properties"]:
        raise ValueError("Original unit invocation/timer changed")
    write(
        STATE / "CONTINUATION_STARTED.json",
        {
            "pid": os.getpid(),
            "original_main_pid": old_main,
            "cgroup": cg,
            "original_start_unix": start,
            "deadline_unix": start + 14400,
            "reason": "Explicit WMSE/auxiliary flags corrected before first GPU work; independent family readiness; healthy CPU preprocessing retained",
        },
    )
    stop = threading.Event()
    watcher = threading.Thread(target=monitor, args=(stop,), daemon=True)
    watcher.start()
    guard = threading.Thread(target=watchdog, args=(stop, start, properties), daemon=True)
    guard.start()
    try:
        with futures.ThreadPoolExecutor(max_workers=2) as pool:
            pending = [
                pool.submit(family, name, [j for j in plan["jobs"] if j["family"] == name], start)
                for name in ("numeric", "nll")
            ]
            for done in futures.as_completed(pending):
                done.result()
        verify(plan)
        write(
            STATE / "COMPLETE.json",
            {"stage": "independent_family_cohort_terminal", "elapsed_seconds": time.time() - start},
        )
    except Exception as error:
        write(STATE / "FAILURE.json", {"error": repr(error), "time_unix": time.time()})
        raise
    finally:
        stop.set()
        watcher.join(timeout=20)
        guard.join(timeout=10)
        subprocess.run(["systemctl", "--user", "stop", "--no-block", UNIT], check=True)


def main() -> None:
    """Take an exclusive permanent claim and stop safely on any post-claim failure."""
    # An accidental second invocation must neither start duplicate fits nor stop
    # the already running controller. This marker is never automatically released.
    with (STATE / "CONTINUATION_ONESHOT.json").open("x") as claim:
        json.dump(
            {
                "pid": os.getpid(),
                "time_unix": time.time(),
                "plan_sha256": os.environ["TABDATA_PLAN_SHA256"],
            },
            claim,
        )
    try:
        outputs = list((STATE / "nll_project/artifacts/runs").glob("*"))
        outputs += list((STATE / "numeric/runs_r2_e3").glob("*"))
        outputs += list((STATE / "logs").glob("*/profile.log"))
        if outputs:
            raise FileExistsError(
                "Scientific/profile outputs already exist; no duplicate admission"
            )
        execute()
    except BaseException as error:
        write(STATE / "CONTINUATION_FAILURE.json", {"error": repr(error), "time_unix": time.time()})
        subprocess.run(["systemctl", "--user", "stop", "--no-block", UNIT], check=True, timeout=10)
        raise


if __name__ == "__main__":
    main()
