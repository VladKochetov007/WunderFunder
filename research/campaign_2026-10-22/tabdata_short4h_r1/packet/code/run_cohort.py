from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from recipe import PACKET, ROOT, SOURCE, STATE, legacy, read, sha, verify_source, write


def check_packet() -> dict:
    """Verify the reviewed finite cohort and every selected component manifest."""
    for name, digest in read(PACKET / "MANIFEST.json")["files"].items():
        if sha(PACKET / name) != digest:
            raise ValueError("Packet changed: " + name)
    plan = read(PACKET / "PLAN.json")
    if plan["fits"] != 24 or plan["hard_wall_seconds"] != 14400:
        raise ValueError("Incorrect cohort authority")
    verify_source(ROOT)
    for name, digest in plan["dependencies"].items():
        if sha(Path(name)) != digest:
            raise ValueError("Inherited helper changed: " + name)
    for job in plan["jobs"]:
        if sha(Path(job["subset"]["path"])) != job["subset"]["sha256"]:
            raise ValueError("Fitting subset changed")
        for name, digest in job["requests"].items():
            if sha(Path(name)) != digest:
                raise ValueError("Sealed actor request changed")
        if job["gpu_index"] not in range(8) or not job["cpus"]:
            raise ValueError("Incorrect hardware allocation")
    return plan


def run_process(
    command: list[str], env: dict, cwd: Path, log: Path, deadline: float, cpus: list[int]
) -> dict:
    """Run only an owned process group and enforce the shared absolute deadline."""
    started = time.time()
    if started >= deadline:
        raise TimeoutError("No time remains for this phase")
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = ["taskset", "-c", ",".join(map(str, cpus)), *command]
    with log.open("x") as output:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = process.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)
            raise TimeoutError("Owned phase stopped at shared cutoff") from None
    record = dict(
        command=argv,
        exit_code=rc,
        started_unix=started,
        finished_unix=time.time(),
        elapsed_seconds=time.time() - started,
        log_sha256=sha(log),
    )
    if rc != 0:
        raise RuntimeError(f"Phase exit {rc}; see {log}")
    return record


def actor(job: dict, phase: str, deadline: float) -> dict:
    """Execute the existing strict-FP32 B300 actor in its declared isolated root."""
    work = STATE / ("profiles" if phase == "profile" else "work") / job["id"]
    modules = legacy(ROOT)
    inherited = modules["worker"]
    env = inherited._phase_environment(
        dict(os.environ), work=work, gpu_uuid=job["gpu_uuid"], phase=phase
    )
    package = ROOT / "runtime/pr15-host-v2"
    name = "export_cell.py" if phase == "export" else "train_cell.py"
    request = work / ("profile_request.json" if phase == "profile" else "fit_request.json")
    argv = [
        sys.executable,
        "-B",
        "-u",
        str(package / "gpu_entrypoint.py"),
        str(package / name),
        "--request",
        str(request),
    ]
    result = run_process(argv, env, SOURCE, work / "logs" / (phase + ".log"), deadline, job["cpus"])
    envelope = work / "envelope" / phase / (Path(name).stem + "_resource_receipt.json")
    receipt = read(envelope)
    if receipt["status"] != "actor_returned" or receipt["visible_devices"] != job["gpu_uuid"]:
        raise ValueError("Actor envelope identity differs")
    result["envelope_sha256"] = sha(envelope)
    if phase == "profile":
        lines = (work / "logs/profile.log").read_text(errors="replace").splitlines()
        values = [
            json.loads(line[len("PROFILE ") :]) for line in lines if line.startswith("PROFILE ")
        ]
        if not values or not math.isfinite(values[-1]["step_wall"]) or values[-1]["step_wall"] <= 0:
            raise ValueError("Missing finite representative profile timing")
        stats = [
            json.loads(line[len("GRAD_STATS ") :])
            for line in lines
            if line.startswith("GRAD_STATS ")
        ]
        if not stats or stats[-1]["nonfinite_steps"] != 0:
            raise ValueError("Missing finite profile gradients")
        result["profile"] = values[-1]
        result["forecast_fit_seconds"] = (
            result["elapsed_seconds"]
            + 1.5
            * 6
            * ((job["subset"]["train"] + job["subset"]["valid"]) // 8)
            * values[-1]["step_wall"]
            / 1000
        )
    write(work / (phase + "_phase.json"), result, fresh=True)
    return result


def deliver(job: dict, start: float) -> dict:
    """Fit once, validate the endpoint and export without reusing old science."""
    from feature_outputs import validate_export
    from validate_fit import validate

    work = STATE / "work" / job["id"]
    try:
        actor(job, "fit", start + 11100)
        fit = validate(job)
        actor(job, "export", start + 12900)
        prediction = validate_export(ROOT, work, job, fit["checkpoint_sha256"])
        endpoint = dict(
            schema="tabdata-short4h-endpoint.v1",
            status="fit_export_completed",
            job=job,
            fit=fit,
            prediction=prediction,
            fit_receipt_sha256=sha(work / "FIT.json"),
        )
        write(work / "EXPORTED.json", endpoint, fresh=True)
        return endpoint
    except Exception as error:
        write(
            work / "FAILURE.json",
            dict(error_type=type(error).__name__, error=str(error), time_unix=time.time()),
            fresh=True,
        )
        raise


def bounded_grade(candidate: dict, control: dict, start: float) -> dict:
    """Run unchanged grading in a child group under the common phase deadline."""
    pair_id = (
        f"{candidate['job']['family']}_f{candidate['job']['fold']}_s{candidate['job']['seed']}"
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    command = [
        sys.executable,
        "-B",
        str(PACKET / "code/run_cohort.py"),
        "--start-unix",
        str(start),
        "--grade",
        candidate["job"]["id"],
        control["job"]["id"],
    ]
    run_process(
        command,
        env,
        ROOT,
        STATE / "grade_process_logs" / (pair_id + ".log"),
        start + 13140,
        [48, 49, 50, 51],
    )
    receipt = read(STATE / "grades" / pair_id / "RECEIPT.json")
    try:
        run_process(
            [sys.executable, "-B", str(PACKET / "code/publish.py")],
            env,
            ROOT,
            STATE / "publication_logs" / (pair_id + ".log"),
            min(start + 13200, time.time() + 90),
            [48, 49, 50, 51],
        )
    except Exception as error:
        write(STATE / "publication_logs" / (pair_id + ".failure.json"), dict(error=repr(error)))
    return receipt


def execute(start: float) -> None:
    """Own one finite 24-slot cohort; stop admission at resource or time failures."""
    plan = check_packet()
    write(
        STATE / "START.json",
        dict(
            start_unix=start,
            deadline_unix=start + 14400,
            first_gpu_phase="profile",
            service="tabdata-short4h-20261022-r1.service",
        ),
        fresh=True,
    )
    profiles, errors = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        pending = {pool.submit(actor, job, "profile", start + 900): job for job in plan["jobs"]}
        for future in concurrent.futures.as_completed(pending):
            job = pending[future]
            try:
                profiles[job["id"]] = future.result()
            except Exception as error:
                errors[job["id"]] = repr(error)
    write(STATE / "PROFILES.json", dict(results=profiles, failures=errors), fresh=True)
    if errors:
        raise RuntimeError("Profile failures; no scientific fits admitted")
    admitted, rejected = [], []
    for i in range(0, 24, 2):
        pair = plan["jobs"][i : i + 2]
        if plan.get("allow_partial_under_hard_cutoff") or all(
            time.time() + profiles[j["id"]]["forecast_fit_seconds"] < start + 10800 for j in pair
        ):
            admitted.extend(pair)
        else:
            rejected.extend(j["id"] for j in pair)
    write(
        STATE / "ADMISSION.json",
        dict(
            admitted=[j["id"] for j in admitted],
            rejected_timing=rejected,
            profile_results_sha256=sha(STATE / "PROFILES.json"),
            observed_unix=time.time(),
        ),
        fresh=True,
    )
    endpoints, fit_errors, grades = {}, {}, {}
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool,
        concurrent.futures.ThreadPoolExecutor(max_workers=2) as grading,
    ):
        fits = {pool.submit(deliver, job, start): job for job in admitted}
        queued_grades = {}
        for future in concurrent.futures.as_completed(fits):
            job = fits[future]
            try:
                endpoints[job["id"]] = future.result()
            except Exception as error:
                fit_errors[job["id"]] = repr(error)
            partner_jobs = [
                j
                for j in admitted
                if (j["family"], j["fold"], j["seed"]) == (job["family"], job["fold"], job["seed"])
            ]
            pair_id = f"{job['family']}_f{job['fold']}_s{job['seed']}"
            if (
                len(partner_jobs) == 2
                and all(j["id"] in endpoints for j in partner_jobs)
                and pair_id not in queued_grades
            ):
                control, candidate = partner_jobs
                queued_grades[pair_id] = grading.submit(
                    bounded_grade, endpoints[candidate["id"]], endpoints[control["id"]], start
                )
            write(
                STATE / "PROGRESS.json",
                dict(
                    admitted=len(admitted),
                    exported=list(endpoints),
                    failures=fit_errors,
                    observed_unix=time.time(),
                ),
            )
        for pair_id, future in queued_grades.items():
            try:
                grades[pair_id] = future.result()
            except Exception as error:
                grades[pair_id] = dict(error=repr(error))
    verify_source(ROOT)
    write(
        STATE / "COMPLETE.json",
        dict(
            stage="bounded_cohort_terminal",
            elapsed_seconds=time.time() - start,
            admitted=len(admitted),
            exported=list(endpoints),
            failed=fit_errors,
            rejected_timing=rejected,
            grades=grades,
            full_data_fits=0,
            scientific_fit_attempts=len(admitted),
            quality_claim="exposed development short-screen only",
        ),
        fresh=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-unix", type=float, required=True)
    parser.add_argument("--grade", nargs=2)
    options = parser.parse_args()
    try:
        if options.grade:
            from feature_outputs import grade_pair

            candidate, control = [
                read(STATE / "work" / name / "EXPORTED.json") for name in options.grade
            ]
            check_packet()
            grade_pair(ROOT, STATE, candidate, control)
        else:
            execute(options.start_unix)
    except Exception as error:
        write(
            STATE
            / (
                "FAILURE.json"
                if not options.grade
                else "GRADE_FAILURE_" + options.grade[0] + ".json"
            ),
            dict(error=repr(error), time_unix=time.time()),
            fresh=True,
        )
        raise
