"""Run only the frozen 120-cell feature panel beside the preserved legacy owners."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from feature_common import event, read, sha, verify_config, verify_source, write
from feature_resources import snapshot
from panel import DensityGate, OwnerGate, next_admissions, validate_panel

FAMILY_NAMES = {"field_age": "age", "causal_innovation": "innovation"}
ARM_NAMES = {
    "group_age": "group",
    "field_age": "field",
    "matched_moments": "moments",
    "moments_plus_innovation": "innovations",
}


def normalized(raw: dict) -> dict:
    """Keep original scientific IDs while using short internal family names."""
    return dict(
        raw,
        family=FAMILY_NAMES[raw["family"]],
        arm=ARM_NAMES[raw["arm"]],
        source_family=raw["family"],
        source_arm=raw["arm"],
        cpus=raw.get("cpu_ids"),
    )


def fsync_directory(path: Path) -> None:
    """Make an atomically replaced receipt name durable on its filesystem."""
    directory_fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def persist_resource_snapshot(state: Path, resource: dict) -> tuple[dict | None, str | None]:
    """Persist the observation before making it eligible to authorize admission."""
    write(state / "RESOURCE.json", resource)
    if resource.get("verified") is not True:
        return None, None
    good_path = state / "RESOURCE_GOOD.json"
    write(good_path, resource)
    fsync_directory(good_path.parent)
    return resource, sha(good_path)


def spawn(argv: list[str], cpus: list[int], log: Path, env: dict) -> dict:
    """Create one owned process group with its exact affinity and durable stdout."""
    handle = log.open("x")
    try:
        child = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=lambda: os.sched_setaffinity(0, set(cpus)),
        )
    except BaseException:
        handle.close()
        raise
    return dict(process=child, log=handle, started=time.time(), cpus=cpus)


def spawn_grade(
    argv: list[str],
    cpus: list[int],
    log: Path,
    env: dict,
    key: str,
    state: Path,
    failed: dict,
    attempted: set[str],
) -> dict | None:
    """Record a grade-launch failure without interrupting healthy fit workers."""
    try:
        return spawn(argv, cpus, log, env)
    except Exception as error:
        attempted.add(key)
        failed["grade:" + key] = "grade launch failed; no fit repeated: " + str(error)
        event(state, "grade_launch_failed", pair=key, error_type=type(error).__name__)
        return None


def terminate(task: dict) -> None:
    """Terminate only this controller's still-live child process group and reap it."""
    child = task["process"]
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def run(config_path: Path, config_sha: str) -> None:
    """Own fit/export workers, independent CPU grades and finite admission budgets."""
    config = verify_config(config_path, config_sha)
    root = Path(config["root"])
    panel = config["panel"]
    validate_panel(panel)
    source = verify_source(root)
    density = config["density_gate"]
    smoke = config["smoke_gate"]
    if (
        sha(Path(smoke["path"])) != smoke["sha256"]
        or read(Path(smoke["path"]))["passed"] is not True
    ):
        raise ValueError("Exact completed fit/export/official-grade smoke gate is missing")
    if (
        sha(Path(density["path"])) != density["sha256"]
        or read(Path(density["path"]))["passed"] is not True
    ):
        raise ValueError("Exact accepted feature-density gate is missing")
    gate = DensityGate(True, True, density["sha256"], 5, frozenset(density["variants"]))
    state = root / "state/parallel-feature-execution-v1"
    state.mkdir(exist_ok=False)
    lock = (state / "owner.lock").open("x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for name in ("claims", "work", "logs", "endpoints", "scored", "failures", "grades"):
        (state / name).mkdir()
    write(state / "PANEL.json", panel, fresh=True)
    write(state / "SOURCE.json", source, fresh=True)
    event(state, "controller_started", config_sha256=config_sha, total_cells=120)
    pending = {job["id"] for job in panel["jobs"]}
    active, grading, exports, failed, attempted_grades = {}, {}, {}, {}, set()
    blocked_families = set()
    deadline = config["absolute_deadline_unix"]
    packet = config_path.parent
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    last_resource = 0.0
    last_good = None
    last_good_sha = None
    admission_open = True
    try:
        while active or grading or (pending and admission_open):
            now = time.time()
            for name, task in list(active.items()):
                process = task["process"]
                if (
                    process.poll() is None
                    and now - task["started"] > config["per_fit_max_hours"] * 3600
                ):
                    terminate(task)
                    event(state, "fit_deadline", job=task["job"])
                if process.poll() is None:
                    continue
                task["log"].close()
                job = task["job"]
                if process.returncode == 0:
                    try:
                        result = read(state / "work" / name / "EXPORTED.json")
                        if result["job"] != job or result["status"] != "fit_export_completed":
                            raise ValueError("Worker returned a different exported cell")
                        exports[name] = result
                        write(state / "endpoints" / (name + ".json"), result, fresh=True)
                        event(
                            state,
                            "fit_export_completed",
                            job=job,
                            endpoint_sha256=sha(state / "endpoints" / (name + ".json")),
                        )
                    except (OSError, ValueError, KeyError) as error:
                        failed[name] = type(error).__name__ + ": " + str(error)
                else:
                    failed[name] = "worker exit " + str(process.returncode)
                if name in failed:
                    blocked_families.add(job["family"])
                    event(state, "fit_failed", job=job, reason=failed[name])
                del active[name]
            for key, task in list(grading.items()):
                process = task["process"]
                if process.poll() is None and now - task["started"] > 3900:
                    terminate(task)
                if process.poll() is None:
                    continue
                task["log"].close()
                event(state, "grade_finished", pair=key, returncode=process.returncode)
                if process.returncode != 0:
                    failed["grade:" + key] = "grade failure; no fit repeated"
                del grading[key]
            if now - last_resource > 10:
                try:
                    resource = snapshot(root)
                    last_good, last_good_sha = persist_resource_snapshot(state, resource)
                    if last_good is None:
                        event(
                            state,
                            "resource_admission_deferred",
                            errors=resource.get("errors", []),
                        )
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    last_good = None
                    last_good_sha = None
                    event(state, "resource_observation_failed", error_type=type(error).__name__)
                last_resource = now
            if now > deadline - config["per_fit_max_hours"] * 3600 and admission_open:
                admission_open = False
                event(
                    state,
                    "admission_closed",
                    reason="Insufficient remaining full-fit runway",
                    pending=len(pending),
                )
            healthy = (
                last_good is not None
                and last_good_sha is not None
                and now - last_good["time_unix"] <= 25
            )
            if healthy:
                try:
                    healthy = sha(state / "RESOURCE_GOOD.json") == last_good_sha
                except OSError:
                    healthy = False
            if not healthy:
                last_good = None
                last_good_sha = None
            if admission_open and healthy:
                observed_sha = last_good_sha
                gates = {
                    name: OwnerGate(
                        "terminal" if done else "live",
                        True,
                        observed_sha,
                        unit_inactive=done,
                        cgroup_empty=done,
                        actors_gone=done,
                        canonical_complete=done,
                    )
                    for name, done in last_good["terminal"].items()
                }
                eligible = {
                    j["id"]
                    for j in panel["jobs"]
                    if j["id"] in pending and FAMILY_NAMES[j["family"]] not in blocked_families
                }
                placements = {
                    name: (task["job"]["gpu_index"], task["job"]["slot_id"])
                    for name, task in active.items()
                }
                plan = next_admissions(panel, eligible, placements, gates, gate)
                for item in plan:
                    job = normalized(item)
                    name = job["id"]
                    if (state / "work" / name).exists() or (
                        state / "claims" / (name + ".json")
                    ).exists():
                        raise FileExistsError("Refusing duplicate scientific admission: " + name)
                    claim = state / "claims" / (name + ".json")
                    write(claim, job, fresh=True)
                    argv = [
                        sys.executable,
                        "-B",
                        "-u",
                        str(packet / "code/feature_worker.py"),
                        "--config",
                        str(config_path),
                        "--config-sha256",
                        config_sha,
                        "--job",
                        str(claim),
                    ]
                    try:
                        task = spawn(
                            argv, job["cpus"], state / "logs" / (name + ".log"), environment
                        )
                    except OSError as error:
                        failed[name] = "Admission failed: " + str(error)
                        blocked_families.add(job["family"])
                        pending.remove(name)
                        event(state, "admission_failed", job=job, error_type=type(error).__name__)
                        continue
                    task["job"] = job
                    active[name] = task
                    pending.remove(name)
                    event(state, "fit_started", job=job, pid=task["process"].pid, argv=argv)
            pairs = {}
            for endpoint in exports.values():
                job = endpoint["job"]
                key = f"{job['family']}_f{job['fold']}_s{job['seed']}"
                pairs.setdefault(key, {})[job["arm"]] = job
            for key, pair in pairs.items():
                if key in attempted_grades or len(pair) != 2 or len(grading) >= 2:
                    continue
                control = pair.get("group", pair.get("moments"))
                candidate = pair.get("field", pair.get("innovations"))
                grade_slot = next(
                    i for i in range(2) if i not in {t["slot"] for t in grading.values()}
                )
                cpus = [160, 162] if grade_slot == 0 else [161, 163]
                argv = [
                    sys.executable,
                    "-B",
                    "-u",
                    str(packet / "code/feature_grade.py"),
                    "--config",
                    str(config_path),
                    "--config-sha256",
                    config_sha,
                    "--candidate",
                    candidate["id"],
                    "--control",
                    control["id"],
                ]
                task = spawn_grade(
                    argv,
                    cpus,
                    state / "logs" / ("grade-" + key + ".log"),
                    environment,
                    key,
                    state,
                    failed,
                    attempted_grades,
                )
                if task is None:
                    continue
                task["slot"] = grade_slot
                grading[key] = task
                attempted_grades.add(key)
                event(state, "grade_started", pair=key, pid=task["process"].pid)
            event(
                state,
                "heartbeat",
                active=list(active),
                active_grades=list(grading),
                exported=len(exports),
                scored_pairs=len(list((state / "scored").glob("*.json"))),
                pending=len(pending),
                failed=len(failed),
                admission_open=admission_open,
                resource_fresh=healthy,
            )
            if (
                pending
                and not active
                and not grading
                and all(
                    FAMILY_NAMES[j["family"]] in blocked_families
                    for j in panel["jobs"]
                    if j["id"] in pending
                )
            ):
                break
            if active or grading or (pending and admission_open):
                time.sleep(5)
    finally:
        for task in (*active.values(), *grading.values()):
            try:
                terminate(task)
            finally:
                task["log"].close()
        lock.close()
    write(
        state / "SUMMARY.json",
        dict(
            status="complete" if not pending and not failed else "partial",
            pending=sorted(pending),
            exported=len(exports),
            scored_pairs=len(list((state / "scored").glob("*.json"))),
            failures=failed,
            time_unix=time.time(),
            config_sha256=config_sha,
        ),
        fresh=True,
    )


def main() -> None:
    """Run one finite, non-restarting admission authority for these two successors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    run(args.config, args.config_sha256)


if __name__ == "__main__":
    main()
