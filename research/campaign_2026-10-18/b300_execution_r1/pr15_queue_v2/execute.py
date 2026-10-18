"""Run only the never-started PR15 cells after the original wave drained."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import time
from pathlib import Path

from fit_validation import (
    _canonical_argv,
    _check_checkpoint,
    _check_contract,
    _check_export_and_score,
    validate_full_fit,
)
from journal import Journal, write_json_fsync
from process_start import Running, start_owned, terminate_owned
from runtime import _monitor
from runtime_gates import cgroup_snapshot, require_gpus
from validation import validate_fit_artifacts, validate_prediction_and_score

ROOT = Path("/data/epanov_s/tabdata")
PACKET = ROOT / "runtime/pr15-host-v2"
STATE = ROOT / "state/pr15-execution-v2"
LOGGER = logging.getLogger(__name__)
PARENT_SHA = "950f5178652ad81ae62220a124dc0253cf5f80f9b1dfef7250f39aa1d0c9f1c8"


def sha(path: Path) -> str:
    """Hash an exact immutable input file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_queue(expected_manifest: str) -> tuple[list[dict], list[dict], float]:
    """Prove old admission is stopped and every original cell has one owner."""
    manifest_path = PACKET / "QUEUE_EXECUTION_MANIFEST.json"
    if sha(manifest_path) != expected_manifest:
        raise ValueError("Successor execution manifest differs")
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files"].items():
        if sha(PACKET / relative) != expected:
            raise ValueError("Successor file changed: " + relative)
    if sha(ROOT / "runtime/pr15-host-v1/EXECUTION_MANIFEST.json") != PARENT_SHA:
        raise ValueError("Original execution packet changed")
    old = ROOT / "state/pr15-execution-v1"
    if sha(old / "PANEL.json") != manifest["original_panel_sha256"]:
        raise ValueError("Original panel changed")
    unit = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                "tabdata-pr15-execution-v1",
                "-p",
                "MainPID",
                "-p",
                "ActiveState",
                "-p",
                "ControlGroup",
            ],
            text=True,
        ).splitlines()
    )
    if unit["MainPID"] != "0" or unit["ActiveState"] not in {"failed", "inactive"}:
        raise ValueError("Original controller has not drained")
    if unit["ControlGroup"] and unit["ControlGroup"] != manifest["original_controller_cgroup"]:
        raise ValueError("Original unit cgroup differs from the pinned unit")
    cgroup = Path("/sys/fs/cgroup") / manifest["original_controller_cgroup"].lstrip("/")
    if cgroup.exists() and any(path.read_text().strip() for path in cgroup.rglob("cgroup.procs")):
        raise ValueError("Original unit still has live process-group members")
    old_worker = str(ROOT / "runtime/pr15-host-v1/worker.py").encode()
    old_gpu_wrapper = str(ROOT / "runtime/pr15-host-v1/gpu_entrypoint.py").encode()
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            tokens = (process / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (old_worker in tokens or old_gpu_wrapper in tokens) and any(
            token.startswith(str(old / "work").encode() + b"/") for token in tokens
        ):
            raise ValueError("An original worker or GPU actor remains alive")
    events = [json.loads(line) for line in (old / "events.jsonl").read_text().splitlines()]
    started = [row["job"] for row in events if row.get("event") == "fit_started"]
    jobs = json.loads((old / "PANEL.json").read_text())["jobs"]
    by_id = {job["id"]: job for job in jobs}
    claimed = {job["id"] for job in started}
    if len(jobs) != 120 or len(by_id) != 120 or len(started) != 16 or len(claimed) != 16:
        raise ValueError("Original roster is not the expected 120/16 split")
    if any(job != by_id.get(job["id"]) for job in started):
        raise ValueError("An original start differs from its frozen cell")
    for job in started:
        endpoint = ROOT / "state/pr15-delivery-recovery-v2" / job["id"] / "ENDPOINT.json"
        record = json.loads(endpoint.read_text())
        if (
            record["status"] != "exported_and_graded"
            or record["job"] != job
            or not record["training_not_repeated"]
        ):
            raise ValueError("An original full fit lacks its recovered endpoint")
        score = record["score"]
        work = old / "work" / job["id"]
        expected_prediction = work / "cache" / ("pred_" + job["id"] + "_valid.npy")
        expected_score = work / "artifacts/runs" / job["id"] / "score.json"
        if (
            Path(score["prediction_path"]) != expected_prediction
            or Path(score["score_path"]) != expected_score
        ):
            raise ValueError("Recovered score paths escape the exact original workspace")
        if (
            sha(Path(score["prediction_path"])) != score["prediction_sha256"]
            or sha(Path(score["score_path"])) != score["score_sha256"]
        ):
            raise ValueError("Recovered original endpoint changed")
        split_path = ROOT / f"splits/fold{job['fold']}.json"
        split_hash = sha(split_path)
        source = ROOT / "runtime/pr15-source-e238-v1"
        request = json.loads((work / "worker_request.json").read_text())
        for phase, key in (("train", "train_argv"), ("predict", "predict_argv")):
            expected_argv = _canonical_argv(
                source, job["id"], job["fold"], job["seed"], job["control"], split_path, phase
            )
            if request.get(key) != expected_argv:
                raise ValueError("Original recovered request has recipe drift")
        fit = _check_checkpoint(
            work,
            job["id"],
            job["fold"],
            job["seed"],
            job["control"],
            split_hash,
            source,
            _check_contract(),
        )
        current_score = validate_prediction_and_score(
            work, job["id"], json.loads(split_path.read_text()), split_hash
        )
        endpoint_check = _check_export_and_score(
            work,
            job["id"],
            split_hash,
            fit["checkpoint_sha256"],
            json.loads(split_path.read_text())["evaluation"]["valid"],
            fit["optimizer_completed_updates"],
            {"score": current_score},
        )
        if current_score != score or endpoint_check != record["independent_endpoint_check"]:
            raise ValueError("Original full-fit or official endpoint differs")
        recovery_path = endpoint.parent / "FIT_RECOVERY.json"
        if sha(recovery_path) != record["fit_recovery_sha256"]:
            raise ValueError("Original fit recovery evidence changed")
        recovery = json.loads(recovery_path.read_text())
        original_failure = work / "worker_receipt.json"
        original_envelope = work / "envelope/fit/train_cell_resource_receipt.json"
        if (
            recovery.get("job") != job
            or recovery.get("original_failure_receipt_sha256") != sha(original_failure)
            or recovery.get("original_resource_receipt_sha256") != sha(original_envelope)
        ):
            raise ValueError("Original recovered fit lost its execution provenance")
        resource = json.loads(original_envelope.read_text())
        if (
            resource.get("status") != "actor_returned"
            or resource.get("visible_devices") != job["gpu_uuid"]
            or resource.get("cpu_affinity") != job["cpus"]
            or resource.get("allocator_budget_bytes") != 6 * 1024**3
        ):
            raise ValueError("Original fit device or CPU assignment differs")
    pending = [job for job in jobs if job["id"] not in claimed]
    for job in pending:
        paths = (old / "work" / job["id"], STATE / "work" / job["id"])
        if any(path.exists() or path.is_symlink() for path in paths):
            raise ValueError("A pending cell already has work evidence: " + job["id"])
    if len(pending) != 104 or any(
        sum(j["arm"] == arm for j in pending) != 26
        for arm in ("baseline", "moments", "dictionary", "combined")
    ):
        raise ValueError("Remaining panel is not 26 cells per original arm")
    return jobs, pending, float(manifest["original_absolute_deadline_unix"])


def launch(job: dict) -> Running:
    """Launch the corrected orchestration around the unchanged canonical trainer."""
    require_gpus([job], job["gpu_uuid"])
    work = STATE / "work" / job["id"]
    argv = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(PACKET / "worker.py"),
        "--root",
        str(ROOT),
        "--name",
        job["id"],
        "--fold",
        str(job["fold"]),
        "--seed",
        str(job["seed"]),
        "--control",
        job["control"],
        "--gpu",
        job["gpu_uuid"],
        "--cpus",
        ",".join(map(str, job["cpus"])),
        "--work",
        str(work),
        "--mode",
        "fit",
    ]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=job["gpu_uuid"],
        WNN_CWD=str(ROOT),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(ROOT / "runtime/pr15-source-e238-v1/scripts"),
    )
    return start_owned(
        name=job["id"],
        argv=argv,
        environment=env,
        cpus=tuple(job["cpus"]),
        log_path=STATE / "logs" / (job["id"] + ".log"),
        timeout_seconds=12 * 3600,
    )


def check_initial_placement() -> None:
    """Require space for the measured two-R01/two-PR15 composition before admission."""
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=30,
    )
    r01_counts: dict[str, int] = {}
    r01_wrapper = str(ROOT / "runtime/b300-r01-v1/code/gpu_entrypoint.py").encode()
    for line in rows.splitlines():
        uuid, pid = [field.strip() for field in line.split(",")]
        try:
            tokens = (Path("/proc") / pid / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        if r01_wrapper in tokens:
            r01_counts[uuid] = r01_counts.get(uuid, 0) + 1
        elif any(token.startswith(str(ROOT / "runtime").encode() + b"/") for token in tokens):
            raise ValueError("Unexpected campaign CUDA actor before PR15 admission")
    if any(count > 2 for count in r01_counts.values()):
        raise ValueError("R01 placement exceeds the measured two actors per GPU")


def delivered(job: dict) -> dict:
    """Validate the complete real fit, fixed predictions and unchanged official score."""
    work = STATE / "work" / job["id"]
    receipt_path = work / "worker_receipt.json"
    payload = json.loads(receipt_path.read_text())
    expected = dict(
        schema="pr15-host-worker-receipt.v1",
        status="fit_export_grade_completed",
        mode="fit",
        name=job["id"],
        fold=job["fold"],
        seed=job["seed"],
        control=job["control"],
        cpu_affinity=job["cpus"],
    )
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed worker does not match its queued cell")
    if payload["gpu"]["uuid"] != job["gpu_uuid"] or payload["source"]["manifest_sha256"] != sha(
        PACKET / "source_manifest.json"
    ):
        raise ValueError("Completed worker source/device differs")
    if payload["request_sha256"] != sha(work / "worker_request.json"):
        raise ValueError("Worker request changed")
    split = ROOT / f"splits/fold{job['fold']}.json"
    if payload["split_hash"] != sha(split):
        raise ValueError("Fold membership changed")
    for phase in ("fit", "export", "score"):
        if Path(payload[phase + "_receipt"]) != work / (phase + "_receipt.json"):
            raise ValueError("Phase receipt escaped the worker directory")
    full = validate_full_fit(work, job["id"], job["fold"], job["seed"], job["control"], sha(split))
    fit = validate_fit_artifacts(work, job["id"])
    score = validate_prediction_and_score(
        work, job["id"], json.loads(split.read_text()), sha(split)
    )
    if score != payload["score"]:
        raise ValueError("Native score changed after the worker receipt")
    return dict(
        job=job,
        full_fit_contract=full,
        fit=fit,
        score=score,
        worker_receipt_sha256=sha(receipt_path),
    )


def run(jobs: list[dict], pending: list[dict], deadline_unix: float) -> None:
    """Drain 104 never-started cells with fail-closed, full-budget delivery."""
    STATE.mkdir(exist_ok=False)
    journal = Journal(STATE / "events.jsonl")
    write_json_fsync(
        STATE / "PANEL.json",
        dict(jobs=jobs, successor_jobs=pending, already_completed_original_fits=16),
    )
    deadline = time.monotonic() + deadline_unix - time.time()
    queues = {
        (gpu, lane): [job for job in pending if (job["gpu_index"], job["lane"]) == (gpu, lane)]
        for gpu in range(8)
        for lane in range(2)
    }
    active, processed, task_lanes = [], set(), {}
    admission = {"open": True}
    baseline = cgroup_snapshot()["memory_events"]

    def start_next(key: tuple[int, int]) -> None:
        if not admission["open"] or not queues[key]:
            return
        if deadline - time.monotonic() < 12 * 3600:
            admission["open"] = False
            journal.record("admissions_closed", reason="Inherited campaign runway below12hours")
            return
        job = queues[key].pop(0)
        work = STATE / "work" / job["id"]
        try:
            if work.exists() or work.is_symlink():
                raise FileExistsError(work)
            journal.record("spawn_intent", job=job)
            task = launch(job)
        except Exception as error:
            LOGGER.exception("Closing admission after spawn failure; draining healthy fits")
            admission["open"] = False
            journal.record("admission_failed", job=job, error=str(error))
            return
        active.append(task)
        task_lanes[task.name] = key
        journal.record("fit_started", job=job, pid=task.process.pid)

    def advance(running: list, results: dict) -> None:
        valid = []
        for name, result in list(results.items()):
            if name in processed:
                continue
            processed.add(name)
            try:
                if result["status"] != "completed" or result["returncode"] != 0:
                    raise RuntimeError("Worker process failed")
                job = next(job for job in pending if job["id"] == name)
                endpoint = delivered(job)
                write_json_fsync(STATE / "endpoints" / (name + ".json"), endpoint)
                valid.append(name)
            except Exception as error:
                LOGGER.exception("Closing admission; preserving healthy fits")
                admission["open"] = False
                journal.record("admissions_closed", failed=name, result=result, reason=str(error))
        for name in valid:
            start_next(task_lanes[name])

    try:
        for key in queues:
            start_next(key)
        results = _monitor(active, journal, ROOT, deadline, baseline, [STATE], advance=advance)
    except BaseException:
        for task in active:
            if task.status == "running":
                terminate_owned(task.process)
        raise
    write_json_fsync(STATE / "RESULTS.json", results)
    if (
        len(results) != 104
        or any(record["status"] != "completed" for record in results.values())
        or len(list((STATE / "endpoints").glob("*.json"))) != 104
    ):
        raise RuntimeError("Successor incomplete; no implicit refit")
    write_json_fsync(
        STATE / "COMPLETE.json", dict(completed_successor_cells=104, preserved_original_fits=16)
    )


def main() -> None:
    """Verify a separately pinned corrected packet before starting any fit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    jobs, pending, deadline = verified_queue(args.manifest_sha256)
    snapshot = cgroup_snapshot()
    if (
        snapshot["memory_max"] != 192 * 1024**3
        or snapshot["swap_max"] != 0
        or snapshot["pids_max"] != 512
        or snapshot["cpu_affinity"] != list(range(16, 48))
    ):
        raise ValueError("Successor cgroup or CPU envelope differs")
    if args.preflight_only:
        print(
            json.dumps(
                dict(
                    status="preflight_passed",
                    pending=len(pending),
                    inherited_deadline_unix=deadline,
                )
            )
        )
        return
    check_initial_placement()
    run(jobs, pending, deadline)


if __name__ == "__main__":
    main()
