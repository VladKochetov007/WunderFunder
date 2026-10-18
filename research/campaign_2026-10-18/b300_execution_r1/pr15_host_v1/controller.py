from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKET = Path(__file__).resolve().parent
sys.path[:0] = [str(PACKET), str(ROOT / "runtime/b300-r01-v1/control")]
from journal import Journal, write_json_fsync
from process_start import start_owned, terminate_owned
from runtime_gates import cgroup_snapshot
from validation import validate_fit_artifacts, validate_prediction_and_score

from runtime import _monitor, run_cohort

LOGGER = logging.getLogger(__name__)

ARMS = ("baseline", "moments", "dictionary", "combined")
CONTROLS = ("all", "moments_only", "dictionary_only", "none")
UUIDS = (
    "GPU-daf615f5-ea55-860a-bfe1-41c36ab96c3e",
    "GPU-25ae492c-db98-e1ae-0f5e-c56365a422c7",
    "GPU-df6fe7d0-64e5-7af2-467d-c1699cb6fc67",
    "GPU-76951d88-add3-0d6b-5bee-c0390915ca47",
    "GPU-e6d65cb1-aba4-14f5-d4e5-bf24ef1ee3d8",
    "GPU-1fc4d921-dc78-6711-c2a8-ec0589b7f106",
    "GPU-2913ac69-eccb-63b0-1112-e09ea9598b57",
    "GPU-be85f55c-71a1-f389-eeed-55c63a0e3ce7",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_identity() -> str:
    """Verify the exact executable packet before trusting prior evidence."""
    manifest_path = PACKET / "FROZEN.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        if digest(PACKET / name) != expected:
            raise ValueError(f"Frozen PR15 runtime changed: {name}")
    return digest(manifest_path)


def cpu_pair(gpu: int, lane: int) -> tuple[int, int]:
    start = 16 + (gpu % 4) * 8 + lane * 4 + (gpu // 4)
    return start, start + 2


def panel() -> list[dict]:
    """Freeze four original arms across thirty paired fold/seed cells."""
    jobs = []
    for index in range(30):
        fold, seed = index % 4, 1001 + index // 4
        for arm_index, (arm, control) in enumerate(zip(ARMS, CONTROLS, strict=True)):
            position = fold * 4 + arm_index
            gpu, lane = (position + index // 4) % 8, position // 8
            jobs.append(
                {
                    "id": f"pr15_b300_v1_{arm}_f{fold}_s{seed}",
                    "pair_index": index,
                    "fold": fold,
                    "seed": seed,
                    "arm": arm,
                    "control": control,
                    "gpu_index": gpu,
                    "gpu_uuid": UUIDS[gpu],
                    "lane": lane,
                    "cpus": list(cpu_pair(gpu, lane)),
                }
            )
    assert len(jobs) == len({j["id"] for j in jobs}) == 120
    assert len({(j["gpu_index"], j["lane"]) for j in jobs}) == 16
    return jobs


def launch(job: dict, state: Path, mode: str):
    """Start one isolated canonical actor, with no change to existing R01."""
    name = job["id"]
    work = state / "work" / name
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": job["gpu_uuid"],
        "WNN_CWD": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    argv = [
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(PACKET / "worker.py"),
        "--root",
        str(ROOT),
        "--name",
        name,
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
        mode,
    ]
    return start_owned(
        name=name,
        argv=argv,
        environment=env,
        cpus=tuple(job["cpus"]),
        log_path=state / "logs" / f"{name}.log",
        timeout_seconds=1800 if mode == "profile" else 12 * 3600,
    )


def profile_record(state: Path, job: dict, result: dict) -> dict:
    """Require actual finite update and strict precision evidence."""
    if result["status"] != "completed" or result["returncode"] != 0:
        raise RuntimeError(f"Profile failed: {job['id']} {result}")
    text = (state / "logs" / f"{job['id']}.log").read_text()

    def record(label: str) -> dict:
        lines = [
            line[len(label) + 1 :]
            for line in text.splitlines()
            if line.startswith(label + " ")
        ]
        if len(lines) != 1:
            raise ValueError(f"Expected one {label}, got {len(lines)}")
        return json.loads(lines[0])

    timing = record("PROFILE")
    updates = record("PROFILE_UPDATE_EVIDENCE")
    resource = record("PROFILE_RESOURCE_EVIDENCE")
    envelope = record("PROCESS_ENVELOPE")
    if (
        envelope["visible_devices"] != job["gpu_uuid"]
        or envelope["cpu_affinity"] != job["cpus"]
        or envelope["allocator_budget_bytes"] != 6 * 1024**3
    ):
        raise ValueError("Profile process resource binding differs from its slot")
    if (
        not updates["valid_for_throughput"]
        or updates["optimizer_completed_updates"] != 100
    ):
        raise ValueError(f"Expected 100 finite completed chunk updates: {updates}")
    if (
        resource["precision"] != "fp32"
        or resource["matmul_tf32"]
        or resource["cudnn_tf32"]
        or resource["float32_matmul_precision"] != "highest"
    ):
        raise ValueError("Strict FP32 profile contract failed")
    if not timing["seq_per_s"] > 0:
        raise ValueError("Invalid sequence throughput")
    return {
        "job": job,
        "process": result,
        "timing": timing,
        "updates": updates,
        "resource": resource,
        "log_sha256": digest(state / "logs" / f"{job['id']}.log"),
    }


def profiles(state: Path, journal: Journal, deadline: float) -> None:
    """Measure 1 extra versus 2 extras with the existing fit always running."""
    receipts = []
    for gpu in (7, 2):
        stages = []
        for count in (1, 2):
            jobs = [
                {
                    "id": f"density_g{gpu}_extra{count}_lane{lane}",
                    "fold": 0,
                    "seed": 1001,
                    "arm": "combined",
                    "control": "none",
                    "gpu_index": gpu,
                    "gpu_uuid": UUIDS[gpu],
                    "lane": lane,
                    "cpus": list(cpu_pair(gpu, lane)),
                }
                for lane in range(count)
            ]
            results = run_cohort(
                jobs,
                lambda j: launch(j, state, "profile"),
                journal=journal,
                root=ROOT,
                global_deadline=deadline,
                owned_paths=[state],
            )
            gates = [profile_record(state, job, results[job["id"]]) for job in jobs]
            stages.append(gates)
            write_json_fsync(state / "receipts" / f"g{gpu}_extra{count}.json", gates)
        gain = (
            sum(g["timing"]["seq_per_s"] for g in stages[1])
            / stages[0][0]["timing"]["seq_per_s"]
        )
        receipts.append(
            {"gpu_index": gpu, "added_process_aggregate_gain": gain, "stages": stages}
        )
        if gain < 1.10:
            raise RuntimeError(
                f"Two extra workers do not improve added throughput enough: {gain}"
            )
    write_json_fsync(
        state / "DENSITY_PASS.json",
        {
            "schema": "tabdata-pr15-density.v1",
            "time_unix": time.time(),
            "frozen_identity_sha256": frozen_identity(),
            "existing_r01_preserved": True,
            "profile_batches": {"warmup": 2, "timed": 8},
            "optimizer_attempts_per_profile": 100,
            "results": receipts,
            "interpretation": "Throughput of the added workers; existing R01 rate not directly measured.",
        },
    )


def validate_delivery(state: Path, job: dict) -> dict:
    """Independently bind a delivered endpoint to its exact queued scientific cell."""
    work = state / "work" / job["id"]
    payload = json.loads((work / "worker_receipt.json").read_text())
    expected = {
        "schema": "pr15-host-worker-receipt.v1",
        "status": "fit_export_grade_completed",
        "mode": "fit",
        "name": job["id"],
        "fold": job["fold"],
        "seed": job["seed"],
        "control": job["control"],
        "cpu_affinity": job["cpus"],
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("Worker receipt does not match the exact queued cell")
    if (
        payload["gpu"]["uuid"] != job["gpu_uuid"]
        or payload["source"]["manifest_sha256"]
        != digest(PACKET / "source_manifest.json")
        or payload["request_sha256"] != digest(work / "worker_request.json")
        or payload["split_hash"] != digest(ROOT / "splits" / f"fold{job['fold']}.json")
    ):
        raise ValueError("Worker source/request/split/GPU binding differs")
    for phase in ("fit", "export", "score"):
        if Path(payload[f"{phase}_receipt"]) != work / f"{phase}_receipt.json":
            raise ValueError("Phase receipt escapes the owned worker directory")
    from fit_validation import validate_full_fit

    full_fit = validate_full_fit(
        work, job["id"], job["fold"], job["seed"], job["control"], payload["split_hash"]
    )
    fit = validate_fit_artifacts(work, job["id"])
    split = json.loads((ROOT / "splits" / f"fold{job['fold']}.json").read_text())
    score = validate_prediction_and_score(work, job["id"], split, payload["split_hash"])
    if score != payload["score"]:
        raise ValueError(
            "Delivered score or prediction bytes changed after worker receipt"
        )
    return {
        "job": job,
        "fit": fit,
        "score": score,
        "worker_receipt_sha256": digest(work / "worker_receipt.json"),
        "full_fit_contract": full_fit,
    }


def fits(state: Path, journal: Journal, deadline: float) -> None:
    """Drain all 120 original cells through sixteen disjoint additional slots."""
    contract = json.loads((PACKET / "FIT_CONTRACT.json").read_text())
    if (
        contract.get("schema") != "tabdata-pr15-fit-contract.v1"
        or set(contract.get("files", {})) != {"fit_validation.py"}
        or contract.get("profile_packet_sha256") != frozen_identity()
    ):
        raise ValueError("Fit-only checkpoint/export contract is absent")
    for relative, expected in contract["files"].items():
        if digest(PACKET / relative) != expected:
            raise ValueError("Fit-only validation code changed")
    gate = ROOT / "state/pr15-density-v4/DENSITY_PASS.json"
    density = json.loads(gate.read_text())
    if (
        density.get("schema") != "tabdata-pr15-density.v1"
        or density.get("frozen_identity_sha256") != frozen_identity()
        or not 0 <= time.time() - density["time_unix"] < 12 * 3600
        or {r["gpu_index"] for r in density["results"]} != {2, 7}
    ):
        raise ValueError("Density evidence is stale or not bound to this packet")
    slowest_step = 0.0
    for result in density["results"]:
        if len(result["stages"]) != 2 or list(map(len, result["stages"])) != [1, 2]:
            raise ValueError("Density evidence must contain one- and two-extra stages")
        for stage in result["stages"]:
            for previous in stage:
                actual = profile_record(
                    gate.parent, previous["job"], previous["process"]
                )
                if actual != previous:
                    raise ValueError("Density log/evidence changed after measurement")
        single = result["stages"][0][0]["timing"]["seq_per_s"]
        combined = sum(r["timing"]["seq_per_s"] for r in result["stages"][1])
        if combined / single < 1.10:
            raise ValueError("Added-worker throughput gain is insufficient")
        slowest_step = max(
            slowest_step,
            *(r["timing"]["step_wall"] / 1000 for r in result["stages"][1]),
        )
    # 44,850 is the largest original three-epoch optimizer-update budget.
    # Ten TBPTT chunks per batch; profile wall values are milliseconds.
    job_budget = 4485 * slowest_step * 1.5 + 3600
    if job_budget > 12 * 3600:
        raise ValueError("Measured full-fit/export forecast exceeds the per-job cap")
    if 8 * job_budget > deadline - time.monotonic():
        raise ValueError(
            "Full panel cannot fit the conservative eight-job lane forecast"
        )
    write_json_fsync(
        state / "FORECAST.json",
        {
            "job_budget_seconds": job_budget,
            "max_jobs_per_lane": 8,
            "lane_forecast_seconds": 8 * job_budget,
            "existing_r01_rate_not_directly_measured": True,
            "profile_to_full_fit_is_provisional": True,
        },
    )
    jobs = panel()
    write_json_fsync(
        state / "PANEL.json", {"jobs": jobs, "density_gate_sha256": digest(gate)}
    )
    queues = {
        (gpu, lane): [j for j in jobs if (j["gpu_index"], j["lane"]) == (gpu, lane)]
        for gpu in range(8)
        for lane in range(2)
    }
    assert max(map(len, queues.values())) <= 8
    active = []
    task_lanes = {}
    processed = set()
    admission = {"open": True}

    def start_next(key: tuple[int, int]) -> None:
        if admission["open"] and queues[key]:
            if deadline - time.monotonic() < max(job_budget, 12 * 3600):
                admission["open"] = False
                journal.record(
                    "admissions_closed", reason="Insufficient full-job runway"
                )
                return
            job = queues[key].pop(0)
            try:
                task = launch(job, state, "fit")
            except Exception as error:
                LOGGER.exception("Closing admission after spawn failure")
                admission["open"] = False
                journal.record("admission_failed", job=job, error=str(error))
                return
            active.append(task)
            task_lanes[task.name] = key
            journal.record("fit_started", job=job, pid=task.process.pid)

    def advance(running: list, results: dict) -> None:
        newly_finished = [
            (name, result) for name, result in results.items() if name not in processed
        ]
        valid_names = []
        # Validate the entire observed completion batch before any new admission.
        for name, result in newly_finished:
            processed.add(name)
            valid = result["status"] == "completed"
            detail = "Failed process"
            if valid:
                try:
                    job = next(job for job in jobs if job["id"] == name)
                    delivered = validate_delivery(state, job)
                    write_json_fsync(state / "endpoints" / f"{name}.json", delivered)
                except Exception as error:
                    LOGGER.exception(
                        "Closing admission after delivery validation failure"
                    )
                    valid = False
                    detail = f"Delivered artifact validation failed: {error}"
            if not valid:
                admission["open"] = False
                journal.record(
                    "admissions_closed", failed=name, result=result, reason=detail
                )
            else:
                valid_names.append(name)
        for name in valid_names:
            start_next(task_lanes[name])

    baseline = cgroup_snapshot()["memory_events"]
    try:
        for key in queues:
            start_next(key)
        results = _monitor(
            active, journal, ROOT, deadline, baseline, [state], advance=advance
        )
    except BaseException:
        for task in active:
            if task.status == "running":
                terminate_owned(task.process)
        raise
    write_json_fsync(state / "RESULTS.json", results)
    if len(results) != 120 or any(r["status"] != "completed" for r in results.values()):
        raise RuntimeError(
            "Panel incomplete; healthy fits drained and artifacts retained"
        )


def main() -> None:
    """Run one bounded density or full-panel attempt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("profiles", "fits"))
    args = parser.parse_args()
    state = ROOT / (
        "state/pr15-density-v4"
        if args.mode == "profiles"
        else "state/pr15-execution-v1"
    )
    frozen_identity()
    state.mkdir(parents=True, exist_ok=False)
    journal = Journal(state / "events.jsonl")
    deadline = time.monotonic() + (3 * 3600 if args.mode == "profiles" else 120 * 3600)
    try:
        (profiles if args.mode == "profiles" else fits)(state, journal, deadline)
        write_json_fsync(
            state / "COMPLETE.json", {"mode": args.mode, "time_unix": time.time()}
        )
    except BaseException as error:
        write_json_fsync(
            state / "FAILED.json", {"error": str(error), "type": type(error).__name__}
        )
        raise


if __name__ == "__main__":
    main()
