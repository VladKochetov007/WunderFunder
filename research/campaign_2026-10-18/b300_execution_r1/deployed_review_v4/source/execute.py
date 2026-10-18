"""Execute the preregistered B300 R01 panel with verified inputs and grading."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKET = Path(__file__).resolve().parent
STATE = ROOT / "state/r01-execution-v4"
sys.path[:0] = [str(PACKET / "control"), str(PACKET / "code")]
from adapter_common import adapter_bundle_identity
from journal import Journal, write_json_fsync
from policy import sha256_file, validate_memory_budget
from profiles import _validate_profile
from rolling import run_rolling
from runtime_gates import cgroup_snapshot, require_gpus, validate_python_runtime


def verify_frozen() -> dict:
    """Bind this invocation to the complete frozen controller and actor."""
    manifest = json.loads((PACKET / "EXECUTION_MANIFEST.json").read_text())
    for relative, expected in manifest["files"].items():
        if sha256_file(PACKET / relative) != expected:
            raise ValueError(f"Execution source changed: {relative}")
    panel = json.loads((PACKET / "PANEL.json").read_text())
    if adapter_bundle_identity() != panel["actor_bundle_identity"]:
        raise ValueError("Actor bundle differs from preregistered panel")
    return panel


def await_preparation(journal: Journal, deadline: float) -> dict:
    """Wait only for the existing bounded exact-data preparation attempt."""
    prep = ROOT / "state/prep-v2"
    while time.monotonic() < deadline:
        if (prep / "FAILED.json").exists():
            raise RuntimeError(
                f"Input preparation failed: {(prep / 'FAILED.json').read_text()}"
            )
        if (prep / "COMPLETE.json").is_file():
            result = json.loads((prep / "COMPLETE.json").read_text())
            if result["status"] != "verified_all_frozen_memberships":
                raise ValueError("Input membership gate did not pass")
            return result
        journal.heartbeat(phase="waiting_input_preparation")
        time.sleep(15)
    raise TimeoutError("Two-hour input preparation deadline expired")


def profile_shapes(panel: dict, journal: Journal, deadline: float) -> list[dict]:
    """Revalidate and reuse all four successful profiles; never refit for unit repair."""
    gates = []
    previous = ROOT / "state/r01-execution-v2"
    initial = panel["jobs"][:8]
    for sharing, gpu_by_arm in (
        ("shared", {"wide": 0, "deep": 2}),
        ("free", {"wide": 1, "deep": 3}),
    ):
        for arm, gpu in gpu_by_arm.items():
            original = next(j for j in initial if j["fold"] == 0 and j["arm"] == arm)
            location = next(j for j in initial if j["gpu_index"] == gpu)
            job = {
                **original,
                "gpu_index": gpu,
                "gpu_uuid": location["gpu_uuid"],
                "cpu_ids_initial": location["cpu_ids_initial"],
            }
            base = previous / f"profile-{sharing}"
            old = json.loads((base / "receipts" / f"profile-{arm}.json").read_text())
            run = base / "profiles" / arm
            log = base / "logs" / f"profile-{arm}.log"
            if sha256_file(log) != old["log_sha256"]:
                raise ValueError("Profile log changed after original admission")
            producer = json.loads((run / "producer_receipt.json").read_text())
            if producer["adapter_bundle_identity"] != panel["actor_bundle_identity"]:
                raise ValueError("Profile used a different source/adapter")
            gated = _validate_profile(job, old["process_result"], run, log)
            if gated["producer_receipt_sha256"] != old["producer_receipt_sha256"]:
                raise ValueError("Profile producer changed after admission")
            gated["reused_from"] = str(base)
            gates.append(gated)
            journal.record("profile_revalidated_reused", arm=arm, sharing=sharing)
    peaks = {
        arm: max(
            g["process_result"]["peak_rss_bytes"] for g in gates if g["arm"] == arm
        )
        for arm in ("wide", "deep")
    }
    snapshot = cgroup_snapshot()
    projected = validate_memory_budget(
        snapshot["memory_max"],
        snapshot["memory_current"],
        peaks["wide"],
        peaks["deep"],
        headroom=16 * 1024**3,
    )
    write_json_fsync(
        STATE / "receipts/profile-admission.json",
        {"gates": gates, "projected_memory_bytes": projected, "snapshot": snapshot},
    )
    return gates


def forecast_panel(
    panel: dict, gates: list[dict], journal: Journal, deadline: float
) -> None:
    """Convert profiler milliseconds and forecast fixed per-GPU lane workloads."""
    import math

    seconds = {}
    for arm in ("wide", "deep"):
        values = []
        for gate in (g for g in gates if g["arm"] == arm):
            timing = gate["timings"]
            value = timing["step_wall"] / 1000
            if not math.isclose(value, 8 / timing["seq_per_s"], rel_tol=0.02):
                raise ValueError("Profile milliseconds disagree with sequences/second")
            values.append(value)
        seconds[arm] = max(values)
    lanes = {job["gpu_uuid"]: 0.0 for job in panel["jobs"]}
    individual = []
    for job in panel["jobs"]:
        split = json.loads((ROOT / "splits" / f"fold{job['fold']}.json").read_text())
        count = sum(len(split["fit"][name]) for name in ("train", "valid"))
        estimate = 1.25 * 6 * math.ceil(count / 8) * seconds[job["arm"]] + 300
        lanes[job["gpu_uuid"]] += estimate
        individual.append(estimate)
    projected = max(lanes.values()) + 8 * 1800
    remaining = deadline - time.monotonic()
    result = {
        "is_estimate": True,
        "profile_timing_unit": "milliseconds",
        "measured_batch_seconds": seconds,
        "slowdown_multiplier": 1.25,
        "per_gpu_lane_seconds": lanes,
        "longest_fit_seconds": max(individual),
        "postprocess_reserve_seconds": 8 * 1800,
        "projected_panel_seconds": projected,
        "remaining_seconds": remaining,
        "queue_policy": "fixed GPU lanes, no global wave barrier",
    }
    write_json_fsync(STATE / "receipts/wall-forecast.json", result)
    journal.record("panel_wall_forecast", **result)
    if projected > remaining or max(individual) > panel["max_fit_seconds"]:
        raise RuntimeError("Measured queue projection exceeds the frozen time envelope")


def main() -> None:
    """Run all cells without outcome-based stopping, preserving every attempt."""
    os.sched_setaffinity(0, set(range(16)))
    STATE.mkdir(parents=True, exist_ok=False)
    journal = Journal(STATE / "events.jsonl")
    start = time.monotonic()
    panel = verify_frozen()
    deadline = start + panel["max_campaign_seconds"]
    try:
        snapshot = cgroup_snapshot()
        if (
            snapshot["memory_max"] != panel["memory_max_bytes"]
            or snapshot["swap_max"] != 0
        ):
            raise RuntimeError("Own unit memory/swap limits differ from frozen panel")
        if snapshot["pids_max"] != 512 or snapshot["cpu_affinity"] != list(range(16)):
            raise RuntimeError("Own unit process/CPU limits differ from frozen panel")
        runtime = validate_python_runtime()
        require_gpus(panel["jobs"][:8])
        write_json_fsync(
            STATE / "receipts/start.json",
            {
                "runtime": runtime,
                "cgroup": snapshot,
                "panel_sha256": sha256_file(PACKET / "PANEL.json"),
            },
        )
        await_preparation(journal, min(deadline, start + 7200))
        gates = profile_shapes(panel, journal, deadline)
        forecast_panel(panel, gates, journal, deadline)
        journal.record("all_shape_profiles_passed", science_fits_started=False)
        result = run_rolling(panel, ROOT, STATE, journal, deadline)
        write_json_fsync(STATE / "RESULT.json", result)
        if result["status"] != "complete":
            raise RuntimeError(
                "Panel delivery partial; completed weights retained, failed attempts need diagnosis"
            )
        write_json_fsync(STATE / "COMPLETE.json", result)
        journal.record("panel_complete", cells=60)
    except BaseException as error:
        write_json_fsync(
            STATE / "FAILED.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        journal.record(
            "execution_failed", error_type=type(error).__name__, error=str(error)
        )
        raise


if __name__ == "__main__":
    main()
