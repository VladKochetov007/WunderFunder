"""Execute the preregistered B300 R01 panel with verified inputs and grading."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKET = Path(__file__).resolve().parent
STATE = ROOT / "state/r01-execution-v1"
sys.path[:0] = [str(PACKET / "control"), str(PACKET / "code")]
from adapter_common import adapter_bundle_identity
from fit_stage import run_fit_cohort
from journal import Journal, write_json_fsync
from policy import sha256_file, validate_memory_budget
from postprocess import finish_cells
from profiles import run_profile
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
    """Measure both shapes on both shared and unshared GPUs before admission."""
    gates = []
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
            gated, _ = run_profile(
                job=job,
                root=ROOT,
                state=STATE / f"profile-{sharing}",
                journal=journal,
                wave_deadline=deadline,
            )
            gated["sharing_at_registration"] = sharing
            gates.append(gated)
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
        profile_shapes(panel, journal, deadline)
        journal.record("all_shape_profiles_passed", science_fits_started=False)
        wave_records = []
        for wave in range(8):
            verify_frozen()
            jobs = [job for job in panel["jobs"] if job["wave"] == wave]
            journal.record(
                "fit_wave_start", wave=wave, jobs=[job["id"] for job in jobs]
            )
            state = STATE / f"wave{wave}"
            records = run_fit_cohort(
                jobs=jobs,
                root=ROOT,
                state=state,
                journal=journal,
                fit_deadline=min(deadline, time.monotonic() + panel["max_fit_seconds"]),
                wave_deadline=deadline,
            )
            journal.record("fit_wave_end", wave=wave, records=records)
            # Export and grade intact fixed endpoints; failed fits are never refit implicitly.
            remaining = deadline - time.monotonic()
            post = finish_cells(ROOT, jobs, time.time() + remaining)
            record = {"wave": wave, "fits": records, "postprocess": post}
            write_json_fsync(state / "receipts/wave-complete.json", record)
            wave_records.append(record)
            write_json_fsync(
                STATE / "PROGRESS.json",
                {
                    "completed_waves": wave_records,
                    "elapsed_seconds": time.monotonic() - start,
                },
            )
            # Engineering failures stop new admissions, with partial science retained.
            if not all(r["status"] == "fixed_epoch5_receipt_verified" for r in records):
                raise RuntimeError(
                    f"Wave {wave} has failed fits; inspect receipts before new admissions"
                )
            if (
                len([g for g in post["grades"] if g["status"] == "graded"])
                != len(jobs) // 2
            ):
                raise RuntimeError(
                    f"Wave {wave} has incomplete export/grading; trained weights preserved"
                )
        write_json_fsync(
            STATE / "COMPLETE.json",
            {
                "waves": wave_records,
                "classification": "exploratory CP30; previously exposed folds",
                "elapsed_seconds": time.monotonic() - start,
            },
        )
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
