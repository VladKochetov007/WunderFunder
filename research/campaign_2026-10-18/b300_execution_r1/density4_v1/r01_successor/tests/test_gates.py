"""Focused tests for slot uniqueness and campaign resource admission."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CONTROL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTROL))

from authority import validate_campaign_resource_sample  # noqa: E402
from frozen_panel import GPU_UUIDS  # noqa: E402
from schedule import available_jobs, lane_plan, second_r01_cpu_pair  # noqa: E402


def _jobs() -> list[dict[str, object]]:
    base_arms = ("wide", "deep", "deep", "wide", "wide", "deep", "deep", "wide")
    base_cpus = ((0, 2), (4, 6), (8, 10), (12, 14), (1, 3), (5, 7), (9, 11), (13, 15))
    jobs = []
    for gpu, (arm, cpus) in enumerate(zip(base_arms, base_cpus, strict=True)):
        jobs.append(
            {
                "id": f"seed1003_fold0_{arm}_gpu{gpu}",
                "seed": 1003,
                "fold": 0,
                "gpu_index": gpu,
                "gpu_uuid": GPU_UUIDS[gpu],
                "arm": arm,
                "cpu_ids_initial": list(cpus),
                "argv": ["python", "train.py"],
                "run_dir": f"/data/runs/gpu{gpu}",
            }
        )
    for gpu in range(8):
        for arm in ("wide", "deep"):
            jobs.append(
                {
                    "id": f"seed1004_fold0_{arm}_gpu{gpu}",
                    "seed": 1004,
                    "fold": 0,
                    "gpu_index": gpu,
                    "gpu_uuid": GPU_UUIDS[gpu],
                    "arm": arm,
                    "cpu_ids_initial": list(base_cpus[gpu]),
                    "argv": ["python", "train.py"],
                    "run_dir": f"/data/runs/seed1004_{arm}_gpu{gpu}",
                }
            )
    return jobs


def test_scheduler_reserves_each_gpu_arm_only_once() -> None:
    jobs = _jobs()
    plan = lane_plan(jobs)
    candidates = available_jobs(jobs, {str(job["id"]) for job in jobs}, {}, plan)
    keys = {(job["gpu_index"], job["arm"]) for job in candidates}
    assert len(candidates) == 16
    assert len(keys) == 16


def test_scheduler_fills_only_the_opposite_arm_on_supplemental_cpus() -> None:
    jobs = _jobs()
    plan = lane_plan(jobs)
    occupied = {(gpu, str(slot["base_arm"])): f"adopted-gpu{gpu}" for gpu, slot in plan.items()}
    candidates = available_jobs(jobs, {str(job["id"]) for job in jobs}, occupied, plan)
    assert len(candidates) == 8
    for job in candidates:
        gpu = int(job["gpu_index"])
        assert job["arm"] == plan[gpu]["other_arm"]
        assert tuple(job["cpu_ids_initial"]) == second_r01_cpu_pair(gpu)


def _unit_record() -> dict[str, object]:
    return {
        "lifecycle": "live",
        "memory_max_bytes": 192 * 1024**3,
        "memory_swap_max_bytes": 0,
        "pids_max": 512,
        "memory_current_bytes": 20 * 1024**3,
        "memory_events": {
            "high": 0,
            "max": 0,
            "oom": 0,
            "oom_group_kill": 0,
            "oom_kill": 0,
        },
    }


def _resource_receipt() -> dict[str, object]:
    return {
        "scope": "b300_campaign_aggregate",
        "verified": True,
        "units": {
            "tabdata-r01-execution-v4": _unit_record(),
            "tabdata-r01-density4-v1": _unit_record(),
            "tabdata-pr15-execution-v2": _unit_record(),
        },
        "host_memory": {
            "MemAvailable_bytes": 128 * 1024**3,
            "MemTotal_bytes": 512 * 1024**3,
        },
        "memory_swap_current_bytes": 0,
        "worker_group_rss_bytes": {"r01-worker": 3 * 1024**3},
    }


def test_campaign_sample_includes_current_pr15_v2_and_per_unit_limits() -> None:
    validate_campaign_resource_sample(
        _resource_receipt(),
        {"high": 0, "max": 0, "oom": 0, "oom_group_kill": 0, "oom_kill": 0},
    )


def test_campaign_sample_fails_closed_if_pr15_v2_is_unobserved() -> None:
    receipt = _resource_receipt()
    receipt["units"].pop("tabdata-pr15-execution-v2")
    with pytest.raises(ValueError, match="PR15 v2"):
        validate_campaign_resource_sample(
            receipt,
            {"high": 0, "max": 0, "oom": 0, "oom_group_kill": 0, "oom_kill": 0},
        )


def test_campaign_sample_accepts_receipted_terminal_pr15_without_fake_live_events() -> None:
    receipt = _resource_receipt()
    pr15 = receipt["units"]["tabdata-pr15-execution-v2"]
    pr15.update(
        lifecycle="terminal_success",
        memory_current_bytes=0,
        memory_current_basis="cgroup_absent_and_no_process_members",
        memory_swap_current_bytes=0,
        pids_current=0,
        memory_events_last_observed={
            "high": 0,
            "max": 0,
            "oom": 0,
            "oom_group_kill": 0,
            "oom_kill": 0,
        },
        memory_events_last_observed_unix=100.0,
        last_observed_control_group="/user.slice/tabdata-pr15-execution-v2.service",
        terminal_receipt={
            "schema": "tabdata-pr15-terminal-success.v1",
            "complete_sha256": "a" * 64,
            "results_sha256": "b" * 64,
            "completed_successor_cells": 104,
            "preserved_original_fits": 16,
            "results_count": 104,
            "systemd_load_state": "not-found",
            "systemd_active_state": "inactive",
            "systemd_result_observed": None,
            "systemd_main_pid": 0,
            "hard_caps_basis": "last_observed_live_cgroup_and_systemd_properties",
            "cgroup_absent": True,
            "process_count": 0,
            "terminal_proof": {
                "schema": "tabdata-pr15-terminal-proof.v1",
                "first_terminal_observed_unix": 110.0,
                "last_live_observed_unix": 100.0,
                "last_live_control_group": "/user.slice/tabdata-pr15-execution-v2.service",
                "last_live_memory_events": {
                    "high": 0,
                    "max": 0,
                    "oom": 0,
                    "oom_group_kill": 0,
                    "oom_kill": 0,
                },
                "last_live_memory_max_bytes": 192 * 1024**3,
                "last_live_memory_swap_max_bytes": 0,
                "last_live_pids_max": 512,
                "complete_sha256": "a" * 64,
                "results_sha256": "b" * 64,
                "completed_successor_cells": 104,
                "preserved_original_fits": 16,
                "results_count": 104,
                "systemd_load_state": "not-found",
                "systemd_result_observed": None,
                "hard_caps_basis": "last_observed_live_cgroup_and_systemd_properties",
            },
        },
    )
    pr15.pop("memory_events")
    validate_campaign_resource_sample(
        receipt,
        {"high": 0, "max": 0, "oom": 0, "oom_group_kill": 0, "oom_kill": 0},
    )


def test_campaign_sample_rejects_terminal_pr15_without_bound_success_receipts() -> None:
    receipt = _resource_receipt()
    pr15 = receipt["units"]["tabdata-pr15-execution-v2"]
    pr15.update(lifecycle="terminal_success", memory_current_bytes=0)
    with pytest.raises(ValueError, match="absent cgroup"):
        validate_campaign_resource_sample(
            receipt,
            {"high": 0, "max": 0, "oom": 0, "oom_group_kill": 0, "oom_kill": 0},
        )
