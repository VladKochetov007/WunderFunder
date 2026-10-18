"""Arm-locked two-lane R01 scheduling on the measured mixed-shape profile."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from frozen_panel import GPU_UUIDS

BASE_CPU_SET = frozenset(range(16))
SECOND_CPU_SET = frozenset(range(48, 64))


def second_r01_cpu_pair(gpu_index: int) -> tuple[int, int]:
    """Return the profiled supplemental CPU pair for one frozen GPU index."""
    if type(gpu_index) is not int or gpu_index not in range(8):
        raise ValueError("GPU index must be one of 0 through 7")
    first = 48 + (gpu_index % 4) * 4 + (gpu_index // 4)
    return first, first + 2


def lane_plan(jobs: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Bind exactly one wide and one deep lane to each unchanged GPU."""
    frozen = tuple(jobs)
    initial = [job for job in frozen if job.get("seed") == 1003]
    if len(initial) != 8:
        raise ValueError("R01-v4 first wave must contain eight seed-1003 cells")
    plan: dict[int, dict[str, Any]] = {}
    for job in initial:
        gpu = job["gpu_index"]
        if gpu in plan or job["gpu_uuid"] != GPU_UUIDS[gpu]:
            raise ValueError("seed-1003 base GPU mapping is duplicated or changed")
        cpus = tuple(job["cpu_ids_initial"])
        if len(cpus) != 2 or not set(cpus) <= BASE_CPU_SET:
            raise ValueError("frozen first lane CPUs must be a pair within CPUs 0-15")
        plan[gpu] = {
            "gpu_index": gpu,
            "gpu_uuid": job["gpu_uuid"],
            "base_arm": job["arm"],
            "other_arm": "deep" if job["arm"] == "wide" else "wide",
            "base_cpus": cpus,
            "other_cpus": second_r01_cpu_pair(gpu),
        }
    if set(plan) != set(range(8)):
        raise ValueError("frozen seed-1003 wave does not cover all GPUs")
    for job in frozen:
        slot = plan[job["gpu_index"]]
        if (
            job["gpu_uuid"] != slot["gpu_uuid"]
            or tuple(job["cpu_ids_initial"]) != slot["base_cpus"]
        ):
            raise ValueError(f"frozen placement changed for {job['id']}")
    all_cpus = [
        cpu
        for slot in plan.values()
        for pair in (slot["base_cpus"], slot["other_cpus"])
        for cpu in pair
    ]
    if len(all_cpus) != len(set(all_cpus)):
        raise ValueError("two-lane R01 CPU pairs overlap")
    if not set(all_cpus).issubset(BASE_CPU_SET | SECOND_CPU_SET):
        raise ValueError("R01 lanes escape the measured CPU sets")
    return plan


def operational_job(job: dict[str, Any], plan: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Select only the measured CPU lane while preserving frozen scientific argv."""
    slot = plan[int(job["gpu_index"])]
    if job["arm"] == slot["base_arm"]:
        lane_name, cpus = "base", slot["base_cpus"]
    elif job["arm"] == slot["other_arm"]:
        lane_name, cpus = "supplemental", slot["other_cpus"]
    else:
        raise ValueError(f"unknown R01 lane arm for {job['id']}")
    result = dict(job)
    result["frozen_cpu_ids_initial"] = list(job["cpu_ids_initial"])
    result["cpu_ids_initial"] = list(cpus)
    result["successor_lane"] = lane_name
    if result["argv"] != job["argv"] or result["run_dir"] != job["run_dir"]:
        raise AssertionError("operational CPU placement changed frozen fit identity")
    return result


def active_slots(
    live: dict[str, dict[str, Any]], plan: dict[int, dict[str, Any]]
) -> dict[tuple[int, str], str]:
    """Return occupied GPU/arm slots and reject any repeated same-arm lane."""
    occupied: dict[tuple[int, str], str] = {}
    for job_id, record in live.items():
        job = record["job"]
        key = (int(job["gpu_index"]), str(job["arm"]))
        if key in occupied:
            raise ValueError(f"two R01 fits occupy the same measured slot: {key}")
        lane_plan = plan[key[0]]
        if job["arm"] not in {lane_plan["base_arm"], lane_plan["other_arm"]}:
            raise ValueError(f"R01 fit arm has no lane on GPU {key[0]}")
        occupied[key] = job_id
    return occupied


def available_jobs(
    jobs: Iterable[dict[str, Any]],
    pending_ids: set[str],
    occupied: dict[tuple[int, str], str],
    plan: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return panel-ordered pending jobs whose exact GPU/arm lane is free."""
    result = []
    reserved = set(occupied)
    for job in jobs:
        job_id = str(job["id"])
        key = (int(job["gpu_index"]), str(job["arm"]))
        if job_id in pending_ids and key not in reserved:
            result.append(operational_job(job, plan))
            reserved.add(key)
    return result
