"""Explicit finite-metadata supplement to the unchanged frozen v3 binder."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import resource
from pathlib import Path

import build_screen_specs_v3 as binder

HERE = Path(__file__).resolve().parent


def finite_nonnegative(value: object, name: str, integer: bool = False) -> float:
    """Reject JSON bool, missing/non-numeric values and nonfinite quantities."""
    if type(value) not in (int, float) or (integer and type(value) is not int):
        raise ValueError(f"{name}: numeric metadata required")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}: finite nonnegative metadata required")
    return value


def job_resources(record: dict, timeout: float) -> dict:
    """Validate actual time/VRAM under the unchanged prescribed allowance."""
    elapsed = finite_nonnegative(record.get("elapsed_seconds"), "job elapsed")
    actual_timeout = finite_nonnegative(record.get("timeout_seconds"), "job timeout")
    peak = finite_nonnegative(record.get("peak_observed_vram_bytes"), "job VRAM", integer=True)
    if actual_timeout != timeout or actual_timeout <= 0 or elapsed > actual_timeout or peak > 5_000_000_000:
        raise ValueError("job resource metadata differs from frozen cap")
    return {"elapsed_seconds": elapsed, "timeout_seconds": actual_timeout, "peak_observed_vram_bytes": peak}


def stage_times(records: list[dict], job_elapsed: float) -> list[dict]:
    """Check finite durations and monotonically accumulated stage timing."""
    previous = 0.0
    result = []
    for record in records:
        duration = finite_nonnegative(record.get("duration_seconds"), "stage duration")
        aggregate = finite_nonnegative(record.get("aggregate_seconds"), "stage aggregate")
        if type(record.get("exit_code")) is not int or record["exit_code"] != 0:
            raise ValueError("stage exit must be integer zero")
        if aggregate < previous + duration or aggregate > job_elapsed:
            raise ValueError("stage aggregate timing inconsistent with duration or job elapsed")
        result.append({"stage": record["stage"], "duration_seconds": duration, "aggregate_seconds": aggregate})
        previous = aggregate
    return result


def inspect(design_path: Path, original_amendment_path: Path, allow_missing: bool = False) -> dict:
    """Check exact frozen fragments; never open targets or prediction values."""
    prior = binder.read(design_path)
    amendment = binder.read(original_amendment_path)
    closure = binder.read(binder.artifact(prior["job_source_closure"]))
    cap = finite_nonnegative(binder.read(binder.artifact(prior["pipeline_recipe"]))["cap_gpu_seconds_per_arm"], "whole arm cap")
    checked, missing = {}, []
    for logical in prior["stages_by_job"]:
        fragments = binder.fragment_plan(prior, logical, amendment)
        entries = []
        for fragment in fragments:
            path = fragment["job_path"]
            if allow_missing and not path.exists() and fragment["label"] == "missing_stages":
                missing.append(str(path.relative_to(binder.ROOT)))
                continue
            bound = binder.bind_fragment(fragment, closure["source_sha256"])
            resources = job_resources(bound["record"], fragment.get("timeout_seconds", cap))
            records = [binder.read(binder.artifact(entry["record"])) for entry in bound["stage_execution"]]
            timing = stage_times(records, resources["elapsed_seconds"])
            entries.append({"job": bound["job"], "label": fragment["label"], "resources": resources,
                            "timing": timing, "records": [entry["record"] for entry in bound["stage_execution"]]})
        elapsed = sum(entry["resources"]["elapsed_seconds"] for entry in entries)
        if elapsed > cap:
            raise ValueError("combined finite-fragment elapsed exceeds original cap")
        checked[logical] = {"fragments": entries, "aggregate_seconds": elapsed, "original_cap_seconds": cap}
    return {"status": "waiting-for-resume" if missing else "pass", "checked": checked,
            "missing": missing, "design": binder.identity(design_path),
            "original_binding_amendment": binder.identity(original_amendment_path),
            "supplemental_source_sha256": binder.file_sha256(Path(__file__)),
            "target_or_prediction_values_read": False}


def self_test() -> dict:
    """Known-answer and adversarial type/finite/cap tests, using synthetic data."""
    rejected = []
    def reject(name: str, action: object) -> None:
        try:
            action()
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(f"accepted invalid metadata: {name}")
    job = {"elapsed_seconds": 3.0, "timeout_seconds": 10.0, "peak_observed_vram_bytes": 1_000_000_000}
    assert job_resources(job, 10)["elapsed_seconds"] == 3
    records = [{"stage": "a", "exit_code": 0, "duration_seconds": 1.0, "aggregate_seconds": 1.1},
               {"stage": "b", "exit_code": 0, "duration_seconds": 1.0, "aggregate_seconds": 2.2}]
    assert len(stage_times(records, 3.0)) == 2
    for field in job:
        for label, value in [("bool", True), ("string", "3"), ("none", None), ("nan", float("nan")),
                             ("positive_inf", float("inf")), ("negative_inf", -float("inf")), ("negative", -1)]:
            reject(f"{field}_{label}", lambda field=field, value=value: job_resources({**job, field: value}, 10))
    reject("VRAM_float_bytes", lambda: job_resources({**job, "peak_observed_vram_bytes": 1.0}, 10))
    reject("VRAM_over_cap", lambda: job_resources({**job, "peak_observed_vram_bytes": 5_000_000_001}, 10))
    reject("elapsed_exceeds_timeout", lambda: job_resources({**job, "elapsed_seconds": 11.0}, 10))
    reject("timeout_changed", lambda: job_resources({**job, "timeout_seconds": 20.0}, 10))
    for field in ["duration_seconds", "aggregate_seconds"]:
        for label, value in [("bool", False), ("string", "1"), ("none", None), ("nan", float("nan")),
                             ("positive_inf", float("inf")), ("negative_inf", -float("inf")), ("negative", -1)]:
            wrong = copy.deepcopy(records)
            wrong[0][field] = value
            reject(f"{field}_{label}", lambda wrong=wrong: stage_times(wrong, 3.0))
    wrong = copy.deepcopy(records)
    wrong[1]["aggregate_seconds"] = 1.5
    reject("aggregate_does_not_include_duration", lambda: stage_times(wrong, 3.0))
    reject("aggregate_exceeds_job_elapsed", lambda: stage_times(records, 2.0))
    wrong = copy.deepcopy(records)
    wrong[0]["exit_code"] = False
    reject("boolean_exit_code", lambda: stage_times(wrong, 3.0))
    return {"passed": True, "rejections": rejected, "known_answer": True,
            "source_sha256": binder.file_sha256(Path(__file__)), "no_real_data_read": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["self-test", "inspect", "bind"])
    parser.add_argument("--design", type=Path)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--finite-amendment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    if args.phase == "self-test":
        result = self_test()
    else:
        if not args.design or not args.amendment:
            parser.error("inspection/binding requires design and original v3 amendment")
        result = inspect(args.design.resolve(), args.amendment.resolve(), allow_missing=args.phase == "inspect")
        if args.phase == "bind":
            if not args.finite_amendment:
                parser.error("binding requires explicit finite-metadata amendment")
            finite = binder.read(args.finite_amendment)
            if (finite["status"] != "approved-strict-finite-supplement-before-score"
                    or binder.artifact(finite["source"]).resolve() != Path(__file__).resolve()
                    or binder.artifact(finite["original_binding_amendment"]).resolve() != args.amendment.resolve()
                    or binder.artifact(finite["operational_preregistration"]).resolve() != args.design.resolve()):
                raise ValueError("supplemental amendment identity differs")
            binder.artifact(finite["checks"])
            if result["status"] != "pass":
                raise ValueError("all successful resume fragments are required")
            spec = binder.bind(args.design.resolve(), args.amendment.resolve())
            spec["strict_finite_metadata_amendment"] = binder.identity(args.finite_amendment.resolve())
            spec["strict_finite_metadata_source"] = finite["source"]
            spec["strict_finite_metadata_checks"] = finite["checks"]
            spec["strict_finite_metadata_preflight"] = result
            result = spec
    result["resource_fallback"] = {"rlimit_as": list(resource.getrlimit(resource.RLIMIT_AS)),
                                   "cpu_affinity": sorted(os.sched_getaffinity(0)),
                                   "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                   "hard_2GiB_cgroup_claim": False}
    binder.publish(args.output, result)
    print(json.dumps({"output": str(args.output), "status": result.get("status", result.get("passed"))}), flush=True)


if __name__ == "__main__":
    main()
