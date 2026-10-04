"""Measure pinned native callback pairs with host-load telemetry, never load gating."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[5]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
OUTPUT_DEFAULT = CAMPAIGN / "staging/timing_window/v4/results/loaded_host_pair_v4"
E462_TIMING_PATH = CAMPAIGN / "staging/e462/time_public_q8.py"
E450_TIMING_PATH = CAMPAIGN / "staging/timing_window/time_seed4_i16.py"
E462_VERIFY_PATH = CAMPAIGN / "staging/e462/verify_public_q8.py"
DATA_PATH = ROOT / "cache/valid_subset20.parquet"
READINESS_PATH = CAMPAIGN / "checks/e450_seed4_native_parity_v1/readiness.json"
E450_ATTEMPT_PATH = CAMPAIGN / "runner/attempts/e450-seed4-native-parity-v1.json"
E462_BUILD_DIR = CAMPAIGN / "E462_public_q8_build_v2"
E462_PARITY_PATH = CAMPAIGN / "E462_public_q8_parity_v2/parity_report.json"
E462_SCORE_PATH = CAMPAIGN / "E462_public_q8_score_v4/report/score_report.json"
E462_ATTEMPTS = {
    "build": "e462-public-q8-build-v3",
    "parity": "e462-public-q8-parity-v2",
    "score": "e462-public-q8-score-v4",
}
E450_ATTEMPT = "e450-seed4-native-parity-v1"
DATA_SHA256 = "cd6a349b3655bb6f98314e5db391e69b1e1ea7f12ab18f1d15c61cefd28daa37"
ZIP_SHA256 = {
    "seed3_i16": "21f9b7368eddd64ad8ae19ae0702990a57a1806531a85148b657e5aed719dafb",
    "seed4_i16": "c4dc258b529a52267a5e108ef0936f7919221cbdd17913270c8eb3482557f94a",
    "q8r": "7edb26d8903bafa2f3c38bf31b1a5d0dad1fd6f79e1f4d5a0ea13eaf3ca044fd",
    "q8x": "b435c04c0b047c72aeaca93a0be5aa2436a885d3abce5da483bd7fa43bfa87e7",
}
SOURCE_SHA256 = {
    "staging/e462/time_public_q8.py": "7178b08cf108e518c9774bbe9432917ce989521ebcf06e5d8f0b15ee508661ae",
    "staging/timing_window/time_seed4_i16.py": "2223ead2ac140a1959fc932c6c75f753e594dca0c39f806303480427a49ee729",
    "staging/e462/verify_public_q8.py": "e04d89520b395dc9435c37debe244c4c182f062a5386a21678235753cd62e149",
}
EXPECTED_ROWS = 60_000
EXPECTED_ROWS_PER_STREAM = 20_000
EXPECTED_STREAMS = 3
EXPECTED_WARMUP_ROWS = 297
EXPECTED_SCORED_ROWS = 59_703
PAIRS_PER_CONTRAST = 5
CPU_ID = 15
FULL_EVALUATION_ROWS = 37_460_000
PUBLIC_SEED3_RUNTIME_SECONDS = 2_267.0
LOG_T_CRITICAL_DF4 = 2.7764451052
PHASES = (
    ("seed4_i16_vs_seed3_i16", "seed3_i16", "seed4_i16"),
    ("q8r_vs_seed3_i16", "seed3_i16", "q8r"),
    ("q8x_vs_seed3_i16", "seed3_i16", "q8x"),
)


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    """Load a pinned helper module from its exact source path."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helper source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require_successful_attempt(attempt_name: str) -> dict[str, Any]:
    path = CAMPAIGN / "runner" / "attempts" / f"{attempt_name}.json"
    record = read_json(path)
    if (
        record.get("id") != attempt_name
        or record.get("status") != "finished"
        or record.get("exit_code") != 0
        or record.get("teardown_verified") is not True
    ):
        raise RuntimeError(f"required attempt did not finish successfully: {attempt_name}")
    return record


def require_artifact(record: dict[str, Any], path: Path) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    observed_sha = sha256_file(path)
    artifact = next(
        (item for item in record.get("artifacts", []) if item.get("path") == relative),
        None,
    )
    if artifact is None or artifact.get("sha256") != observed_sha:
        raise ValueError(f"successful attempt does not bind exact artifact: {relative}")
    return {"path": relative, "sha256": observed_sha}


def require_code_and_package_hashes() -> dict[str, Any]:
    pinned_sources = {
        relative: sha256_file(ROOT / relative) for relative in SOURCE_SHA256
    }
    if pinned_sources != SOURCE_SHA256:
        raise ValueError(f"frozen timing helper source changed: {pinned_sources}")
    observed_sources = {
        **pinned_sources,
        Path(__file__).resolve().relative_to(ROOT).as_posix(): sha256_file(Path(__file__).resolve()),
    }
    observed_zips = {
        "seed3_i16": sha256_file(ROOT / "submissions/c20261004_combo_s3_i16/submission.zip"),
        "seed4_i16": sha256_file(ROOT / "submissions/c20261004_combo_s4_i16/submission.zip"),
        "q8r": sha256_file(E462_BUILD_DIR / "q8r/submission.zip"),
        "q8x": sha256_file(E462_BUILD_DIR / "q8x/submission.zip"),
    }
    if observed_zips != ZIP_SHA256:
        raise ValueError(f"fixed timing package hashes changed: {observed_zips}")
    observed_data = sha256_file(DATA_PATH)
    if observed_data != DATA_SHA256:
        raise ValueError("timing stream data differs from the exact E457 20-stream panel")
    return {"source_sha256": observed_sources, "package_sha256": observed_zips, "data_sha256": observed_data}


def require_output_binding(record: dict[str, Any], paths: tuple[Path, ...]) -> list[dict[str, str]]:
    return [require_artifact(record, path) for path in paths]


def require_prerequisites(timing: Any, e450: Any, paths: dict[str, Path]) -> dict[str, Any]:
    e450_record = require_successful_attempt(E450_ATTEMPT)
    e450_binding = e450._require_e450_parity(
        READINESS_PATH,
        E450_ATTEMPT_PATH,
        paths["seed4_i16"],
        DATA_PATH,
    )
    e450_binding["readiness_artifact"] = require_artifact(e450_record, READINESS_PATH)

    build_record = require_successful_attempt(E462_ATTEMPTS["build"])
    parity_record = require_successful_attempt(E462_ATTEMPTS["parity"])
    score_record = require_successful_attempt(E462_ATTEMPTS["score"])
    build_report = E462_BUILD_DIR / "build_report.json"
    build_bindings = require_output_binding(
        build_record,
        (build_report, paths["q8r"], paths["q8x"]),
    )
    parity_binding = require_artifact(parity_record, E462_PARITY_PATH)
    score_binding = require_artifact(score_record, E462_SCORE_PATH)
    score_sha, build_sha = timing._score_gate(
        E462_BUILD_DIR,
        E462_PARITY_PATH,
        E462_SCORE_PATH,
        {"i16": paths["seed3_i16"], "q8r": paths["q8r"], "q8x": paths["q8x"]},
    )
    if score_sha != score_binding["sha256"] or build_sha != build_bindings[0]["sha256"]:
        raise ValueError("frozen E462 helper returned hashes inconsistent with attempt artifacts")
    return {
        "e450_seed4_parity": e450_binding,
        "e462_build_artifacts": build_bindings,
        "e462_parity_artifact": parity_binding,
        "e462_score_artifact": score_binding,
        "attempts": {
            "e450_seed4_parity": e450_record["spec_sha256"],
            "e462_build": build_record["spec_sha256"],
            "e462_parity": parity_record["spec_sha256"],
            "e462_score": score_record["spec_sha256"],
        },
    }


def read_cpu_snapshot() -> dict[str, Any]:
    """Record host load, CPU pressure/counters, memory pressure, and reclaim indicators."""
    result: dict[str, Any] = {"loadavg": list(os.getloadavg())}
    proc_stat: dict[str, list[int]] = {}
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            fields = line.split()
            if fields and (fields[0] == "cpu" or fields[0] == "cpu15"):
                proc_stat[fields[0]] = [int(value) for value in fields[1:]]
    except (OSError, ValueError):
        proc_stat = {}
    result["proc_stat"] = proc_stat
    for name in ("cpu", "memory"):
        path = Path(f"/proc/pressure/{name}")
        try:
            lines = path.read_text().splitlines()
            result[f"pressure_{name}"] = {
                fields[0]: {
                    key: float(value) if key.startswith("avg") else int(value)
                    for token in fields[1:]
                    for key, value in [token.split("=", 1)]
                }
                for fields in (line.split() for line in lines)
                if fields
            }
        except (OSError, ValueError):
            result[f"pressure_{name}"] = None
    wanted_vmstat = {
        "pgfault", "pgmajfault", "pgscan_kswapd", "pgscan_direct",
        "pgsteal_kswapd", "pgsteal_direct", "pswpin", "pswpout",
    }
    try:
        result["vmstat"] = {
            fields[0]: int(fields[1])
            for line in Path("/proc/vmstat").read_text().splitlines()
            if (fields := line.split()) and fields[0] in wanted_vmstat
        }
    except (OSError, ValueError):
        result["vmstat"] = None
    return result


def cpu_busy_fraction(before: list[int] | None, after: list[int] | None) -> float | None:
    if before is None or after is None or len(before) < 5 or len(after) < 5:
        return None
    delta = [right - left for left, right in zip(before, after, strict=True)]
    # guest/guest_nice are already included in user/nice in /proc/stat.
    total = sum(delta[:8])
    idle = delta[3] + delta[4]
    if total <= 0:
        return None
    return (total - idle) / total


def snapshot_deltas(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    cpu_before = before.get("proc_stat", {})
    cpu_after = after.get("proc_stat", {})
    cpu_deltas = {}
    busy_fractions = {}
    for name in ("cpu", "cpu15"):
        left = cpu_before.get(name)
        right = cpu_after.get(name)
        if left is not None and right is not None and len(left) == len(right):
            cpu_deltas[name] = [b - a for a, b in zip(left, right, strict=True)]
            busy_fractions[name] = cpu_busy_fraction(left, right)
        else:
            cpu_deltas[name] = None
            busy_fractions[name] = None
    vm_before = before.get("vmstat") or {}
    vm_after = after.get("vmstat") or {}
    vm_deltas = {
        key: vm_after[key] - vm_before[key]
        for key in vm_before.keys() & vm_after.keys()
    }
    return {
        "host_cpu_busy_fraction": busy_fractions.get("cpu"),
        "cpu15_busy_fraction": busy_fractions.get("cpu15"),
        "host_cpu_counter_delta": cpu_deltas.get("cpu"),
        "cpu15_counter_delta": cpu_deltas.get("cpu15"),
        "vmstat_counter_delta": vm_deltas,
    }


class CallbackClock:
    """Capture process/thread CPU clocks at the frozen helper's wall-clock boundaries."""

    def __init__(self) -> None:
        self.samples: list[dict[str, float]] = []

    def perf_counter(self) -> float:
        wall = time.perf_counter()
        self.samples.append(
            {
                "wall": wall,
                "process_cpu": time.process_time(),
                "thread_cpu": time.thread_time(),
            }
        )
        return wall


def run_one_trial(
    timing: Any,
    cycles: Any,
    label: str,
    package: Path,
    streams: list[tuple[int, Any, Any]],
    pair_id: str,
    pair_order: list[str],
    hashes: dict[str, Any],
) -> dict[str, Any]:
    host_before = read_cpu_snapshot()
    clock = CallbackClock()
    previous_time = timing.time
    timing.time = clock
    started = time.perf_counter()
    process_started = time.process_time()
    row: dict[str, Any] = {
        "pair_id": pair_id,
        "pair_order": pair_order,
        "candidate": label,
        "package_sha256": hashes["package_sha256"][label],
        "native_library_sha256": hashes["native_library_sha256"][label],
        "stream_rows": EXPECTED_ROWS,
        "host_before": host_before,
        "status": "failed",
    }
    error: str | None = None
    callback_result = None
    try:
        callback_result = timing._trial(label, package, streams, cycles)
        if callback_result.get("rows") != EXPECTED_ROWS:
            raise ValueError("frozen callback helper did not return exactly 60,000 complete rows")
        if len(clock.samples) != 2:
            raise ValueError(f"expected exactly two timed-window clock samples, got {len(clock.samples)}")
        first, last = clock.samples
        wall = float(callback_result["elapsed_seconds"])
        process_cpu = last["process_cpu"] - first["process_cpu"]
        thread_cpu = last["thread_cpu"] - first["thread_cpu"]
        cycles_value = callback_result.get("cycles")
        row.update(
            {
                "status": "passed",
                "callback_result": callback_result,
                "callback_wall_seconds": wall,
                "process_cpu_seconds": process_cpu,
                "thread_cpu_seconds": thread_cpu,
                "wall_microseconds_per_row": wall * 1e6 / EXPECTED_ROWS,
                "process_cpu_microseconds_per_row": process_cpu * 1e6 / EXPECTED_ROWS,
                "thread_cpu_microseconds_per_row": thread_cpu * 1e6 / EXPECTED_ROWS,
                "process_cpu_over_wall": process_cpu / wall if wall > 0 else None,
                "thread_cpu_over_wall": thread_cpu / wall if wall > 0 else None,
                "hardware_thread_cycles": cycles_value,
                "hardware_thread_cycles_per_row": (
                    None if cycles_value is None else cycles_value / EXPECTED_ROWS
                ),
                "clock_boundary_samples": clock.samples,
                "cycle_counter_event": "Linux PERF_COUNT_HW_CPU_CYCLES for current user thread; not literal RDTSC",
            }
        )
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
        row["error"] = error
        row["clock_boundary_samples"] = clock.samples
    finally:
        timing.time = previous_time
        host_after = read_cpu_snapshot()
        row["host_after"] = host_after
        row["host_deltas"] = snapshot_deltas(host_before, host_after)
        row["whole_trial_process_cpu_seconds"] = time.process_time() - process_started
        row["whole_trial_wall_seconds"] = time.perf_counter() - started
    if error is not None:
        row["status"] = "failed"
    return row


def geometric_mean_ci(values: list[float]) -> dict[str, float]:
    if len(values) != PAIRS_PER_CONTRAST or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("expected five finite positive values for the preregistered interval")
    logged = [math.log(value) for value in values]
    mean = statistics.mean(logged)
    standard_error = statistics.stdev(logged) / math.sqrt(len(logged))
    margin = LOG_T_CRITICAL_DF4 * standard_error
    return {
        "n": len(values),
        "values": values,
        "arithmetic_mean": statistics.mean(values),
        "median": statistics.median(values),
        "geometric_mean": math.exp(mean),
        "ci95_low": math.exp(mean - margin),
        "ci95_high": math.exp(mean + margin),
        "log_mean": mean,
        "log_standard_error": standard_error,
        "interval_method": "two-sided Student t on five log values, df=4, t*=2.7764451052",
    }


def summarize_phase(name: str, control: str, variant: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    phase_trials = [trial for trial in trials if trial["phase"] == name]
    pairs: list[dict[str, Any]] = []
    for repeat in range(PAIRS_PER_CONTRAST):
        pair_id = f"{name}-pair-{repeat + 1}"
        members = [trial for trial in phase_trials if trial["pair_id"] == pair_id]
        by_label = {trial["candidate"]: trial for trial in members}
        if set(by_label) != {control, variant}:
            raise ValueError(f"incomplete paired sample {pair_id}")
        ratio = (
            by_label[variant]["callback_wall_seconds"]
            / by_label[control]["callback_wall_seconds"]
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "order": members[0]["pair_order"],
                "control_wall_seconds": by_label[control]["callback_wall_seconds"],
                "variant_wall_seconds": by_label[variant]["callback_wall_seconds"],
                "variant_over_control_wall_ratio": ratio,
                "control_cycles_per_row": by_label[control]["hardware_thread_cycles_per_row"],
                "variant_cycles_per_row": by_label[variant]["hardware_thread_cycles_per_row"],
                "control_process_cpu_over_wall": by_label[control]["process_cpu_over_wall"],
                "variant_process_cpu_over_wall": by_label[variant]["process_cpu_over_wall"],
            }
        )
    ratios = [pair["variant_over_control_wall_ratio"] for pair in pairs]
    ratio_ci = geometric_mean_ci(ratios)
    variant_us_per_row = [
        float(trial["wall_microseconds_per_row"])
        for trial in phase_trials
        if trial["candidate"] == variant
    ]
    control_us_per_row = [
        float(trial["wall_microseconds_per_row"])
        for trial in phase_trials
        if trial["candidate"] == control
    ]
    variant_rate = geometric_mean_ci(variant_us_per_row)
    control_rate = geometric_mean_ci(control_us_per_row)
    return {
        "contrast": name,
        "control": control,
        "variant": variant,
        "pairs": pairs,
        "variant_over_control_wall_ratio": ratio_ci,
        "candidate_callback_us_per_row": variant_rate,
        "control_callback_us_per_row": control_rate,
        "scaled_seed3_public_runtime_projection_seconds": {
            "point": PUBLIC_SEED3_RUNTIME_SECONDS * ratio_ci["geometric_mean"],
            "ci95_low": PUBLIC_SEED3_RUNTIME_SECONDS * ratio_ci["ci95_low"],
            "ci95_high": PUBLIC_SEED3_RUNTIME_SECONDS * ratio_ci["ci95_high"],
            "anchor_seconds": PUBLIC_SEED3_RUNTIME_SECONDS,
            "anchor_display": "37m47s (2,267 seconds), user-reported seed-3 i16 public runtime",
            "interpretation": "relative-speed projection only; cross-platform and host-load transfer is unknown",
        },
        "absolute_callback_projection_seconds": {
            "point": variant_rate["geometric_mean"] * FULL_EVALUATION_ROWS / 1e6,
            "ci95_low": variant_rate["ci95_low"] * FULL_EVALUATION_ROWS / 1e6,
            "ci95_high": variant_rate["ci95_high"] * FULL_EVALUATION_ROWS / 1e6,
            "row_count": FULL_EVALUATION_ROWS,
            "row_count_basis": "1,873 sequences times 20,000 rows per sequence",
            "interpretation": "callback-only projection; excludes package setup, evaluator I/O/scoring, and hardware differences",
        },
    }


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    if CPU_ID not in os.sched_getaffinity(0):
        raise RuntimeError(f"required CPU{CPU_ID} is unavailable in worker affinity")
    os.sched_setaffinity(0, {CPU_ID})
    if os.sched_getaffinity(0) != {CPU_ID}:
        raise RuntimeError("failed to pin the timing worker to CPU15")

    helper_hashes = require_code_and_package_hashes()
    timing = load_module("loaded_host_e462_timing_v4", E462_TIMING_PATH)
    e450 = load_module("loaded_host_e450_parity_v4", E450_TIMING_PATH)
    prerequisites = require_prerequisites(
        timing,
        e450,
        {
            "seed3_i16": ROOT / "submissions/c20261004_combo_s3_i16/submission.zip",
            "seed4_i16": ROOT / "submissions/c20261004_combo_s4_i16/submission.zip",
            "q8r": E462_BUILD_DIR / "q8r/submission.zip",
            "q8x": E462_BUILD_DIR / "q8x/submission.zip",
        },
    )
    streams, preload_seconds = timing._load_streams(DATA_PATH)
    if len(streams) != EXPECTED_STREAMS:
        raise ValueError(f"expected {EXPECTED_STREAMS} streams; observed {len(streams)}")
    row_counts = [int(features.shape[0]) for _, features, _ in streams]
    warmup_rows = sum(int((~need).sum()) for _, _, need in streams)
    scored_rows = sum(int(need.sum()) for _, _, need in streams)
    if (
        row_counts != [EXPECTED_ROWS_PER_STREAM] * EXPECTED_STREAMS
        or sum(row_counts) != EXPECTED_ROWS
        or warmup_rows != EXPECTED_WARMUP_ROWS
        or scored_rows != EXPECTED_SCORED_ROWS
    ):
        raise ValueError("fixed callback panel row/warmup/score counts changed")

    output.mkdir(parents=True, exist_ok=False)
    journal_path = output / "trials.jsonl"
    packages = {
        "seed3_i16": ROOT / "submissions/c20261004_combo_s3_i16/submission.zip",
        "seed4_i16": ROOT / "submissions/c20261004_combo_s4_i16/submission.zip",
        "q8r": E462_BUILD_DIR / "q8r/submission.zip",
        "q8x": E462_BUILD_DIR / "q8x/submission.zip",
    }
    native_hashes = {
        label: timing.zip_member_sha256(package, "v3/wnn_native.abi3.so")
        for label, package in packages.items()
    }
    hashes = {**helper_hashes, "native_library_sha256": native_hashes}
    cycles = timing.ThreadCycles()
    trials: list[dict[str, Any]] = []
    command = list(sys.argv)
    job_start = time.perf_counter()
    try:
        with journal_path.open("x", encoding="utf-8") as journal:
            for phase, control, variant in PHASES:
                for repeat in range(PAIRS_PER_CONTRAST):
                    pair_id = f"{phase}-pair-{repeat + 1}"
                    order = [control, variant] if repeat % 2 == 0 else [variant, control]
                    for label in order:
                        row = run_one_trial(
                            timing,
                            cycles,
                            label,
                            packages[label],
                            streams,
                            pair_id,
                            order,
                            hashes,
                        )
                        row["phase"] = phase
                        row["repeat"] = repeat + 1
                        row["command"] = command
                        journal.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        journal.flush()
                        os.fsync(journal.fileno())
                        trials.append(row)
                        if row["status"] != "passed":
                            raise RuntimeError(str(row.get("error", "callback trial failed")))
    except BaseException as error:
        (output / "failure.json").write_text(
            json.dumps(
                {
                    "schema": "e465-loaded-host-paired-callback-timing-v4",
                    "status": "failed_callback_or_provenance",
                    "error": f"{type(error).__name__}: {error}",
                    "completed_trial_count": len(trials),
                    "trials_file": journal_path.name,
                    "host_load_is_not_a_validity_gate": True,
                    "source_and_input_hashes": hashes,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        cycles.close()

    summaries = [summarize_phase(*phase, trials) for phase in PHASES]
    report = {
        "schema": "e465-loaded-host-paired-callback-timing-v4",
        "status": "complete_diagnostic_only",
        "claim_scope": {
            "public_runtime_claim": False,
            "submission_readiness_claim": False,
            "holdout_score_claim": False,
            "training_or_export_performed": False,
            "host_load_aborts_trials": False,
        },
        "protocol": {
            "cpu_affinity": [CPU_ID],
            "pairs_per_contrast": PAIRS_PER_CONTRAST,
            "rows_per_callback": EXPECTED_ROWS,
            "streams_per_callback": EXPECTED_STREAMS,
            "rows_per_stream": EXPECTED_ROWS_PER_STREAM,
            "warmup_rows": EXPECTED_WARMUP_ROWS,
            "score_rows": EXPECTED_SCORED_ROWS,
            "order": "seed4_i16 vs seed3_i16, then q8r vs seed3_i16, then q8x vs seed3_i16; A/B order reverses on alternating pairs",
            "counter": "Linux PERF_COUNT_HW_CPU_CYCLES user-thread event; reported as hardware cycles, not literal RDTSC",
            "interval": "two-sided Student-t on five log values, df=4; all raw pairs retained",
        },
        "inputs": {
            **hashes,
            "prerequisite_bindings": prerequisites,
            "sequence_ids": [int(sequence) for sequence, _, _ in streams],
            "preload_seconds_outside_callback_timing": preload_seconds,
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "load_is_telemetry_only": True,
            "job_wall_seconds": time.perf_counter() - job_start,
        },
        "public_anchor": {
            "seed3_i16_runtime_seconds": PUBLIC_SEED3_RUNTIME_SECONDS,
            "display": "37m47s",
            "source": "user-reported E412 public result for the exact seed-3 i16 ZIP",
            "future_public_runtime_is_unobserved": True,
        },
        "summaries": summaries,
        "trial_count": len(trials),
        "trials_file": journal_path.name,
        "interpretation_limits": [
            "Host-load and CPU counters are covariates, not a post-hoc exclusion rule.",
            "No direct memory-controller bandwidth counter is collected; bandwidth attribution is not supported.",
            "Callback projections omit setup, evaluator I/O/scoring, and public-hardware differences.",
            "Five pairs provide a small-sample uncertainty interval and cannot establish a strict runtime guarantee.",
        ],
    }
    (output / "timing_report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    run(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
