"""Measure a fixed seed-4 i16 ZIP against the original seed-3 i16 ZIP.

The callback loader/trial loop is imported from the frozen E462 timing module.
This wrapper adds only the distinct E450 readiness gate and a five-pair journal.
It does not score predictions or alter either ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
E462_TIMING = CAMPAIGN / "staging/e462/time_public_q8.py"
SEED3_ZIP_SHA256 = "21f9b7368eddd64ad8ae19ae0702990a57a1806531a85148b657e5aed719dafb"
E457_DATA_SHA256 = "cd6a349b3655bb6f98314e5db391e69b1e1ea7f12ab18f1d15c61cefd28daa37"
EXPECTED_E450_ZIP_SHA256 = "c4dc258b529a52267a5e108ef0936f7919221cbdd17913270c8eb3482557f94a"
E450_PARITY_ATTEMPT = "e450-seed4-native-parity-v1"
E450_PARITY_SPEC = CAMPAIGN / "specs/e450_seed4_native_parity_v1.sealed.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def zip_member_sha256(path: Path, member: str) -> str:
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive, archive.open(member) as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_timing():
    spec = importlib.util.spec_from_file_location("e462_timing_for_e450", E462_TIMING)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen E462 timing source: {E462_TIMING}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _require_e450_parity(readiness_path: Path, attempt_path: Path,
                         seed4_zip: Path, data_path: Path) -> dict[str, object]:
    if not E450_PARITY_SPEC.is_file():
        raise FileNotFoundError("sealed E450 native parity spec is missing")
    if sha256_file(seed4_zip) != EXPECTED_E450_ZIP_SHA256:
        raise ValueError("seed-4 package differs from the exact E450 parity package")
    readiness = json.loads(readiness_path.read_text())
    attempt = json.loads(attempt_path.read_text())
    sealed_spec = json.loads(E450_PARITY_SPEC.read_text())
    actual_spec_sha = hashlib.sha256(
        json.dumps(sealed_spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
    ).hexdigest()
    if (
        readiness.get("status") != "parity_passed_runtime_pending"
        or readiness.get("package", {}).get("sha256") != EXPECTED_E450_ZIP_SHA256
        or readiness.get("package", {}).get("precision") != "i16"
        or readiness.get("package", {}).get("members") != [{
            "weights": "weights_0.npz", "coef": 1.0,
            "source": "c20261004_combo_s4_final@ep3|c20261004_combo_head_s4_final@ep0",
        }]
        or readiness.get("fixed_input_sha256", {}).get("package") != EXPECTED_E450_ZIP_SHA256
        or readiness.get("fixed_input_sha256", {}).get("callback_dataset") != E457_DATA_SHA256
        or readiness.get("callback_dataset", {}).get("sha256") != E457_DATA_SHA256
        or readiness.get("callback_dataset", {}).get("sequences") != 20
        or readiness.get("callback_dataset", {}).get("rows_per_sequence") != 20_000
        or readiness.get("callback_parity", {}).get("passed") is not True
        or readiness.get("callback_parity", {}).get("sequences") != 20
        or readiness.get("callback_parity", {}).get("rows") != 400_000
        or readiness.get("callback_parity", {}).get("v2_v3_bitwise") is not True
        or readiness.get("callback_parity", {}).get("targets_masks_exact") is not True
        or readiness.get("callback_parity", {}).get("official_metric_id") != "global_wp_clip2_v1"
        or readiness.get("callback_parity", {}).get("max_i16_f32_error", float("inf")) > 0.0003
        or readiness.get("callback_parity", {}).get("wp_i16_f32_delta", float("inf")) > 0.000005
        or readiness.get("reset_and_causality", {}).get("reset_bitwise") is not True
        or readiness.get("reset_and_causality", {}).get("past_only_bitwise") is not True
        or readiness.get("reset_and_causality", {}).get("dataset_sha256") != E457_DATA_SHA256
        or attempt.get("id") != E450_PARITY_ATTEMPT
        or attempt.get("status") != "finished"
        or attempt.get("exit_code") != 0
        or attempt.get("teardown_verified") is not True
        or attempt.get("spec") != sealed_spec
        or attempt.get("spec_sha256") != actual_spec_sha
    ):
        raise ValueError("actual E450 full-stream parity readiness is absent or differs")
    artifact = next((item for item in attempt.get("artifacts", [])
                     if item.get("path") == readiness_path.relative_to(ROOT).as_posix()), None)
    if artifact is None or artifact.get("sha256") != sha256_file(readiness_path):
        raise ValueError("E450 parity attempt does not bind the actual readiness artifact")
    if sha256_file(data_path) != E457_DATA_SHA256:
        raise ValueError("timing callback data differs from E457's exact 20-stream panel")
    return {
        "readiness_sha256": sha256_file(readiness_path),
        "parity_attempt_sha256": sha256_file(attempt_path),
        "parity_spec_sha256": sha256_file(E450_PARITY_SPEC),
        "parity_spec_sha256_in_attempt": attempt.get("spec_sha256"),
        "package_sha256": sha256_file(seed4_zip),
        "data_sha256": sha256_file(data_path),
        "readiness_status": readiness["status"],
        "callback_parity": readiness["callback_parity"],
        "reset_and_causality": readiness["reset_and_causality"],
    }


def _quiet(cpu: int, max_load1: float) -> tuple[bool, dict[str, object]]:
    affinity = sorted(os.sched_getaffinity(0))
    load1 = os.getloadavg()[0]
    return (affinity == [cpu] and load1 <= max_load1), {
        "affinity": affinity, "loadavg": list(os.getloadavg()),
    }


def run(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    if sha256_file(args.control) != SEED3_ZIP_SHA256:
        raise ValueError("seed-3 control ZIP differs from the original E462 public combo")
    parity = _require_e450_parity(args.readiness, args.parity_attempt, args.seed4, args.data)
    timing = _load_frozen_timing()
    if sha256_file(E462_TIMING) != args.expected_e462_timing_sha256:
        raise ValueError("frozen E462 callback timing loop changed")
    quiet, quiet_state = _quiet(args.cpu, args.max_load1)
    if not quiet:
        raise RuntimeError(f"seed-4 timing lost its admitted quiet window before starting: {quiet_state}")
    streams, preload_seconds = timing._load_streams(args.data)
    if len(streams) != 3 or any(features.shape[0] != 20_000 for _, features, _ in streams):
        raise ValueError("frozen E462 loader did not return three complete 20k streams")
    warmup_rows = sum(int((~need).sum()) for _, _, need in streams)
    finite_prediction_rows = sum(int(need.sum()) for _, _, need in streams)
    if warmup_rows != 297 or finite_prediction_rows != 59_703:
        raise ValueError("timing streams do not have 99 warmup rows and 19,901 scored rows each")

    args.output.mkdir(parents=True, exist_ok=False)
    trials_path = args.output / "trials.jsonl"
    cycles = timing.ThreadCycles()
    trials: list[dict[str, object]] = []
    command = list(sys.argv)
    sources = {
        "frozen_e462_timing_module_sha256": sha256_file(E462_TIMING),
        "e450_seed4_readiness": parity,
        "seed3_zip_sha256": sha256_file(args.control),
        "seed4_zip_sha256": sha256_file(args.seed4),
        "seed3_v3_native_sha256": timing.zip_member_sha256(
            args.control, "v3/wnn_native.abi3.so"),
        "seed4_v3_native_sha256": zip_member_sha256(
            args.seed4, "v3/wnn_native.abi3.so"),
        "e457_data_sha256": sha256_file(args.data),
    }
    hardware = {
        "platform": platform.platform(), "python": sys.version,
        "affinity": sorted(os.sched_getaffinity(0)),
        "loadavg_start": list(os.getloadavg()),
        "preload_seconds_outside_timing": preload_seconds,
        "command": command,
    }
    if hardware["affinity"] != [args.cpu] or float(hardware["loadavg_start"][0]) > args.max_load1:
        raise RuntimeError("timing host left CPU15/load1 quiet window before first trial")

    packages = {"seed3_i16": args.control, "seed4_i16": args.seed4}
    try:
        with trials_path.open("x", encoding="utf-8") as journal:
            for repeat in range(args.repeats):
                order = (("seed3_i16", "seed4_i16") if repeat % 2 == 0
                         else ("seed4_i16", "seed3_i16"))
                for label in order:
                    quiet, quiet_state = _quiet(args.cpu, args.max_load1)
                    record: dict[str, object] = {
                        "comparison": "seed4_i16_vs_seed3_i16", "repeat": repeat,
                        "order": list(order), "candidate": label,
                        "zip_sha256": sha256_file(packages[label]),
                        "native_level": "v3",
                        "v3_native_sha256": zip_member_sha256(
                            packages[label], "v3/wnn_native.abi3.so"),
                        "streams": 3, "rows_per_stream": 20_000, "rows": 60_000,
                        "warmup_rows_per_stream": 99,
                        "warmup_rows_per_trial": warmup_rows,
                        "finite_prediction_rows": finite_prediction_rows,
                        "affinity": quiet_state["affinity"],
                        "loadavg_before": quiet_state["loadavg"],
                        "command": command,
                    }
                    try:
                        if not quiet:
                            raise RuntimeError("host left the CPU15/load1 quiet window")
                        result = timing._trial(label, packages[label], streams, cycles)
                        if result.get("rows") != 60_000 or not result.get("elapsed_seconds", 0) > 0:
                            raise ValueError("frozen E462 trial did not return the full 60k-row callback")
                        record.update(result)
                        record["status"] = "passed"
                    except Exception as error:
                        record.update(status="failed", error=f"{type(error).__name__}: {error}")
                        journal.write(json.dumps(record, sort_keys=True) + "\n")
                        journal.flush()
                        os.fsync(journal.fileno())
                        raise
                    journal.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    trials.append(record)
    except BaseException as error:
        failure = {
            "schema": "e450-seed4-i16-timing-v1", "status": "failed_closed",
            "error": f"{type(error).__name__}: {error}", "source_hashes": sources,
            "hardware": hardware, "completed_trial_count": len(trials),
            "trials_file": "trials.jsonl",
        }
        (args.output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        cycles.close()

    hardware["loadavg_end"] = list(os.getloadavg())
    summaries = {}
    for label in ("seed3_i16", "seed4_i16"):
        values = [float(row["us_per_row"]) for row in trials if row["candidate"] == label]
        summaries[label] = {
            "trials": len(values), "median_us_per_row": statistics.median(values),
            "all_us_per_row": values,
            "all_kcycles_per_row": [row["kcycles_per_row"] for row in trials
                                    if row["candidate"] == label],
            "median_kcycles_per_row": statistics.median(
                [float(row["kcycles_per_row"]) for row in trials
                 if row["candidate"] == label and row["kcycles_per_row"] is not None]
            ) if any(row["kcycles_per_row"] is not None for row in trials
                     if row["candidate"] == label) else None,
            "projected_39m_rows_seconds": statistics.median(values) * 39_000_000 / 1_000_000,
            "projected_39m_rows_under_52min": (
                statistics.median(values) * 39_000_000 / 1_000_000 <= args.projection_limit_seconds
            ),
        }
    by_repeat = {
        repeat: {row["candidate"]: float(row["us_per_row"])
                 for row in trials if row["repeat"] == repeat}
        for repeat in range(args.repeats)
    }
    if any(set(pair) != {"seed3_i16", "seed4_i16"} for pair in by_repeat.values()):
        raise RuntimeError("seed-4 timing is missing a complete matched pair")
    report = {
        "schema": "e450-seed4-i16-timing-v1", "status": "timing_complete",
        "passed": len(trials) == args.repeats * 2,
        "comparison": "fixed original seed-3 i16 ZIP versus the actual fixed seed-4 i16 ZIP",
        "paired_repeats": args.repeats, "order": "alternating, starting seed3 then seed4",
        "sequences_per_trial": 3, "rows_per_trial": 60_000,
        "warmup_rows_per_stream": 99,
        "finite_prediction_rows_per_trial": 59_703,
        "native_level": "v3", "source_hashes": sources, "hardware": hardware,
        "summaries": summaries, "matched_pairs": by_repeat,
        "median_seed4_minus_seed3_us_per_row": statistics.median(
            pair["seed4_i16"] - pair["seed3_i16"] for pair in by_repeat.values()),
        "median_seed4_over_seed3": statistics.median(
            pair["seed4_i16"] / pair["seed3_i16"] for pair in by_repeat.values()),
        "trials_file": "trials.jsonl", "score_measured": False,
        "scope": "quiet CPU callback timing diagnostic; not a score or manual-upload readiness decision",
    }
    with (args.output / "timing_report.json").open("x", encoding="utf-8") as output:
        output.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--seed4", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--parity-attempt", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "cache/valid_subset20.parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-e462-timing-sha256", required=True)
    parser.add_argument("--cpu", type=int, default=15)
    parser.add_argument("--max-load1", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, choices=(5,), default=5)
    parser.add_argument("--projection-limit-seconds", type=int, default=3120)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
