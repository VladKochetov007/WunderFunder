"""Seal the measured four-teacher production amendment after independent review."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

ROOT = Path("/data/epanov_s/tabdata")
SOURCE = ROOT / "runtime/pr32-source-f1beeb4c"
CAMPAIGN = ROOT / "runs/pr32-blackwell-20261024"
DESTINATION = ROOT / "runs/pr32-production-admission-20261024-r2"
COMMIT = "f1beeb4c210a9da01735e03b1ff6e27be878b873"


def sha(path: Path) -> str:
    """Hash small source/config/receipt files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    """Create a new immutable admission artifact."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    """Bind profiles, resource estimates and exact launch inputs into one plan."""
    os.environ["CUBLAS_EMULATE_SINGLE_PRECISION"] = "1"
    sys.path.insert(0, str(SOURCE / "src"))
    from wnn.training.sequence_config import load_config
    from wnn.training.sequence_provenance import (
        canonical_sha256,
        source_identity,
        validate_profile_receipt,
    )

    manifest = SOURCE / "TRANSPORT_MANIFEST.json"
    transport = json.loads(manifest.read_text())
    assert transport["commit"] == COMMIT
    for name, expected in transport["files"].items():
        assert sha(SOURCE / name) == expected
    identity = source_identity(SOURCE, SOURCE / "scripts/train_sequence.py")
    numerical = json.loads(
        (CAMPAIGN / "acceptance-ad876fa0/compile-numeric-r4c.json").read_text()
    )
    assert numerical["exact_future_prefix"] is True
    assert (
        numerical["comparisons"]["compiled"]["muon_optimizer_state_fp32_finite"] is True
    )
    DESTINATION.mkdir(exist_ok=False)
    jobs, measured = [], []
    host_memory_estimate = 0
    for cell in ("t1280_f0", "t1792_f0", "t1280_f2", "t1792_f2"):
        directory = CAMPAIGN / "repaired-profile-w2" / cell
        config_path = directory / "config.json"
        config = load_config(config_path)
        profile_path = directory / "profile/profile_receipt.json"
        profile = json.loads(profile_path.read_text())
        operator = json.loads((directory / "operator.json").read_text())
        assert operator["exit_code"] == 0 and operator["source_commit"] == COMMIT
        assert profile["updates"] == 12 and profile["errors"] == []
        assert profile["identity"]["source_sha256"] == identity["sha256"]
        assert profile["identity"]["config_sha256"] == canonical_sha256(config)
        assert profile["identity"]["runtime"]["compiler"]["recompile_limit"] == 64
        log = (directory / "console.log").read_text()
        assert (
            "hit config.recompile_limit" not in log
            and "hit config.cache_size_limit" not in log
        )
        assert all(
            item["peak_reserved_bytes"] < 0.8 * item["device_total_memory_bytes"]
            for item in profile["memory_by_rank"]
        )
        profile["approved"] = True
        profile["receiving_operator_approval"] = {
            "revision": "pr32-blackwell-compiled-r2",
            "source_commit": COMMIT,
            "review": "Independent numerical and execution review by next_b300_pr36_audit; operator review by next_b300_pr35_audit; root admission.",
            "scope": "Finite source-bound engineering admission; not FP32 reproduction or model-quality equivalence.",
            "numerical_receipt_sha256": sha(
                CAMPAIGN / "acceptance-ad876fa0/compile-numeric-r4c.json"
            ),
        }
        approved = DESTINATION / (cell + "-approved.json")
        write(approved, profile)
        validate_profile_receipt(approved, profile["identity"])
        fold = config["expected_fold"]
        split = ROOT / "splits" / f"fold{fold}.json"
        jobs.append(
            {
                "name": cell,
                "config": str(config_path),
                "config_file_sha256": sha(config_path),
                "config_sha256": canonical_sha256(config),
                "split_manifest": str(split),
                "split_sha256": sha(split),
                "profile_receipt": str(approved),
                "profile_sha256": sha(approved),
                "gpu_uuids": [row[1] for row in operator["gpus"]],
                "cpus": operator["cpus"],
                "max_seconds": 7 * 86400,
            }
        )
        ranks = []
        for rank in range(2):
            records = [
                json.loads(line)
                for line in (directory / "profile" / f"rank{rank}.jsonl")
                .read_text()
                .splitlines()
            ]
            ranks.append({row["step"]: row["window_seconds"] for row in records})
        per_step = [
            max(ranks[rank][step] for rank in range(2)) for step in range(6, 13)
        ]
        median = statistics.median(per_step)
        horizon = profile["identity"]["optimizer"]["total_updates"]
        measured.append(
            {
                "cell": cell,
                "world_size": 2,
                "per_rank_batch": 8,
                "global_batch_sequences": 16,
                "steady_seconds_per_update": median,
                "optimizer_updates": horizon,
                "fit_hours_from_steady_windows": median * horizon / 3600,
                "peak_reserved_gib_per_gpu": profile["peak_reserved_bytes"] / 2**30,
                "profile_elapsed_seconds_including_compile": profile["elapsed_seconds"],
            }
        )
        parameters = profile["model_parameters"]["total"]
        host_memory_estimate += 12 * parameters * 2 + sum(
            profile["host_peak_rss_bytes"]
        )
    assert host_memory_estimate < 0.8 * 512 * 2**30
    assert shutil.disk_usage(ROOT / "runs").free > 200 * 2**30
    data_files = {}
    for name, receipt in (
        ("train.parquet", "data-train.parquet-v2.json"),
        ("valid.parquet", "data-valid.parquet-v1.json"),
    ):
        data_files["datasets/" + name] = json.loads(
            (ROOT / "receipts" / receipt).read_text()
        )["sha256"]
    plan = {
        "schema": "wnn.pr32.teacher-cohort.v1",
        "revision": "pr32-blackwell-compiled-r2",
        "source": {
            "root": str(SOURCE),
            "commit": COMMIT,
            "manifest_path": str(manifest),
            "manifest_sha256": sha(manifest),
        },
        "data_root": str(ROOT / "data"),
        "data_files": data_files,
        "run_root": str(ROOT / "runs/pr32-teachers-20261024-r2"),
        "tmp_root": str(ROOT / "tmp/pr32-production-20261024-r2"),
        "jobs": jobs,
    }
    write(DESTINATION / "plan.json", plan)
    estimate = {
        "measured_cells": measured,
        "node_steady_fit_hours": max(
            row["fit_hours_from_steady_windows"] for row in measured
        ),
        "planned_host_ram_gib": 512,
        "conservative_host_staging_plus_profile_rss_gib": host_memory_estimate / 2**30,
        "planned_disk_gib": 200,
        "logical_cpus": 32,
        "precision": "BF16 model compute; FP32 master/grad/state; process-global BF16x9 FP32 emulation; compiled blocks; exact causal efficient attention",
        "quality_equivalence_claim": False,
    }
    write(DESTINATION / "RESOURCE_ESTIMATE.json", estimate)
    print(
        json.dumps(
            {
                "plan": str(DESTINATION / "plan.json"),
                "plan_sha256": sha(DESTINATION / "plan.json"),
                "estimate": estimate,
            }
        )
    )


if __name__ == "__main__":
    main()
