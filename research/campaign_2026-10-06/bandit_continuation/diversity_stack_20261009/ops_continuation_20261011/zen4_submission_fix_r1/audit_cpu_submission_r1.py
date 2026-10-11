#!/usr/bin/env python3
"""Create a bounded, hash-bound report for the CPU submission experiment."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any

WORKTREE = Path(__file__).resolve().parents[6]
MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "CPU_SUBMISSION_FIX_EVIDENCE_R3.json"
SUPERSEDED_OUTPUT = HERE / "CPU_SUBMISSION_FIX_EVIDENCE_R2.json"

BASE_ZIP = (
    MAIN / "submissions/10-11-exploit-now_r2/h390_student_h388_teacher_pair_q8/submission_r1.zip"
)
BASE_BUILD = BASE_ZIP.parent / "BUILD_RECEIPT_R1.json"
PGO_ZIP = (
    WORKTREE / "submissions/zen4_submission_fix_r1/current11_h390_h388_q8_pgo_r1/submission.zip"
)
PGO_BUILD = PGO_ZIP.parent / "BUILD_RECEIPT_R1.json"
PGO_BUILD_RECORD = HERE / "PGO_OPTIMIZATION_RECORD_R1.json"
PORTAL_TIMEOUT = (
    MAIN
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/PORTAL_TRIO_TIMEOUT_20261012_R1.json"
)
PUBLIC_SCORE_RECORD = (
    MAIN
    / "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/PUBLIC_SCORES_20261011.md"
)


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def archive_inventory(path: Path) -> dict[str, str]:
    """Check ZIP CRC and return member SHA-256 values."""
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"CRC failure in {path}: {bad_member}")
        return {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}


def elf_requirements(path: Path) -> dict[str, Any]:
    """Read dynamic dependencies and maximum GLIBC symbol version."""
    dynamic = subprocess.run(
        ["readelf", "-d", str(path)], check=True, capture_output=True, text=True
    ).stdout
    versions = subprocess.run(
        ["readelf", "--version-info", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    dependencies = sorted(set(re.findall(r"Shared library: \[([^]]+)\]", dynamic)))
    glibc_versions = {
        tuple(map(int, match.split("_", 1)[1].split(".")))
        for match in re.findall(r"GLIBC_\d+\.\d+", versions)
    }
    if not glibc_versions:
        raise ValueError(f"no GLIBC version references found in {path}")
    maximum = max(glibc_versions)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "needed_libraries": dependencies,
        "maximum_glibc_symbol": ".".join(map(str, maximum)),
    }


def panel_record(filename: str) -> dict[str, Any]:
    """Summarize a fixed benchmark receipt and bind its bytes."""
    path = HERE / filename
    value = read_json(path)
    timing = value["timing"]
    runtime = value["runtime"]
    workload = value["workload"]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "zip_sha256": value["submission_sha256"],
        "zip_bytes": value["submission_bytes"],
        "precision": runtime["precision"],
        "native_level": runtime["native_level"],
        "member_count": runtime["model_count"],
        "cpu_model": runtime["cpu_model"],
        "cpu_affinity": runtime["affinity_used"],
        "cpu_max": runtime["cgroup_after_measure"]["cpu.max"],
        "memory_peak_bytes": int(runtime["cgroup_after_measure"]["memory.peak"]),
        "member_storage_bytes": runtime.get("member_storage_bytes"),
        "data_sha256": workload["data_sha256"],
        "row_groups": workload["row_groups"],
        "sequence_ids": workload["sequence_ids"],
        "sequence_lengths": workload["sequence_lengths"],
        "need_prediction_rows_each": workload["need_prediction_rows_each"],
        "repeats": workload["repeats"],
        "callback_count": timing["total_callback_calls"],
        "median_us_per_row": timing["median_repeat_us_per_row"],
        "post_decode_wall_seconds": timing["elapsed_wall_seconds_after_sequence_decode"],
        "reset_replay_equal": workload["predictions_bitwise_identical_across_replays"],
        "prediction_hashes": workload.get("predictions_sha256_by_sequence"),
    }


def main() -> None:
    """Verify package identity, output equality, ABI floor, and evidence scope."""
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")

    base_build = read_json(BASE_BUILD)
    pgo_build = read_json(PGO_BUILD)
    pgo_record = read_json(PGO_BUILD_RECORD)
    base_members = archive_inventory(BASE_ZIP)
    pgo_members = archive_inventory(PGO_ZIP)
    if base_build["artifact"]["sha256"] != sha256_file(BASE_ZIP):
        raise ValueError("baseline ZIP does not match its build receipt")
    if pgo_build["artifact"]["sha256"] != sha256_file(PGO_ZIP):
        raise ValueError("PGO ZIP does not match its build receipt")
    if set(base_members) != set(pgo_members):
        raise ValueError("baseline and PGO ZIP member sets differ")
    changed_members = sorted(
        name for name in base_members if base_members[name] != pgo_members[name]
    )
    if changed_members != ["v3/wnn_native.abi3.so"]:
        raise ValueError(f"unexpected PGO ZIP payload changes: {changed_members}")
    if base_members["solution.py"] != pgo_members["solution.py"]:
        raise ValueError("PGO package changed the callback source")
    if base_members["precision.txt"] != pgo_members["precision.txt"]:
        raise ValueError("PGO package changed declared precision")
    if base_build["fixed_members"] != pgo_build["fixed_members"]:
        raise ValueError("PGO package changed the fixed model blend")

    panel_names = [
        "bench_current11_baseline_hash_r1.json",
        "bench_current11_pgo_training_r1.json",
        "bench_current11_baseline_alt1.json",
        "bench_current11_pgo_alt1.json",
        "bench_current11_baseline_alt2.json",
        "bench_current11_pgo_alt2.json",
        "bench_h388_s4s5_pair_r1.json",
        "bench_trio12_r1.json",
        "bench_4_r3.json",
        "bench_5_r1.json",
    ]
    panels = {Path(name).stem: panel_record(name) for name in panel_names}
    base_training = read_json(HERE / "bench_current11_baseline_hash_r1.json")
    pgo_training = read_json(HERE / "bench_current11_pgo_training_r1.json")
    if (
        base_training["workload"]["predictions_sha256_by_sequence"]
        != pgo_training["workload"]["predictions_sha256_by_sequence"]
    ):
        raise ValueError("instrumented training build changed the fixed panel outputs")
    baseline_alt1 = read_json(HERE / "bench_current11_baseline_alt1.json")
    pgo_alt1 = read_json(HERE / "bench_current11_pgo_alt1.json")
    baseline_alt2 = read_json(HERE / "bench_current11_baseline_alt2.json")
    pgo_alt2 = read_json(HERE / "bench_current11_pgo_alt2.json")
    for baseline, optimized in ((baseline_alt1, pgo_alt1), (baseline_alt2, pgo_alt2)):
        baseline_hashes = baseline["workload"].get("predictions_sha256_by_sequence")
        optimized_hashes = optimized["workload"].get("predictions_sha256_by_sequence")
        if not isinstance(baseline_hashes, dict) or len(baseline_hashes) != 10:
            raise ValueError("baseline paired panel lacks ten prediction hashes")
        if not isinstance(optimized_hashes, dict) or len(optimized_hashes) != 10:
            raise ValueError("PGO paired panel lacks ten prediction hashes")
        if baseline_hashes != optimized_hashes:
            raise ValueError("paired PGO panel changed a per-sequence prediction")
        if baseline["workload"]["sequence_ids"] != optimized["workload"]["sequence_ids"]:
            raise ValueError("paired PGO panel sequence identities differ")
        if baseline["workload"]["row_groups"] != optimized["workload"]["row_groups"]:
            raise ValueError("paired PGO panel row groups differ")
        if baseline["workload"]["sequence_lengths"] != [20_000] * 10:
            raise ValueError("baseline panel is not ten complete 20k sequences")
        if optimized["workload"]["sequence_lengths"] != [20_000] * 10:
            raise ValueError("PGO panel is not ten complete 20k sequences")
        if baseline["workload"]["data_sha256"] != optimized["workload"]["data_sha256"]:
            raise ValueError("paired PGO panel data differ")
        if baseline["timing"]["total_callback_calls"] != 400_000:
            raise ValueError("unexpected baseline paired callback count")
        if optimized["timing"]["total_callback_calls"] != 400_000:
            raise ValueError("unexpected PGO paired callback count")
        for panel in (baseline, optimized):
            if panel["runtime"]["precision"] != "q8vnni":
                raise ValueError("panel did not load actual Q8VNNI precision")
            if panel["runtime"]["native_level"] != "v3":
                raise ValueError("panel did not load the v3 native library")
            if not panel["runtime"]["python"].startswith("3.11."):
                raise ValueError("paired panel did not run under Python 3.11")
            if panel["workload"]["predictions_bitwise_identical_across_replays"] is not True:
                raise ValueError("panel failed bitwise replay determinism")
            if panel["runtime"]["cgroup_after_measure"]["cpu.max"] != "100000 100000":
                raise ValueError("panel was not bounded to one CPU")

    base_elf = elf_requirements(Path("/tmp/wnn-zen4-pgo-current11-r1/base-v3.so"))
    pgo_elf = elf_requirements(
        Path(
            "/tmp/wnn-zen4-pgo-current11-r1/target/x86_64-unknown-linux-gnu/release/libwnn_native.so"
        )
    )
    if base_elf["sha256"] != base_build["artifact"]["members"]["v3/wnn_native.abi3.so"]:
        raise ValueError("baseline ELF audited is not the exact ZIP library")
    if pgo_elf["sha256"] != pgo_build["artifact"]["members"]["v3/wnn_native.abi3.so"]:
        raise ValueError("PGO ELF audited is not the exact ZIP library")
    if base_elf["needed_libraries"] != pgo_elf["needed_libraries"]:
        raise ValueError("PGO library introduced or removed a dynamic dependency")
    if tuple(map(int, base_elf["maximum_glibc_symbol"].split("."))) > (2, 36):
        raise ValueError("baseline library exceeds Debian Bookworm GLIBC")
    if tuple(map(int, pgo_elf["maximum_glibc_symbol"].split("."))) > (2, 36):
        raise ValueError("PGO library requires a newer GLIBC than Debian Bookworm")

    paired = []
    for suffix in ("alt1", "alt2"):
        baseline = panels[f"bench_current11_baseline_{suffix}"]["median_us_per_row"]
        optimized = panels[f"bench_current11_pgo_{suffix}"]["median_us_per_row"]
        paired.append(
            {
                "pair": suffix,
                "base_us_per_row": baseline,
                "pgo_us_per_row": optimized,
                "pgo_delta_percent": (optimized / baseline - 1.0) * 100.0,
            }
        )

    input_manifest = {
        "base_zip_sha256": sha256_file(BASE_ZIP),
        "base_build_receipt_sha256": sha256_file(BASE_BUILD),
        "pgo_zip_sha256": sha256_file(PGO_ZIP),
        "pgo_build_receipt_sha256": sha256_file(PGO_BUILD),
        "pgo_build_record_sha256": sha256_file(PGO_BUILD_RECORD),
        "benchmark_runner_sha256": sha256_file(
            WORKTREE / "scripts/benchmark_submission_callbacks.py"
        ),
        "rust_source_tree_sha256": pgo_record["native_source_sha256"],
        "user_reported_portal_timeout_sha256": sha256_file(PORTAL_TIMEOUT),
        "historical_public_scores_sha256": sha256_file(PUBLIC_SCORE_RECORD),
    }
    base_bytes = BASE_ZIP.stat().st_size
    pgo_bytes = PGO_ZIP.stat().st_size
    if base_bytes >= 20_000_000 or pgo_bytes >= 20_000_000:
        raise ValueError("two-member package exceeds the 20 MB competition limit")
    report = {
        "schema": "wnn.cpu-submission-fix-evidence.v1",
        "supersedes_receipt": {
            "path": str(SUPERSEDED_OUTPUT),
            "sha256": sha256_file(SUPERSEDED_OUTPUT),
            "reason": "R3 also requires ten per-sequence output hashes, complete sequence shapes, Python 3.11, and package-size bounds.",
        },
        "decision": "Do not promote the PGO package as a runtime optimization; retain the unchanged current11 package as the faster measured two-member candidate, with host/portal-transfer limitations explicit.",
        "input_manifest": input_manifest,
        "artifacts": {
            "baseline_two_member": {
                "path": str(BASE_ZIP),
                "sha256": sha256_file(BASE_ZIP),
                "bytes": base_bytes,
                "build_receipt_sha256": sha256_file(BASE_BUILD),
                "member_count": 2,
                "blend": base_build["fixed_members"],
                "size_under_20mb": base_bytes < 20_000_000,
            },
            "pgo_two_member": {
                "path": str(PGO_ZIP),
                "sha256": sha256_file(PGO_ZIP),
                "bytes": pgo_bytes,
                "build_receipt_sha256": sha256_file(PGO_BUILD),
                "member_count": 2,
                "changed_zip_members": changed_members,
                "size_under_20mb": pgo_bytes < 20_000_000,
                "same_fixed_blend": True,
                "same_solution_and_precision_files": True,
            },
            "local_architecture_pair": {
                "path": str(
                    MAIN
                    / "submissions/10-11-exploit-now_r2/h390_student_h388_teacher_pair_q8/submission_r1.zip"
                ),
                "sha256": sha256_file(BASE_ZIP),
                "members": base_build["fixed_members"],
                "ancestry": "All-data homogeneous H390 student s3@ep5 paired with H388 teacher s5@ep2; full-data package has no held-out score claim.",
            },
        },
        "panels": panels,
        "pgo_paired_comparisons": paired,
        "member_storage_bytes": panels["bench_current11_baseline_alt1"]["member_storage_bytes"],
        "elf_abi": {
            "baseline": base_elf,
            "pgo": pgo_elf,
            "debian_bookworm_glibc_ceiling": "2.36",
            "pgo_maximum_glibc_compatible": True,
            "dependencies_unchanged": True,
            "python311_import_and_exact_callback_panel_passed": True,
        },
        "portal_evidence": {
            "timeout_receipt_path": str(PORTAL_TIMEOUT),
            "submission_id": read_json(PORTAL_TIMEOUT)["submission_id"],
            "zip_sha256": read_json(PORTAL_TIMEOUT)["zip_sha256"],
            "reported_status": read_json(PORTAL_TIMEOUT)["portal_status"],
            "reported_seconds": read_json(PORTAL_TIMEOUT)["portal_duration_seconds"],
            "reported_memory_mb": read_json(PORTAL_TIMEOUT)["portal_memory_mb"],
            "censored_completion": read_json(PORTAL_TIMEOUT)["completion_time_censored"],
            "historical_public_score_record_path": str(PUBLIC_SCORE_RECORD),
            "historical_public_score_record_is_separate_transfer_evidence": True,
        },
        "interpretation": {
            "paired_pgo_result": "Both paired comparisons are slightly slower with PGO (+0.22% and +0.14%); larger unpaired swings track host load and do not establish a gain.",
            "precision_and_determinism": "Both packages load the shipped Q8VNNI native v3 runtime and produce bitwise-equal predictions on the fixed ten-sequence panel; resets replay bitwise identically.",
            "runtime_transfer": "Local timing is from a one-core Intel i9-11900H, not the competition EPYC 9224. Prior portal timings are not a direct runtime guarantee for these new members.",
            "quality": "This is a serving-capacity experiment, not a grade. The H390/H388 package is full-data trained; do not present its exposed fold1 prototype or old public score as a held-out score for this ZIP.",
            "trio": "The three-H388 Q8/F16-transport candidate that the user uploaded timed out at 61m56 on Kaggle. Its fixed local panel is about 2x the current two-member panel callback time; no recommendation to re-upload it.",
            "head_skip": "The measured sequences require prediction on 19,901 of 20,000 rows each, so skipping the stateless head on unrequested rows can affect at most 0.5% of callback rows and is not a timeout remedy.",
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha256_file(OUTPUT),
                "changed_zip_members": changed_members,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
