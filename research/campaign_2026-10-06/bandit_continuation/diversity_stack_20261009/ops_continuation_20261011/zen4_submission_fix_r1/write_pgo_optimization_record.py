#!/usr/bin/env python3
"""Bind Rust PGO build inputs and exact instrumented training-panel evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def tool_version(command: list[str]) -> str:
    """Return the first line of a tool version command."""
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return (result.stdout or result.stderr).splitlines()[0]


def main() -> None:
    """Write a create-only PGO provenance record after the profile-use build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--instrumented-zip", type=Path, required=True)
    parser.add_argument("--instrumented-receipt", type=Path, required=True)
    parser.add_argument("--training-panel", type=Path, required=True)
    parser.add_argument("--baseline-panel", type=Path, required=True)
    parser.add_argument("--profile-directory", type=Path, required=True)
    parser.add_argument("--merged-profile", type=Path, required=True)
    parser.add_argument("--final-library", type=Path, required=True)
    parser.add_argument("--target-directory", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    native_source = args.native_source.resolve(strict=True)
    base_zip = args.base_zip.resolve(strict=True)
    base_receipt = args.base_receipt.resolve(strict=True)
    instrumented_zip = args.instrumented_zip.resolve(strict=True)
    instrumented_receipt = args.instrumented_receipt.resolve(strict=True)
    training_panel = args.training_panel.resolve(strict=True)
    baseline_panel = args.baseline_panel.resolve(strict=True)
    profile_directory = args.profile_directory.resolve(strict=True)
    merged_profile = args.merged_profile.resolve(strict=True)
    final_library = args.final_library.resolve(strict=True)
    target_directory = args.target_directory.resolve(strict=True)

    base_build = load_json(base_receipt)
    instrumented_build = load_json(instrumented_receipt)
    training = load_json(training_panel)
    baseline = load_json(baseline_panel)
    if base_build["artifact"]["sha256"] != sha256_file(base_zip):
        raise ValueError("base ZIP hash does not match its build receipt")
    if instrumented_build["artifact"]["sha256"] != sha256_file(instrumented_zip):
        raise ValueError("instrumented ZIP hash does not match its build receipt")
    if training["submission_sha256"] != sha256_file(instrumented_zip):
        raise ValueError("training panel was not run against the instrumented ZIP")
    if baseline["submission_sha256"] != sha256_file(base_zip):
        raise ValueError("baseline panel was not run against the exact base ZIP")
    if training["runtime"]["precision"] != "q8vnni":
        raise ValueError("instrumented training run did not load q8vnni")
    if training["workload"]["data_sha256"] != baseline["workload"]["data_sha256"]:
        raise ValueError("training and baseline data differ")
    if training["workload"]["sequence_ids"] != baseline["workload"]["sequence_ids"]:
        raise ValueError("training and baseline sequence identities differ")
    if (
        training["workload"]["features_sha256_by_sequence"]
        != baseline["workload"]["features_sha256_by_sequence"]
    ):
        raise ValueError("training and baseline input sequences differ")
    if (
        training["workload"]["predictions_sha256_by_sequence"]
        != baseline["workload"]["predictions_sha256_by_sequence"]
    ):
        raise ValueError("instrumented build changed exact predictions on the fixed panel")

    raw_profiles = sorted(profile_directory.glob("*.profraw"))
    if not raw_profiles or any(path.stat().st_size == 0 for path in raw_profiles):
        raise ValueError("instrumented run produced no nonempty raw PGO profile")
    if merged_profile.stat().st_size == 0:
        raise ValueError("merged PGO profile is empty")
    source_files = sorted(path for path in native_source.rglob("*") if path.is_file())
    source_hashes = {
        path.relative_to(native_source).as_posix(): sha256_file(path) for path in source_files
    }
    raw_hashes = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in raw_profiles
    }
    record = {
        "schema": "wnn.native-rust-pgo-build.v1",
        "stage": "profile-use-build",
        "method": "Rust LLVM instrumentation PGO using the exact candidate callback path",
        "base": {
            "zip_path": str(base_zip),
            "zip_sha256": sha256_file(base_zip),
            "build_receipt_path": str(base_receipt),
            "build_receipt_sha256": sha256_file(base_receipt),
        },
        "instrumented_training": {
            "zip_path": str(instrumented_zip),
            "zip_sha256": sha256_file(instrumented_zip),
            "build_receipt_sha256": sha256_file(instrumented_receipt),
            "panel_receipt_path": str(training_panel),
            "panel_receipt_sha256": sha256_file(training_panel),
            "callback_count": training["timing"]["total_callback_calls"],
            "native_v3_library_sha256": instrumented_build["runtime_optimization"][
                "new_native_library_sha256"
            ],
            "prediction_hashes_match_base": True,
        },
        "baseline_panel": {
            "receipt_path": str(baseline_panel),
            "receipt_sha256": sha256_file(baseline_panel),
            "median_callback_us_per_row": baseline["timing"]["median_repeat_us_per_row"],
            "prediction_hashes": baseline["workload"]["predictions_sha256_by_sequence"],
        },
        "pgo_data": {
            "raw_profiles": raw_hashes,
            "merged_profile_path": str(merged_profile),
            "merged_profile_bytes": merged_profile.stat().st_size,
            "merged_profile_sha256": sha256_file(merged_profile),
        },
        "final_native_library": {
            "path": str(final_library),
            "bytes": final_library.stat().st_size,
            "sha256": sha256_file(final_library),
        },
        "build": {
            "target": "x86_64-unknown-linux-gnu",
            "target_cpu": "x86-64-v3",
            "rustflags": [
                "-C target-cpu=x86-64-v3",
                f"-C profile-use={merged_profile}",
                "-C llvm-args=-pgo-warn-missing-function",
            ],
            "cargo_target_directory": str(target_directory),
            "cargo_build_jobs": 1,
            "rustc": tool_version(["rustc", "--version"]),
            "cargo": tool_version(["cargo", "--version"]),
            "python": platform.python_version(),
            "host_cpu": next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unknown",
            ),
        },
        "native_source_sha256": source_hashes,
        "claim_scope": "A local compiler/runtime experiment only; no Zen4 timing, Kaggle runtime, score, or public outcome is inferred.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"record": str(output), "sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
