"""Prepare verified shared inputs for the authorized fixed eight-fit L4 panel."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def write_receipt(path: Path, state: dict[str, object]) -> None:
    """Publish a small atomic operational receipt without changing source data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_phase(source: Path, data: Path, name: str, arguments: list[str], logs: Path) -> None:
    """Execute one existing canonical producer and retain its complete output."""
    environment = dict(os.environ, PYTHONPATH=str(source / "src"), CUDA_VISIBLE_DEVICES="")
    with (logs / f"{name}.log").open("xb") as log:
        subprocess.run(
            [sys.executable, *arguments],
            cwd=source,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def prewarm_normalization(source: Path, data_root: Path, split: Path) -> dict[str, object]:
    """Compute train-only raw112/event211 moments once, without constructing weights."""
    sys.path.insert(0, str(source / "src"))
    from wnn.training.sequence_config import load_config
    from wnn.training.sequence_data import build_data

    config_root = source / "research/experiment_drafts/galam-l4-quality8-20261028/launch/configs"
    results: dict[str, object] = {}
    for name in ("g1-galam-lstm64", "g0-plain-gru256x2"):
        started = time.monotonic()
        config = load_config(config_root / f"{name}.json")
        data = build_data(config, data_root, split)
        mean, std = data.normalization()
        mode = str(config["model"].get("input_feature_mode", "raw112"))
        results[mode] = {
            "elapsed_seconds": time.monotonic() - started,
            "fit_sequences": len(data.fit),
            "validation_sequences": len(data.evaluation),
            "feature_count": mean.numel(),
            "normalization": data.provenance["normalization"],
            "std_min": float(std.min()),
            "std_max": float(std.max()),
        }
    return results


def main() -> None:
    """Wait for finalized data, then audit, index, build membership and prewarm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=10_800)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not 1 <= args.wait_seconds <= 14_400:
        parser.error("workers must be1..16 and wait-seconds1..14400")
    source, data, output = (
        args.source_root.resolve(),
        args.data_root.resolve(),
        args.output.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    split = output / "chronological_primary.json"
    receipt = output / "preparation.json"
    state: dict[str, object] = {
        "schema": "tabdata.fixed-eight-data-preparation.v1",
        "pid": os.getpid(),
        "started_utc": dt.datetime.now(dt.UTC).isoformat(),
        "source_root": str(source),
        "data_root": str(data),
        "status": "waiting_for_finalized_train",
        "scientific_fits_started": 0,
        "phases": [],
    }
    write_receipt(receipt, state)
    started = time.monotonic()
    try:
        while not (data / "datasets/train.parquet").is_file():
            if time.monotonic() - started > args.wait_seconds:
                raise TimeoutError("finalized train.parquet did not arrive before the wait bound")
            time.sleep(15)
        commands = [
            ("source_audit", ["scripts/audit_chronological_sources.py", "--root", str(data)]),
            (
                "train_index",
                [
                    "scripts/cache_targets.py",
                    "train",
                    "--source-qualified",
                    "--root",
                    str(data),
                    "--expected-bytes",
                    "29240219011",
                    "--expected-row-groups",
                    "10607",
                    "--expected-sha256",
                    "4a7756449a4bcf272cf715bb405a2152987041605f6c1cce650243ab2abf4e81",
                ],
            ),
            (
                "valid_index",
                [
                    "scripts/cache_targets.py",
                    "valid",
                    "--source-qualified",
                    "--root",
                    str(data),
                    "--expected-bytes",
                    "5615509128",
                    "--expected-row-groups",
                    "1873",
                    "--expected-sha256",
                    "2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782",
                ],
            ),
            (
                "component_graph",
                [
                    "scripts/sequence_chains.py",
                    "--source-qualified",
                    "--root",
                    str(data),
                    "--workers",
                    str(args.workers),
                ],
            ),
            (
                "primary_split",
                [
                    "scripts/build_chronological_primary_split.py",
                    "--root",
                    str(data),
                    "--output",
                    str(split),
                ],
            ),
        ]
        phases: list[dict[str, object]] = []
        for name, command in commands:
            state["status"] = name
            write_receipt(receipt, state)
            phase_started = time.monotonic()
            run_phase(source, data, name, command, output)
            phases.append({"name": name, "elapsed_seconds": time.monotonic() - phase_started})
            state["phases"] = phases
            write_receipt(receipt, state)
        state["status"] = "normalization"
        write_receipt(receipt, state)
        state["normalization"] = prewarm_normalization(source, data, split)
        state["split_manifest"] = str(split)
        state["split_sha256"] = hashlib.sha256(split.read_bytes()).hexdigest()
        state["status"] = "ready_for_actual_data_profiles"
        state["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
        state["elapsed_seconds"] = time.monotonic() - started
        write_receipt(receipt, state)
    except BaseException as error:
        state["failed_phase"] = state["status"]
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        write_receipt(receipt, state)
        raise


if __name__ == "__main__":
    main()
