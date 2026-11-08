"""Freeze one source-bound eight-cell receiving panel without launching work."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

from receiving_checks import (
    FAMILIES,
    SOURCE_INVENTORY_SHA,
    SPLIT_SHA,
    runtime_recipe,
    source_files,
    verify_source,
)

from wnn.multistage_search import atomic_write_json, canonical_json_sha256, file_sha256, read_json
from wnn.multistage_search_budget import initialize_budget
from wnn.splits import load_campaign_split
from wnn.training.sequence_config import load_config

DEFAULT_RUNTIME = "/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1"


def gpu_inventory(path: Path) -> dict[int, str]:
    """Read a ROOT-captured index/physical-UUID inventory with eight unique L4 slots."""
    rows = list(csv.reader(path.read_text().splitlines()))
    result = {
        int(row[0].strip()): row[1].strip()
        for row in rows
        if len(row) >= 2 and row[0].strip().isdigit()
    }
    if (
        set(result) != set(range(8))
        or len(set(result.values())) != 8
        or not all(value.startswith("GPU-") for value in result.values())
    ):
        raise ValueError("GPU inventory must bind exactly eight unique physical UUIDs")
    return result


def coverage_receipt(path: Path) -> dict[str, object]:
    """Require the actual ROOT CPU mask audit, independently of model profiles."""
    value = read_json(path)
    expected = {
        "split_sha256": SPLIT_SHA,
        "fit_train": 10325,
        "fit_valid": 0,
        "evaluation_sequences": 1873,
        "selected_rows": 3_528_273,
    }
    if value.get("status") not in ("passed", "complete", "completed") or any(
        value.get(key) != item for key, item in expected.items()
    ):
        raise ValueError("data coverage receipt does not prove exact train/full-valid counts")
    return value


def build(args: argparse.Namespace) -> Path:
    """Create an immutable panel plus the existing shared attempt/deadline ledgers."""
    runtime = args.runtime.resolve()
    source = runtime / "source-r1"
    inventory = runtime / "manifests/l4-source-r1-manifest.json"
    files = source_files(inventory)
    if len(files) != 620:
        raise ValueError("staged source inventory must retain all620 frozen files")
    trainer_sha = verify_source(source, files)
    data_root = runtime / "project"
    split_path = args.split_manifest.resolve()
    if file_sha256(split_path) != SPLIT_SHA:
        raise ValueError("receiving split is not the frozen official chronological boundary")
    split = load_campaign_split(split_path, data_root)
    if (
        len(split.train_groups) != 10325
        or split.valid_groups
        or len(split.evaluation_groups) != 1873
        or split.evaluation_source != "valid"
    ):
        raise ValueError("primary split membership differs from10325train/0validfit/1873eval")
    coverage_path = runtime / "receipts/data-coverage-r1.json"
    coverage_receipt(coverage_path)
    clock_path = runtime / "receipts/first-profile-r1.json"
    first = float(read_json(clock_path)["first_profile_epoch"])
    if not math.isfinite(first) or first > time.time() + 30 or time.time() >= first + 82800:
        raise ValueError("actual first-profile clock is invalid or collection reserve has started")
    devices = gpu_inventory(args.gpu_inventory)
    runs = runtime / "runs"
    if not runs.is_dir() or not runs.resolve().is_relative_to(
        Path("/data/lc-epanov_s/tabdata-architecture-20261107")
    ):
        raise ValueError("cohort output must resolve into its fresh owned physical/data placement")
    control = runtime / "control/l4-cohort-r1"
    control.mkdir(parents=True, exist_ok=True)
    packet = source / "research/experiment_drafts/architecture-execution-20261107/l4"
    cells = []
    for index, (family, seed) in enumerate(
        (family, seed) for family in FAMILIES for seed in (101001, 101002)
    ):
        name = f"{family}_s{seed}"
        old = packet / "configs" / f"{name}.json"
        config = runtime / "configs/runtime-r3" / f"{name}.json"
        runtime_recipe(read_json(old), read_json(config))
        output = runs / name
        cells.append(
            {
                "name": name,
                "family": family,
                "seed": seed,
                "gpu_index": index,
                "gpu_uuid": devices[index],
                "old_config": str(old),
                "old_config_sha256": file_sha256(old),
                "config": str(config),
                "config_sha256": file_sha256(config),
                "normalized_config_sha256": canonical_json_sha256(load_config(config)),
                "profile_output": str(output / "profile"),
                "fit_output": str(output / "fit"),
                "profile_owned_receipt": str(runtime / "receipts" / f"owned-profile-{name}.json"),
                "fit_owned_receipt": str(runtime / "receipts" / f"owned-fit-{name}.json"),
                "approved_profile": str(runtime / "approvals" / f"{name}.json"),
                "oracle_receipt": str(runtime / "receipts" / f"mamba-oracle-{name}.json"),
            }
        )
    panel = {
        "schema": "wnn.compact-l4-panel.v1",
        "runtime": str(runtime),
        "control": str(control),
        "source": str(source),
        "python": "/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python",
        "source_inventory": str(inventory),
        "source_inventory_sha256": SOURCE_INVENTORY_SHA,
        "trainer_source_sha256": trainer_sha,
        "data_root": str(data_root),
        "split_manifest": str(split_path),
        "split_sha256": SPLIT_SHA,
        "fit_members": [["train", group] for group in split.train_groups],
        "evaluation_members": [["valid", group] for group in split.evaluation_groups],
        "coverage_receipt": {"path": str(coverage_path), "sha256": file_sha256(coverage_path)},
        "first_profile_receipt": {"path": str(clock_path), "sha256": file_sha256(clock_path)},
        "authority": {
            "first_profile_epoch": first,
            "deadline_epoch": first + 86400,
            "fit_deadline_epoch": first + 82800,
            "scientific_fit_cap": 8,
            "profile_cap": 12,
            "profile_updates": 20,
            "output_cap_bytes": 8_000_000_000,
            "output_free_floor_bytes": 4_000_000_000,
            "data2_free_floor_bytes": 3_000_000_000,
            "aggregate_worker_threads": 64,
            "threads_per_worker": 8,
            "worker_rss_cap_bytes": 384 * 1024**3,
            "page_cache_headroom_bytes": 64 * 1024**3,
        },
        "cells": cells,
        "receiver_files": {
            str(path.resolve()): file_sha256(path) for path in Path(__file__).parent.glob("*.py")
        },
    }
    initialize_budget(
        control,
        max_fit_attempts=8,
        max_profile_attempts=12,
        stage_fit_caps={"1": 8},
        prior_fit_attempts=[],
        prior_profile_attempts=args.prior_profile_attempts,
        first_profile_epoch=first,
        campaign_wall_seconds=86400,
        collection_reserve_seconds=3600,
    )
    path = control / "panel.json"
    atomic_write_json(path, panel, replace=False)
    print(json.dumps({"panel": str(path), "sha256": file_sha256(path)}))
    return path


def main() -> None:
    """Freeze the ROOT-supplied inventories, measured membership and prior attempts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path(DEFAULT_RUNTIME))
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--gpu-inventory", required=True, type=Path)
    parser.add_argument("--prior-profile-attempts", required=True, type=int)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
