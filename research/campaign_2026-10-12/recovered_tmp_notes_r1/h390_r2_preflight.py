from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

REGISTRY_RELATIVE = Path(
    "research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/multifold_r1/portable_inputs_fold1.json"
)
REGISTRY_SHA256 = "551905f3363c65608e2d5a0f49e8eeca362f574ab6d98d443e31cee0790fb5fd"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve(strict=True)
    data_root = args.data_root.resolve(strict=True)
    sys.path.insert(0, str(source_root / "src"))
    from wnn import final_distillation_cache as cachemod
    from wnn.portability import verify_portable_source

    registry_path = data_root / REGISTRY_RELATIVE
    registry_sha = cachemod.file_sha256(registry_path)
    if registry_sha != REGISTRY_SHA256:
        raise ValueError("portable source registry differs from the root-approved hash")
    os.environ["WNN_PROJECT_ROOT"] = str(data_root)
    os.environ["WNN_PORTABLE_DATA_MANIFEST"] = str(registry_path)

    builder_path = source_root / "scripts" / "build_final_distill_cache.py"
    spec = importlib.util.spec_from_file_location("h390_r2_builder_preflight", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen H390 R2 builder")
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)

    source_snapshot = builder.verify_h390_source_snapshot(source_root)
    split_path = data_root / cachemod.TEACHER_SPLIT_MANIFEST_RELATIVE
    split_sha = builder.file_sha256(split_path)
    if split_sha != builder.EXPECTED_SPLIT_SHA256:
        raise ValueError("actual teacher split manifest does not match the H390 pin")
    split_manifest = json.loads(split_path.read_text())
    target_orders = {name: builder._target_index(data_root, name) for name in ("train", "valid")}
    population = cachemod.verify_final_population_sources(data_root, split_manifest)
    if population != target_orders:
        raise ValueError("verified raw-source membership differs from target index order")
    if len(target_orders["train"]) != 10_606 or len(target_orders["valid"]) != 1_873:
        raise ValueError("H390 all-data final membership is not the frozen 12,479 rows")

    portable_registry = json.loads(registry_path.read_text())
    raw_source_receipts = {}
    target_index_receipts = {}
    for split in ("train", "valid"):
        parquet_name = f"datasets/{split}.parquet"
        parquet_path = data_root / parquet_name
        original_identity = split_manifest["sources"][parquet_name]
        observed_sha = verify_portable_source(
            parquet_path,
            parquet_name,
            original_identity,
            registry_path,
        )
        registry_entry = portable_registry["sources"][parquet_name]
        raw_source_receipts[split] = {
            "path": parquet_name,
            "bytes": parquet_path.stat().st_size,
            "original_split_identity": original_identity,
            "registry_sha256": registry_entry["sha256"],
            "observed_sha256": observed_sha,
        }
        target_name = f"cache/{split}_targets.npz"
        target_path = data_root / target_name
        target_identity = split_manifest["sources"][target_name]
        digest = hashlib.sha256()
        with np.load(target_path, allow_pickle=False) as targets:
            for field in ("row_group", "seq_ix"):
                digest.update(np.asarray(targets[field], dtype="<i8").tobytes())
        observed_index_sha = digest.hexdigest()
        if (
            target_path.stat().st_size != target_identity["bytes"]
            or observed_index_sha != target_identity["index_sha256"]
        ):
            raise ValueError(f"target index differs from the exact frozen identity: {target_name}")
        target_index_receipts[split] = {
            "path": target_name,
            "bytes": target_path.stat().st_size,
            "expected_index_sha256": target_identity["index_sha256"],
            "observed_index_sha256": observed_index_sha,
            "row_groups_sha256": builder.row_groups_sha256(target_orders[split]),
            "rows": len(target_orders[split]),
        }

    output = data_root / "cache/h390_all_data_homogeneous_r2"
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite an existing H390 cache or partial output")
    disk = os.statvfs(output.parent)
    free_bytes = disk.f_bavail * disk.f_frsize
    output_bytes, reserve_bytes, admitted_bytes = builder.cache_disk_requirement(
        sum(map(len, target_orders.values()))
    )

    teacher_records = []
    for run in builder.TEACHERS:
        record, arrays = builder._teacher_contract(
            run,
            data_root / "artifacts/runs",
            data_root / "cache",
            target_orders,
            split_manifest,
        )
        if set(arrays) != {"train", "valid"}:
            raise ValueError(f"teacher {run} does not have both full-source exports")
        teacher_records.append(record)
        del arrays
        gc.collect()

    parent_records = [
        builder._parent_contract(
            builder.PARENTS[seed]["run"], data_root / "artifacts/runs", target_orders
        )
        for seed in (3, 4)
    ]
    if output.exists() or partial.exists():
        raise FileExistsError("cache appeared during read-only H390 preflight")

    print(
        json.dumps(
            {
                "status": "preflight_pass",
                "mode": "all_data_final",
                "heldout_claim": False,
                "evaluation_row_groups": [],
                "membership": {
                    split: {
                        "count": len(groups),
                        "row_groups_sha256": builder.row_groups_sha256(groups),
                    }
                    for split, groups in target_orders.items()
                },
                "total_rows": sum(map(len, target_orders.values())),
                "split_manifest_sha256": split_sha,
                "portable_registry_sha256": registry_sha,
                "raw_sources": raw_source_receipts,
                "target_indexes": target_index_receipts,
                "source_snapshot": source_snapshot,
                "teachers": teacher_records,
                "parents": parent_records,
                "cache_path_absent": True,
                "disk_free_bytes": free_bytes,
                "cache_output_bytes": output_bytes,
                "cache_safety_reserve_bytes": reserve_bytes,
                "cache_admitted_bytes": admitted_bytes,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
