from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    data_root = args.data_root.resolve(strict=True)
    cache_root = args.cache_root.resolve(strict=True)
    sys.path.insert(0, str(source_root / "src"))
    from wnn.final_distillation_cache import (
        EXPECTED_SPLIT_SHA256,
        FINAL_CACHE_FORMAT,
        FINAL_CACHE_VERSION,
        PARENTS,
        SEQUENCE_ROWS,
        TARGET_MULTIPLIER,
        TEACHERS,
        TARGET_WIDTH,
        file_sha256,
        row_groups_sha256,
        source_qualified_groups_sha256,
        target_row_groups,
        verify_h390_source_snapshot,
    )

    source_snapshot = verify_h390_source_snapshot(source_root)
    metadata_path = cache_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("format") != FINAL_CACHE_FORMAT
        or metadata.get("version") != FINAL_CACHE_VERSION
        or metadata.get("mode") != "all_data_final"
        or metadata.get("heldout_claim") is not False
        or metadata.get("source_split_manifest_sha256") != EXPECTED_SPLIT_SHA256
    ):
        raise ValueError("H390 cache metadata contract mismatch")
    membership = metadata.get("membership", {})
    if membership.get("evaluation_row_groups") != []:
        raise ValueError("H390 final cache has evaluation rows")
    target_orders = {name: target_row_groups(data_root, name) for name in ("train", "valid")}
    file_records = metadata.get("files", {})
    expected_keys = {"main_homogeneous_train", "main_homogeneous_valid"}
    if set(file_records) != expected_keys:
        raise ValueError("H390 cache does not contain exactly the two homogeneous arrays")
    expected_entries = {"metadata.json", "main_homogeneous_train.npy", "main_homogeneous_valid.npy"}
    actual_entries = {path.name for path in cache_root.iterdir()}
    if actual_entries != expected_entries:
        raise ValueError(f"unexpected cache files: {sorted(actual_entries)}")

    array_receipts = {}
    for split, rows in target_orders.items():
        key = f"main_homogeneous_{split}"
        entry = file_records[key]
        path = cache_root / f"{key}.npy"
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (len(rows), SEQUENCE_ROWS, TARGET_WIDTH)
        if array.shape != expected_shape or array.dtype != np.float32:
            raise ValueError(f"unexpected NPY header for {key}: {array.shape} {array.dtype}")
        digest = file_sha256(path)
        if digest != entry.get("sha256") or path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"H390 cache content hash/size mismatch: {key}")
        if entry.get("row_groups") != rows or entry.get("row_groups_sha256") != row_groups_sha256(rows):
            raise ValueError(f"H390 cache row order mismatch: {key}")
        if membership.get(split, {}).get("source_qualified_sha256") != source_qualified_groups_sha256(f"datasets/{split}.parquet", rows):
            raise ValueError(f"H390 cache source-qualified membership mismatch: {split}")
        for start in range(0, len(rows), 64):
            if not np.isfinite(array[start : start + 64]).all():
                raise ValueError(f"nonfinite H390 cache values: {key}")
        array_receipts[split] = {
            "path": path.name,
            "bytes": path.stat().st_size,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": digest,
            "rows": len(rows),
            "row_groups_sha256": row_groups_sha256(rows),
        }
        del array

    teacher_records = metadata.get("sources", [])
    if [item.get("run") for item in teacher_records] != list(TEACHERS):
        raise ValueError("H390 cache teacher roster mismatch")
    for record in teacher_records:
        spec = TEACHERS[record["run"]]
        if (
            record.get("checkpoint_sha256") != spec["checkpoint_sha256"]
            or record.get("provenance_sha256") != spec["provenance_sha256"]
            or set(record.get("predictions", {})) != {"train", "valid"}
        ):
            raise ValueError(f"H390 teacher lineage mismatch: {record.get('run')}")
    parent_records = metadata.get("parents", [])
    for seed in (3, 4):
        spec = PARENTS[seed]
        parent = next((item for item in parent_records if item.get("run") == spec["run"]), None)
        if (
            parent is None
            or parent.get("checkpoint_sha256") != spec["checkpoint_sha256"]
            or parent.get("provenance_sha256") != spec["provenance_sha256"]
        ):
            raise ValueError(f"H390 parent lineage mismatch for seed {seed}")
    if metadata.get("target_contract", {}).get("coefficient_multiplier") != TARGET_MULTIPLIER:
        raise ValueError("H390 target multiplier mismatch")

    print(
        json.dumps(
            {
                "status": "cache_verify_pass",
                "cache_root": str(cache_root),
                "metadata_sha256": file_sha256(metadata_path),
                "metadata_bytes": metadata_path.stat().st_size,
                "source_manifest_sha256": source_snapshot["source_manifest_sha256"],
                "packet_manifest_sha256": source_snapshot["packet_manifest_sha256"],
                "portable_registry_sha256": metadata.get("portable_registry_sha256"),
                "mode": metadata["mode"],
                "heldout_claim": metadata["heldout_claim"],
                "evaluation_row_groups": membership["evaluation_row_groups"],
                "teachers": [
                    {"run": item["run"], "checkpoint_sha256": item["checkpoint_sha256"], "provenance_sha256": item["provenance_sha256"]}
                    for item in teacher_records
                ],
                "parents": [
                    {"run": item["run"], "checkpoint_sha256": item["checkpoint_sha256"], "provenance_sha256": item["provenance_sha256"]}
                    for item in parent_records
                ],
                "arrays": array_receipts,
                "estimated_bytes": metadata["estimated_bytes"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
