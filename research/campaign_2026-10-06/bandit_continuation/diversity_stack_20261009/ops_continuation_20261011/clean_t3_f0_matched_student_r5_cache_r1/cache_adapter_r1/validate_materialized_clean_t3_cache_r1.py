"""Validate the built clean T3 cache with the frozen consumer and real inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

CONTRACT_SHA256 = "2b94f8454009af6a549e555456185d02bd2d2c6d8628def30dc3a3d27fae62c9"
BUILDER_SHA256 = "13b20f62a988df90f17da495b2cc7e4e219de0792291412165a10d2cd8b22380"
RELEASE_SHA256 = "707c38a9a19c1c87103393ac8dd17fb2cd61af6b9f0f84daafd56820861691a0"
INPUTS_SHA256 = "3e5c09731a1dbf83b34de06f34fdf76c54ec3ceb7ea53376d63d46c798441910"
S5_VALIDATION_SHA256 = "2a8b66682623e8e8f58f2fdd529790dbbbd7a308b3074514877dfe006cd9429e"
PARENT_CGROUP = Path(
    "/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/wnn.slice/wnn-campaign.slice"
)
PARENT_HIGH_BYTES = 18_000_000_000
VALIDATOR_RESERVE_BYTES = 650_000_000


def sha256_file(path: Path) -> str:
    """Hash a file incrementally, without reading a producer array into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read structured artifacts whose nested fields are defined by frozen producers.

    `Any` is necessary because the producer-owned JSON contracts have nested,
    versioned structures that this validator verifies at runtime.
    """
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    """Fail closed when a cache source or output binding differs."""
    if not condition:
        raise ValueError(message)


def resource_snapshot() -> dict[str, int]:
    """Read aggregate RAM and swap for the bounded consumer validation."""
    return {
        "memory_current_bytes": int((PARENT_CGROUP / "memory.current").read_text()),
        "memory_swap_current_bytes": int((PARENT_CGROUP / "memory.swap.current").read_text()),
    }


def sample_target_parity(
    input_receipt: dict[str, Any],
    metadata: dict[str, Any],
    cache_root: Path,
    source_split: str,
) -> dict[str, Any]:
    """Recompute eight actual joined sequence targets in the pinned float32 order."""
    output_name = f"main_homogeneous_{source_split}"
    output_entry = metadata["files"][output_name]
    output_groups = list(map(int, output_entry["row_groups"]))
    sample_groups = output_groups[:8]
    expected: np.ndarray | None = None
    seeds: list[int] = []
    for teacher in input_receipt["teacher_records"]:
        seed = int(teacher["provenance"]["seed"])
        source = teacher["predictions"][source_split]
        source_groups = list(map(int, source["row_groups"]))
        positions = {group: index for index, group in enumerate(source_groups)}
        indices = np.fromiter(
            (positions[group] for group in sample_groups), dtype=np.int64, count=len(sample_groups)
        )
        source_array = np.load(source["array"]["path"], mmap_mode="r", allow_pickle=False)
        block = np.asarray(source_array[indices], dtype=np.float32)
        if expected is None:
            expected = np.array(block, dtype=np.float32, copy=True)
        else:
            np.add(expected, block, out=expected)
        seeds.append(seed)

    require(expected is not None and seeds == [3, 4, 5], "target sample teacher order differs")
    np.divide(expected, np.float32(3.0), out=expected)
    np.multiply(expected, np.float32(0.8125), out=expected)
    actual_array = np.load(cache_root / output_entry["path"], mmap_mode="r", allow_pickle=False)
    require(
        bool(np.array_equal(actual_array[: len(sample_groups)], expected)),
        f"{source_split} sampled cache targets differ from the float32 teacher formula",
    )
    return {
        "sample_sequence_count": len(sample_groups),
        "sample_group_order": sample_groups,
        "teacher_seed_order": seeds,
        "exact_float32_formula_match": True,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    """Verify producer lineage, exact split projection, consumer acceptance, and bytes."""
    started = time.monotonic()
    root = args.project_root.resolve(strict=True)
    contract_path = args.contract.resolve(strict=True)
    release_path = args.release_receipt.resolve(strict=True)
    inputs_path = args.input_receipt.resolve(strict=True)
    s5_path = args.s5_validation.resolve(strict=True)
    materialization_path = args.materialization_receipt.resolve(strict=True)
    contract = read_json(contract_path)
    release = read_json(release_path)
    inputs = read_json(inputs_path)
    s5 = read_json(s5_path)
    materialization = read_json(materialization_path)

    before = resource_snapshot()
    require(
        before["memory_current_bytes"] + VALIDATOR_RESERVE_BYTES <= PARENT_HIGH_BYTES,
        "campaign parent plus cache-validation reserve exceeds MemoryHigh",
    )
    require(before["memory_swap_current_bytes"] == 0, "campaign swap is nonzero")
    require(sha256_file(contract_path) == CONTRACT_SHA256, "reviewed cache contract changed")
    require(sha256_file(release_path) == RELEASE_SHA256, "root materialization release changed")
    builder_path = contract_path.parent / "build_clean_t3_f0_homogeneous_cache_r1.py"
    require(sha256_file(builder_path) == BUILDER_SHA256, "reviewed adapter builder changed")
    require(sha256_file(inputs_path) == INPUTS_SHA256, "actual input validation receipt changed")
    require(sha256_file(s5_path) == S5_VALIDATION_SHA256, "s5 terminal validation changed")
    require(
        release.get("cache_materialization_authorized") is True
        and release.get("cache_adapter_contract_sha256") == CONTRACT_SHA256
        and release.get("builder_sha256") == BUILDER_SHA256,
        "root release does not bind this cache adapter",
    )
    require(
        inputs.get("status") == "PASS_ALL_TEACHER_INPUTS_READY", "actual source inspection failed"
    )
    require(
        all(
            item["predictions"][split]["array"].get("hash_recomputed") is True
            for item in inputs["teacher_records"]
            for split in ("train", "valid")
        ),
        "one or more producer arrays were not streamed and hashed",
    )
    require(s5.get("status") == "PASS_TERMINAL_ACTUAL_PRODUCER_VALIDATION", "s5 source gate failed")
    require(
        materialization.get("status") == "CACHE_MATERIALIZED"
        and materialization.get("cache_root") == str(root / contract["output"]["cache_root"]),
        "reviewed builder did not report the released output root",
    )

    lineage = contract["lineage"]
    source_root = root / lineage["source_snapshot_root"]
    consumer_path = source_root / "src/wnn/distillation_cache.py"
    require(
        sha256_file(consumer_path) == lineage["consumer_cache_validator_sha256"],
        "frozen consumer validator source changed",
    )
    sys.path.insert(0, str(source_root / "src"))
    splits = importlib.import_module("wnn.splits")
    cache_module = importlib.import_module("wnn.distillation_cache")
    split = splits.load_campaign_split(root / contract["lineage"]["split_manifest_path"], root)
    cache_root = root / contract["output"]["cache_root"]
    consumer_result = cache_module.validate_distillation_cache(
        cache_root, root, split, "homogeneous", require_families=False
    )
    metadata = consumer_result["metadata"]
    metadata_path = cache_root / "metadata.json"
    metadata_sha256 = sha256_file(metadata_path)
    require(
        consumer_result["metadata_sha256"] == metadata_sha256,
        "frozen consumer metadata hash differs from output file",
    )
    require(
        metadata.get("cache_profile") == "cleanfold_homogeneous_strong_t3_v1",
        "cache profile differs",
    )
    require(metadata.get("homogeneous_scale") == 0.8125, "homogeneous scale differs")
    require(
        metadata.get("target_contract", {}).get("formula")
        == "float32(0.8125 * float32_mean(three_epoch2_predictions))"
        and metadata["target_contract"].get("clipping") == "none during cache construction",
        "target scaling or clipping metadata differs",
    )

    source_records = {int(item["seed"]): item for item in metadata["sources"]}
    input_records = {int(item["provenance"]["seed"]): item for item in inputs["teacher_records"]}
    expected_teachers = {int(item["seed"]): item for item in contract["teachers"]}
    require(
        set(source_records) == {3, 4, 5}, "cache does not have exactly the three clean T3 teachers"
    )
    ancestry: list[dict[str, Any]] = []
    for seed in (3, 4, 5):
        record = source_records[seed]
        expected = expected_teachers[seed]
        actual_input = input_records[seed]
        require(
            record.get("run") == expected["run"]
            and record.get("epoch") == 2
            and record.get("checkpoint_sha256") == expected["checkpoint_sha256"]
            and record.get("provenance_sha256") == expected["provenance_sha256"],
            f"seed-{seed} checkpoint or provenance ancestry differs",
        )
        for source_split in ("train", "valid"):
            output_source = record["predictions"][source_split]
            inspected = actual_input["predictions"][source_split]
            require(
                output_source.get("prediction_sha256") == inspected["prediction_sha256"]
                and output_source.get("sidecar_sha256") == inspected["sidecar_sha256"],
                f"seed-{seed} {source_split} source array or sidecar binding differs",
            )
        ancestry.append(
            {
                "seed": seed,
                "run": record["run"],
                "epoch": record["epoch"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "provenance_sha256": record["provenance_sha256"],
                "train_prediction_sha256": record["predictions"]["train"]["prediction_sha256"],
                "valid_prediction_sha256": record["predictions"]["valid"]["prediction_sha256"],
            }
        )
    coefficients = metadata["source_coefficients"]
    require(
        len(coefficients) == 3
        and all(
            item.get("family") == "strong"
            and item.get("epoch") == 2
            and item.get("coefficient") == 0.1
            for item in coefficients
        ),
        "source coefficients are not the three declared epoch-2 strong teachers",
    )

    target_orders = metadata["files"]
    train_entry = target_orders["main_homogeneous_train"]
    valid_entry = target_orders["main_homogeneous_valid"]
    expected_train = inputs["target_indices"]["train"]["selected_fit_order_sha256"]
    expected_valid = inputs["target_indices"]["valid"]["selected_fit_order_sha256"]
    require(
        train_entry["row_groups_sha256"] == expected_train,
        "train target order differs from actual train index",
    )
    require(
        valid_entry["row_groups_sha256"] == expected_valid,
        "valid target order differs from actual valid index",
    )
    evaluation_groups = set(map(int, split.evaluation_groups))
    require(
        len(valid_entry["row_groups"]) == 1405
        and not set(map(int, valid_entry["row_groups"])) & evaluation_groups,
        "FIT_VALID output includes evaluation membership",
    )

    sample_checks = {
        source_split: sample_target_parity(inputs, metadata, cache_root, source_split)
        for source_split in ("train", "valid")
    }
    output_files = {}
    for name, expected_shape in (
        ("main_homogeneous_train", [10535, 20000, 2]),
        ("main_homogeneous_valid", [1405, 20000, 2]),
    ):
        entry = target_orders[name]
        output_files[name] = {
            "path": str(cache_root / entry["path"]),
            "shape": entry["shape"],
            "dtype": entry["dtype"],
            "bytes": entry["bytes"],
            "sha256": consumer_result["verified_files"][name],
            "finite_validation": "PASS_ALL_VALUES_BY_FROZEN_CONSUMER",
            "finite_value_count": int(np.prod(expected_shape)),
            "ordered_row_groups_sha256": entry["row_groups_sha256"],
        }
        require(
            entry["shape"] == expected_shape and entry["dtype"] == "float32",
            f"{name} NPY header differs",
        )

    after = resource_snapshot()
    require(
        after["memory_current_bytes"] + VALIDATOR_RESERVE_BYTES <= PARENT_HIGH_BYTES
        and after["memory_swap_current_bytes"] == 0,
        "campaign resource gate changed during consumer validation",
    )
    return {
        "schema": "wnn.clean-t3-f0-cache-materialization-validation.r1",
        "status": "PASS_FROZEN_CONSUMER_AND_ACTUAL_OUTPUT_VALIDATION",
        "validated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root_release_sha256": sha256_file(release_path),
        "adapter_contract_sha256": sha256_file(contract_path),
        "adapter_builder_sha256": sha256_file(builder_path),
        "s5_terminal_validation_sha256": sha256_file(s5_path),
        "actual_producer_input_validation_sha256": sha256_file(inputs_path),
        "cache_root": str(cache_root),
        "metadata_sha256": metadata_sha256,
        "frozen_consumer": {
            "path": str(consumer_path),
            "sha256": sha256_file(consumer_path),
            "variant": "homogeneous",
            "require_families": False,
            "validation_status": "PASS",
            "hashes_and_full_array_finite_scans": "PASS",
        },
        "source_qualified_membership": {
            "fold": 0,
            "split_manifest_sha256": split.sha256,
            "fit_train_source": "datasets/train.parquet",
            "fit_train_count": len(split.train_groups),
            "fit_train_order_sha256": train_entry["row_groups_sha256"],
            "fit_valid_source": "datasets/valid.parquet",
            "fit_valid_count": len(split.valid_groups),
            "fit_valid_order_sha256": valid_entry["row_groups_sha256"],
            "evaluation_valid_count": len(split.evaluation_groups),
            "fit_valid_evaluation_overlap": 0,
            "target_order_source": "train_targets.npz and valid_targets.npz row_group members only",
        },
        "teacher_ancestry": ancestry,
        "target_contract": {
            "formula": metadata["target_contract"]["formula"],
            "teacher_mean": metadata["target_contract"]["teacher_mean"],
            "homogeneous_scale": metadata["homogeneous_scale"],
            "clipping": metadata["target_contract"]["clipping"],
            "sampled_exact_float32_parity": sample_checks,
        },
        "output_files": output_files,
        "validator_resources": {
            "parent_before": before,
            "parent_after": after,
            "reserve_bytes": VALIDATOR_RESERVE_BYTES,
            "swap_bytes": 0,
            "elapsed_seconds": time.monotonic() - started,
        },
        "student_fit_or_profile_launched": False,
    }


def main() -> int:
    """Run the bounded actual-cache consumer validation and write its receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--release-receipt", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--s5-validation", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "cache_root": result["cache_root"],
                "metadata_sha256": result["metadata_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
