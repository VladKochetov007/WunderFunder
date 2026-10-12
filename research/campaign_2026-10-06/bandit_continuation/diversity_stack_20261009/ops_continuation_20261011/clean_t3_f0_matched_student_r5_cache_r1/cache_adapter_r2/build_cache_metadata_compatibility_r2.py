"""Add one missing frozen-consumer provenance field without rewriting cache arrays."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

ROOT = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
CONTRACT_PATH = Path(__file__).with_name("CACHE_METADATA_COMPATIBILITY_ADAPTER_R2.json")
TRAINER_SHA256 = "a2b5440ff4b8e5179db01cc047102ef634522327374fced31ca5fdf8afcd17a6"
EPOCH_CONVENTION = {
    "student_epochs": 6,
    "student_checkpoint_policy": "fixed",
    "student_checkpoint_epoch": 5,
}
PROVENANCE_FIELDS = {
    "source_coefficients",
    "coefficient_sum",
    "homogeneous_family",
    "family_order",
    "epoch_convention",
}
ARRAY_KEYS = ("main_homogeneous_train", "main_homogeneous_valid")


def json_object(value: object, label: str) -> dict[str, object]:
    """Require an object at a JSON contract boundary."""
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def read_json_object(path: Path, label: str) -> dict[str, object]:
    """Read and validate one small lineage JSON object."""
    return json_object(json.loads(path.read_text()), label)


def sha256_file(path: Path) -> str:
    """Hash a file with a bounded memory buffer."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def mapping(value: object, label: str) -> dict[str, object]:
    """Return a typed view of one nested lineage mapping."""
    return json_object(value, label)


def trainer_cache_metadata_keys(trainer_path: Path) -> list[str]:
    """Extract every frozen cache metadata key indexed in the provenance block."""
    tree = ast.parse(trainer_path.read_text(), filename=str(trainer_path))
    matches: list[ast.Dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "provenance"
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "distillation_cache"
            for target in node.targets
        ):
            matches.append(node.value)
    if len(matches) != 1:
        raise ValueError("frozen trainer must have one distillation provenance block")

    keys: list[str] = []
    for node in ast.walk(matches[0]):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "cache_metadata"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.append(node.slice.value)
    if len(keys) != len(set(keys)):
        raise ValueError("frozen trainer repeats a cache metadata provenance key")
    return sorted(keys)


def validate_actual_consumer(
    root: Path,
    cache_path: Path,
    source_root: Path,
    trainer_path: Path,
    split_path: Path,
    expected_split_sha256: str,
) -> dict[str, object]:
    """Run the frozen cache validator and provenance-key audit on actual artifacts."""
    sys.path.insert(0, str(source_root / "src"))
    from wnn.distillation_cache import validate_distillation_cache
    from wnn.splits import load_campaign_split

    split = load_campaign_split(split_path, root)
    if split.sha256 != expected_split_sha256:
        raise ValueError("actual consumer loaded a different frozen split manifest")
    validation = validate_distillation_cache(
        cache_path,
        root,
        split,
        "homogeneous",
        require_families=False,
    )
    metadata = mapping(validation["metadata"], "validated cache metadata")
    observed_fields = trainer_cache_metadata_keys(trainer_path)
    if set(observed_fields) != PROVENANCE_FIELDS:
        raise ValueError(f"frozen trainer cache provenance keys changed: {observed_fields!r}")
    missing = sorted(PROVENANCE_FIELDS - metadata.keys())
    if missing:
        raise ValueError(f"validated metadata is missing trainer provenance fields: {missing}")
    if metadata["epoch_convention"] != EPOCH_CONVENTION:
        raise ValueError("epoch convention differs from the frozen cleanfold producer")

    return {
        "validator_metadata_sha256": validation["metadata_sha256"],
        "verified_files": validation["verified_files"],
        "actual_trainer_path": str(trainer_path),
        "actual_trainer_sha256": sha256_file(trainer_path),
        "ast_extracted_cache_metadata_fields": observed_fields,
        "all_unconditional_fields_present": True,
        "epoch_convention_matches_frozen_producer": True,
    }


def materialize_compatibility_cache(root: Path, contract_path: Path) -> dict[str, object]:
    """Hard-link validated R1 arrays into a unique R2 cache with compatible metadata."""
    contract = read_json_object(contract_path, "adapter contract")
    if contract.get("schema") != "wnn.clean-t3-f0-cache-metadata-compatibility-adapter.r2":
        raise ValueError("unexpected compatibility adapter contract schema")
    script_entry = mapping(contract["adapter_script"], "adapter script contract")
    if script_entry.get("sha256") != sha256_file(Path(__file__)):
        raise ValueError("adapter source SHA256 disagrees with its contract")

    source_record = mapping(contract["source_cache"], "source cache contract")
    source_root = Path(str(source_record["root"])).resolve(strict=True)
    source_metadata_path = source_root / "metadata.json"
    if sha256_file(source_metadata_path) != source_record["metadata_sha256"]:
        raise ValueError("immutable R1 cache metadata changed")
    metadata = read_json_object(source_metadata_path, "source cache metadata")
    if "epoch_convention" in metadata:
        raise ValueError("R1 source unexpectedly already has the compatibility field")

    source_contract_path = Path(str(source_record["adapter_contract_path"]))
    if sha256_file(source_contract_path) != source_record["adapter_contract_sha256"]:
        raise ValueError("immutable R1 adapter contract changed")
    if metadata.get("adapter_contract") != str(source_contract_path):
        raise ValueError("R1 metadata points to an unexpected adapter contract")
    if metadata.get("adapter_contract_sha256") != source_record["adapter_contract_sha256"]:
        raise ValueError("R1 metadata adapter contract SHA256 differs")

    corrected_record = mapping(
        contract["corrected_consumer_metadata"], "corrected metadata contract"
    )
    producer_path = root / str(corrected_record["value_source_path"])
    if sha256_file(producer_path) != corrected_record["value_source_sha256"]:
        raise ValueError("frozen cleanfold metadata producer changed")
    epoch_convention = mapping(corrected_record["value"], "epoch convention")
    if epoch_convention != EPOCH_CONVENTION:
        raise ValueError("compatibility field differs from exact frozen producer object")

    output_record = mapping(contract["output_cache"], "output cache contract")
    output_root = Path(str(output_record["root"]))
    if not output_root.is_absolute():
        output_root = (root / output_root).resolve()
    output_partial = output_root.with_name(output_root.name + ".partial")
    if output_root.exists() or output_partial.exists():
        raise FileExistsError("unique R2 output or partial directory already exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    file_contract = mapping(source_record["files"], "source file contract")
    source_file_receipts: dict[str, dict[str, object]] = {}
    for key in ARRAY_KEYS:
        entry = mapping(file_contract[key], key)
        path = source_root / str(entry["path"])
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"R1 cache array missing or wrong size: {key}")
        if sha256_file(path) != entry["sha256"]:
            raise ValueError(f"R1 cache array SHA256 changed: {key}")
        source_file_receipts[key] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": str(entry["sha256"]),
            "device": path.stat().st_dev,
            "inode": path.stat().st_ino,
        }

    if output_root.parent.stat().st_dev != source_root.stat().st_dev:
        raise OSError("R1 and R2 cache roots are on different filesystems")
    output_partial.mkdir()
    output_file_receipts: dict[str, dict[str, object]] = {}
    for key in ARRAY_KEYS:
        entry = mapping(file_contract[key], key)
        source_path = source_root / str(entry["path"])
        output_path = output_partial / str(entry["path"])
        os.link(source_path, output_path)
        source_stat = source_path.stat()
        output_stat = output_path.stat()
        if (
            source_stat.st_dev != output_stat.st_dev
            or source_stat.st_ino != output_stat.st_ino
            or source_stat.st_size != output_stat.st_size
        ):
            raise OSError(f"R2 array is not an unchanged hard link: {key}")
        output_file_receipts[key] = {
            "path": str(output_path),
            "bytes": output_stat.st_size,
            "sha256": str(entry["sha256"]),
            "device": output_stat.st_dev,
            "inode": output_stat.st_ino,
            "same_inode_as_r1": True,
        }

    corrected_metadata = dict(metadata)
    corrected_metadata["epoch_convention"] = EPOCH_CONVENTION
    corrected_metadata["adapter_contract"] = str(contract_path.resolve())
    corrected_metadata["adapter_contract_sha256"] = sha256_file(contract_path)
    output_metadata_path = output_partial / "metadata.json"
    with output_metadata_path.open("x", encoding="utf-8") as output_metadata:
        output_metadata.write(json.dumps(corrected_metadata, indent=2) + "\n")
    metadata_sha256 = sha256_file(output_metadata_path)

    consumer_record = mapping(contract["frozen_consumer"], "frozen consumer contract")
    trainer_path = root / str(consumer_record["trainer_path"])
    if (
        consumer_record["trainer_sha256"] != TRAINER_SHA256
        or sha256_file(trainer_path) != TRAINER_SHA256
    ):
        raise ValueError("frozen student trainer source changed")
    source_binding = mapping(
        read_json_object(source_contract_path, "R1 adapter contract")["lineage"],
        "R1 lineage",
    )
    split_path = root / str(source_binding["split_manifest_path"])
    consumer_validation = validate_actual_consumer(
        root,
        output_partial,
        root / str(source_binding["source_snapshot_root"]),
        trainer_path,
        split_path,
        str(source_binding["split_manifest_sha256"]),
    )
    if consumer_validation["validator_metadata_sha256"] != metadata_sha256:
        raise ValueError("frozen validator returned unexpected R2 metadata hash")

    final_files = {
        key: {
            **value,
            "path": str(output_root / str(mapping(file_contract[key], key)["path"])),
        }
        for key, value in output_file_receipts.items()
    }
    final_metadata_path = output_root / "metadata.json"
    usage_before = shutil.disk_usage(output_root.parent)
    os.replace(output_partial, output_root)
    usage_after = shutil.disk_usage(output_root.parent)

    receipt = {
        "schema": "wnn.clean-t3-f0-cache-metadata-compatibility-validation.r2",
        "status": "PASS_ACTUAL_FROZEN_CACHE_AND_PROVENANCE_CONSUMER",
        "validated_utc": datetime.now(UTC).isoformat(),
        "adapter_contract_path": str(contract_path.resolve()),
        "adapter_contract_sha256": sha256_file(contract_path),
        "adapter_script_path": str(Path(__file__).resolve()),
        "adapter_script_sha256": sha256_file(Path(__file__)),
        "source_cache_root": str(source_root),
        "source_metadata_sha256": sha256_file(source_metadata_path),
        "source_arrays": source_file_receipts,
        "output_cache_root": str(output_root),
        "output_metadata_path": str(final_metadata_path),
        "output_metadata_sha256": sha256_file(final_metadata_path),
        "output_arrays": final_files,
        "array_recalculation": False,
        "array_copy": False,
        "same_filesystem_hardlinks": True,
        "added_metadata_field": {"epoch_convention": EPOCH_CONVENTION},
        "consumer_validation": consumer_validation,
        "logical_array_bytes": sum(int(value["bytes"]) for value in final_files.values()),
        "metadata_bytes": final_metadata_path.stat().st_size,
        "disk_free_before_bytes": usage_before.free,
        "disk_free_after_bytes": usage_after.free,
        "observed_disk_free_change_bytes": usage_after.free - usage_before.free,
        "profile_fit_or_grade": False,
        "student_fit_authorized": False,
        "student_profile_authorized": False,
    }
    receipt_path = Path(__file__).with_name("CACHE_METADATA_COMPATIBILITY_VALIDATION_R2.json")
    if receipt_path.exists():
        raise FileExistsError(f"compatibility validation receipt exists: {receipt_path}")
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    logging.info(
        "R2 cache metadata adapter validated: cache=%s metadata_sha256=%s",
        output_root,
        receipt["output_metadata_sha256"],
    )
    return receipt


def parse_args() -> argparse.Namespace:
    """Parse the single identified R2 materialization invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    return parser.parse_args()


def main() -> None:
    """Materialize the additive cache sidecar and validate its actual consumer."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    materialize_compatibility_cache(args.root.resolve(), args.contract.resolve(strict=True))


if __name__ == "__main__":
    main()
