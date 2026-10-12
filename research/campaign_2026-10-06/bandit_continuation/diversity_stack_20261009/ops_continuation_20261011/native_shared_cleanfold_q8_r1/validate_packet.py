#!/usr/bin/env python3
"""Validate frozen producers, outer-fold membership, data footer, and Py311 ELF."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import sysconfig
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKET = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def footer_sha256(path: Path, footer_bytes: int) -> str:
    """Hash the serialized footer plus the producer's eight-byte trailer range."""
    remaining = footer_bytes + 8
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(-remaining, os.SEEK_END)
        while remaining:
            block = stream.read(min(remaining, 1 << 20))
            if not block:
                raise ValueError(f"truncated parquet footer: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def check_file(path: Path, expected_sha256: str) -> None:
    """Fail closed when a declared source or producer artifact differs."""
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {path}")


def validate_fold(fold: int, binding: dict[str, Any]) -> dict[str, Any]:
    """Check split, fit, evaluation, and producer provenance identities."""
    split_binding = binding["split_manifest"]
    split_path = Path(split_binding["path"])
    check_file(split_path, split_binding["sha256"])
    split = json.loads(split_path.read_text(encoding="utf-8"))
    evaluation = [int(item) for item in split["evaluation"]["valid"]]
    expected_evaluation = split_binding["evaluation_row_groups_sorted"]
    if sorted(evaluation) != expected_evaluation:
        raise ValueError(f"fold {fold}: frozen evaluation group order/membership changed")
    if (
        len(set(evaluation)) != len(evaluation)
        or len(evaluation) != split_binding["evaluation_valid_count"]
    ):
        raise ValueError(f"fold {fold}: evaluation membership is duplicated or incomplete")
    if set(evaluation) & set(split["fit"]["valid"]):
        raise ValueError(f"fold {fold}: fit-valid and evaluation membership overlap")
    if set(evaluation) != set(split["excluded"]["valid"]):
        raise ValueError(f"fold {fold}: excluded-valid and evaluation membership differ")
    train_fit = set(int(item) for item in split["fit"]["train"])
    valid_fit = set(int(item) for item in split["fit"]["valid"])
    train_components = {
        int(item["component"])
        for item in split["sequences"]["train"]
        if int(item["row_group"]) in train_fit
    }
    valid_sequences = split["sequences"]["valid"]
    valid_components = {
        int(item["component"]) for item in valid_sequences if int(item["row_group"]) in valid_fit
    }
    evaluation_components = {
        int(item["component"])
        for item in valid_sequences
        if int(item["row_group"]) in set(evaluation)
    }
    fit_components = train_components | valid_components
    if fit_components & evaluation_components:
        raise ValueError(f"fold {fold}: fit and evaluation components overlap")
    if len(fit_components) != split_binding["fit_component_count"]:
        raise ValueError(f"fold {fold}: fit component count differs")
    if len(evaluation_components) != split_binding["evaluation_component_count"]:
        raise ValueError(f"fold {fold}: evaluation component count differs")
    if split_binding["fit_evaluation_component_overlap_count"] != 0:
        raise ValueError(f"fold {fold}: contract records a component overlap")
    sequences = split["sequences"]["valid"]
    if len(sequences) != 1873:
        raise ValueError(f"fold {fold}: expected 1873 complete valid row groups")
    if any(int(row["row_group"]) != index for index, row in enumerate(sequences)):
        raise ValueError(f"fold {fold}: valid split map is not keyed by row-group order")

    model_results = {}
    for role, model in binding["models"].items():
        checkpoint = Path(model["checkpoint_path"])
        provenance_path = Path(model["provenance_path"])
        check_file(checkpoint, model["checkpoint_sha256"])
        check_file(provenance_path, model["provenance_sha256"])
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("split_manifest_sha256") != split_binding["sha256"]:
            raise ValueError(f"fold {fold} {role}: split-manifest binding differs")
        if set(provenance.get("evaluation_row_groups", [])) != set(evaluation):
            raise ValueError(f"fold {fold} {role}: producer evaluation set differs")
        if (
            provenance.get("checkpoint_sha256", {}).get(checkpoint.name)
            != model["checkpoint_sha256"]
        ):
            raise ValueError(f"fold {fold} {role}: checkpoint/provenance binding differs")
        fit = provenance["fit_row_groups"]
        if role == "student":
            if set(fit["train"]) != set(split["fit"]["train"]):
                raise ValueError(f"fold {fold}: student fit-train membership differs")
            if set(fit["valid"]) != set(split["fit"]["valid"]):
                raise ValueError(f"fold {fold}: student fit-valid membership differs")
        else:
            if fit["train"] or set(fit["valid"]) != set(split["fit"]["valid"]):
                raise ValueError(f"fold {fold}: HFT must fit only on fold-fit valid groups")
            args = provenance.get("args", {})
            if args.get("train_source") != "valid" or args.get("mask_weight") is not True:
                raise ValueError(f"fold {fold}: HFT direct-fit recipe differs")
        model_results[role] = {
            "checkpoint_sha256": model["checkpoint_sha256"],
            "provenance_sha256": model["provenance_sha256"],
            "fit_source": model["fit_source"],
            "fit_membership_identity": provenance["fit_membership_identity"],
            "initial_checkpoint_sha256": provenance.get("initial_checkpoint_sha256"),
        }

    data = binding["valid_dataset"]
    valid_path = Path(data["path"])
    source = split["sources"]["datasets/valid.parquet"]
    if (
        valid_path.stat().st_size != data["bytes"]
        or int(source["bytes"]) != data["bytes"]
        or int(source["footer_bytes"]) != data["footer_bytes"]
        or source["footer_sha256"] != data["footer_sha256_including_8_byte_trailer"]
        or footer_sha256(valid_path, data["footer_bytes"])
        != data["footer_sha256_including_8_byte_trailer"]
    ):
        raise ValueError(f"fold {fold}: valid.parquet footer identity differs")

    return {
        "fold": fold,
        "split_manifest_sha256": split_binding["sha256"],
        "evaluation_valid_groups": len(evaluation),
        "evaluation_group_set_sha256": split_binding[
            "evaluation_row_groups_sorted_canonical_sha256"
        ],
        "fit_train_groups": len(split["fit"]["train"]),
        "fit_valid_groups": len(split["fit"]["valid"]),
        "excluded_train_groups": len(split["excluded"]["train"]),
        "valid_row_groups": len(sequences),
        "valid_parquet_bytes": valid_path.stat().st_size,
        "valid_footer_sha256_including_trailer": data["footer_sha256_including_8_byte_trailer"],
        "models": model_results,
    }


def validate_native(contract: dict[str, Any]) -> dict[str, Any]:
    """Load the exact CPython 3.11 extension and reject libpython linkage."""
    runtime = contract["runtime"]
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"expected CPython 3.11, got {sys.version}")
    native_dir = Path(runtime["native_package_dir"]).resolve(strict=True)
    sys.path.insert(0, str(native_dir))
    package_path = Path(importlib.import_module("wnn_native").__file__).resolve(strict=True)
    if package_path != Path(runtime["package_init_path"]).resolve(strict=True):
        raise ValueError(f"native package path mismatch: {package_path}")
    if sha256_file(package_path) != runtime["package_init_sha256"]:
        raise ValueError("native package initializer differs from its frozen SHA")
    module = importlib.import_module("wnn_native.wnn_native")
    actual_path = Path(module.__file__).resolve(strict=True)
    expected_path = Path(runtime["loaded_elf_path"]).resolve(strict=True)
    if actual_path != expected_path:
        raise ValueError(f"loaded native path mismatch: {actual_path}")
    loaded_sha = sha256_file(actual_path)
    if loaded_sha != runtime["loaded_elf_sha256"]:
        raise ValueError("loaded native ELF does not match the frozen Py311 SHA")
    output = subprocess.run(
        ["readelf", "-d", str(actual_path)], check=True, capture_output=True, text=True
    ).stdout
    if "libpython" in output.lower():
        raise ValueError("native ELF must not depend on libpython")
    return {
        "python": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "platform": platform.platform(),
        "package_init_path": str(package_path),
        "package_init_sha256": runtime["package_init_sha256"],
        "module_path": str(actual_path),
        "loaded_elf_sha256": loaded_sha,
        "libpython_dependency": False,
    }


def main() -> None:
    """Validate all exact producer, split, dataset-footer, and runtime bindings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=PACKET / "CONTRACT_R1.json")
    parser.add_argument("--output", type=Path, default=PACKET / "VALIDATE_PACKET_R1.json")
    parser.add_argument("--load-native", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite validation receipt: {args.output}")
    contract_sha = sha256_file(args.contract)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract.get("schema") != "wnn.native.shared-cleanfold-q8-quality-contract.r1":
        raise ValueError("unexpected contract schema")
    if contract.get("status") != "prepared_pending_root_execution_admission":
        raise ValueError("contract is not pending root admission")
    source_checks = {}
    for name, item in contract["source_bindings"].items():
        path = Path(item["path"])
        check_file(path, item["sha256"])
        source_checks[name] = item["sha256"]
    model_checks = {
        str(fold): validate_fold(int(fold), binding) for fold, binding in contract["folds"].items()
    }
    runtime = contract["runtime"]
    check_file(Path(runtime["wheel_path"]), runtime["wheel_sha256"])
    check_file(Path(runtime["loaded_elf_path"]), runtime["loaded_elf_sha256"])
    native_result = validate_native(contract) if args.load_native else None
    receipt = {
        "schema": "wnn.native.shared-cleanfold-q8-preflight.r1",
        "status": "PASS_PREFLIGHT_ONLY",
        "created_utc": datetime.now(UTC).isoformat(),
        "contract_path": str(args.contract.resolve()),
        "contract_sha256": contract_sha,
        "folds": model_checks,
        "source_sha256": source_checks,
        "native_loaded": native_result is not None,
        "native": native_result,
        "labels_loaded": False,
        "predictions_generated": False,
        "quality_evaluation_performed": False,
        "scope": "identity, split-membership, Parquet-footer and optional exact Py311 ELF checks only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "contract_sha256": contract_sha,
                "fold_counts": {
                    f: model_checks[f]["evaluation_valid_groups"] for f in model_checks
                },
                "native_loaded": receipt["native_loaded"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
