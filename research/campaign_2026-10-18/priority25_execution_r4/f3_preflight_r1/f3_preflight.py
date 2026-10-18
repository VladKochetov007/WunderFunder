"""Validate the canonical F3 data and source through existing bounded verifiers."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from wnn.splits import CampaignSplit

BASE = Path("/home/grinchenkoeg/computer-vision-homework-20261018")
FOLD_SHA = "21c7fb52e8e69f38126362cbc5dc60dbc46539041d649b9206ec35e0f21ada1f"
VENDOR_SHA = "3212ce79d0c9d37de41ba679f47929f4c7abbecff6d3682e8acc827051c52aa4"
SOURCE_HELPER_SHA = "5e89f0fafa4fd68948fe625d882a055fce0a4554f8d4567bdcb1fa09f24e2eba"
COUNTS = (10_539, 1_368, 505, 145)
OUTPUT = BASE / "receipts/F3_BELERIAND_PREFLIGHT_V1.json"


def checked_module(name: str, digest: str) -> ModuleType:
    """Import an exact small helper copied from a successful F1 preflight."""
    path = Path(__file__).resolve().with_name(f"{name}.py")
    if not 0 < path.stat().st_size < 100_000:
        raise ValueError("helper is outside the metadata byte bound")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"frozen helper SHA mismatch: {name}")
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != path:
        raise ValueError(f"wrong helper imported: {name}")
    return module


def membership_record(split: CampaignSplit) -> dict[str, object]:
    """Require the exact new fold's ordered membership and schedule counts."""
    if (split.fold, split.n_folds, split.sha256) != (3, 4, FOLD_SHA):
        raise ValueError("canonical loader returned a different fold")
    components = {(source, group): comp for source, group, comp in split.components}
    eval_components = {components[("valid", g)] for g in split.evaluation_groups}
    actual = (
        len(split.train_groups),
        len(split.valid_groups),
        len(split.evaluation_groups),
        len(eval_components),
    )
    if actual != COUNTS:
        raise ValueError(f"F3 counts differ: {actual}")
    return {
        "fold": 3,
        "n_folds": 4,
        "manifest_sha256": FOLD_SHA,
        "fit_train_groups": actual[0],
        "fit_valid_groups": actual[1],
        "evaluation_valid_groups": actual[2],
        "evaluation_components": actual[3],
        "component_disjointness": "passed by frozen load_campaign_split/assert_fit_groups",
        "native_source_identity_route": True,
        "portable_registry_env_used": False,
    }


def main() -> None:
    """Create a receipt only after all source, content and membership checks pass."""
    sys.dont_write_bytecode = True
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("preflight must run with CUDA hidden")
    os.environ.pop("WNN_PORTABLE_DATA_MANIFEST", None)
    # The immutable helpers retain their historical F1 names. Only their generic
    # hashing/source functions are reused; F3 membership is checked below.
    data = checked_module("frozen_f1_preflight", VENDOR_SHA)
    source = checked_module("frozen_source_runtime_preflight", SOURCE_HELPER_SHA)
    source_receipt = source.build_receipt(
        BASE, BASE / "runtime/attempts" / f"{source.ATTEMPT_ID}.json", OUTPUT
    )
    fold_path = BASE / "splits/fold3.json"
    registry_path = BASE / "support-source/terminal-v1/PORTABLE_SOURCES.json"
    archive_path = BASE / "support-source/terminal-v1/PORTABLE_ARCHIVES.json"
    fold, fold_evidence = data._read_small_json(fold_path, FOLD_SHA, "fold3")
    registry, registry_evidence = data._read_small_json(
        registry_path, data.REGISTRY_SHA256, "portable sources"
    )
    archive_evidence = data._validate_archive_manifest(archive_path)
    source_root = BASE / "source-a266b036"
    sys.path.insert(0, str(source_root / "src"))
    modules = {}
    for name, expected in (
        ("splits", data.SPLITS_MODULE_SHA256),
        ("portability", data.PORTABILITY_MODULE_SHA256),
    ):
        module = importlib.import_module(f"wnn.{name}")
        path = source_root / f"src/wnn/{name}.py"
        if Path(module.__file__).resolve() != path.resolve():
            raise ValueError(f"noncanonical imported module: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"module hash changed: {name}")
        modules[name] = module
    files = data._hash_registered_inputs(
        BASE / "data",
        registry_path,
        fold,
        registry,
        modules["portability"].verify_portable_source,
    )
    split = modules["splits"].load_campaign_split(fold_path, BASE / "data")
    membership = membership_record(split)
    for item in files.values():
        item["stat_after_loader"] = data._same_file_generation(
            Path(str(item["path"])), cast(dict[str, int], item["stat_after_hash"])
        )
    data._read_small_json(fold_path, FOLD_SHA, "fold3 after checks")
    data._read_small_json(registry_path, data.REGISTRY_SHA256, "registry after checks")
    schedule = data._schedule_evidence(membership)
    if (
        schedule["batches_per_epoch"],
        schedule["stage1_updates"],
        schedule["terminal_updates"],
        schedule["terminal_batches"],
    ) != (1488, 29760, 89280, 8928):
        raise ValueError("F3 static schedule differs")
    import torch

    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("unexpected CUDA initialization")
    receipt = {
        "schema": "priority25.f3_beleriand_preflight.v1",
        "status": "passed",
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "read-only source/data/membership/runtime validation; no fit or cell admission",
        "source_runtime": source_receipt,
        "fold_manifest": fold_evidence,
        "portable_registry": registry_evidence,
        "portable_archive_manifest": archive_evidence,
        "data_files": files,
        "membership": membership,
        "static_schedule": schedule,
        "torch_version": str(torch.__version__),
        "cuda_initialized": False,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "immutable_generic_helper_sha256": VENDOR_SHA,
        "immutable_source_helper_sha256": SOURCE_HELPER_SHA,
        "limitations": [
            "Nested source-runtime receipt retains historical F1 kind and verifies frozen F0 source inventory.",
            "The new F3 fold and all six data sources are independently checked by this invocation.",
            "Schedule numbers are expected, not observed optimizer counters.",
            "This preflight does not stage a plan, claim a cell, or authorize a fit.",
        ],
    }
    with OUTPUT.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
