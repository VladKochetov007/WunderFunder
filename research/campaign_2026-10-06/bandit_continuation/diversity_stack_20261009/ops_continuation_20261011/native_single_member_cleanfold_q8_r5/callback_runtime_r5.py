"""Small R5-only compatibility helpers for the actual Py311 callback fixture."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

EXPECTED_STABLE_SCHEMA_SHA256 = "d67c226b887afa79b766439219836d3bb1e649b15c117210774e3625ef9b4283"
PARQUET_SCHEMA_HEADER = re.compile(r"^<pyarrow\._parquet\.ParquetSchema object at 0x[0-9a-fA-F]+>$")
STABLE_SCHEMA_HEADER = "<pyarrow._parquet.ParquetSchema object at <address>>"


def import_pinned_r4(path: Path) -> Any:
    """Import the exact R4 module after its source hash has been verified."""
    spec = importlib.util.spec_from_file_location("r4_single_control_consumer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned R4 consumer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def import_bound_native_package(contract: dict[str, Any], r4: Any) -> Any:
    """Load the exact ABI3 extension as a package with its bound package path."""
    assets = contract["runtime_assets"]
    for name, binding in assets.items():
        if "sha256" not in binding:
            continue
        path = Path(binding["absolute_path"]).resolve(strict=True)
        if r4.sha256_file(path) != binding["sha256"]:
            raise r4.GateError(f"R4 runtime asset changed: {name}")

    init_path = Path(assets["native_package_init"]["absolute_path"]).resolve(strict=True)
    package_dir = Path(assets["native_package"]["absolute_path"]).resolve(strict=True)
    if init_path.parent != package_dir or init_path.name != "__init__.py":
        raise r4.GateError("native package initializer is outside its bound package directory")
    if "wnn_native" in sys.modules:
        raise r4.GateError("an unbound wnn_native module is already imported")

    spec = importlib.util.spec_from_file_location(
        "wnn_native", init_path, submodule_search_locations=[str(package_dir)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import the bound native package: {init_path}")
    package = importlib.util.module_from_spec(spec)
    sys.modules["wnn_native"] = package
    spec.loader.exec_module(package)
    if Path(package.__file__).resolve(strict=True) != init_path:
        raise r4.GateError("loaded native package initializer differs from its binding")
    extension = importlib.import_module("wnn_native.wnn_native")
    if Path(extension.__file__).resolve(strict=True) != Path(
        assets["native_elf"]["absolute_path"]
    ).resolve(strict=True):
        raise r4.GateError("loaded native ELF differs from its binding")
    return package


def check_stable_metadata(
    parquet: Any,
    r3: dict[str, Any],
    preflight: dict[str, Any],
    feature_columns: list[str],
) -> dict[str, str]:
    """Check row/schema content while recording R3's process-address hash as diagnostic."""
    metadata = parquet.metadata
    expected = r3["valid_dataset"]
    if metadata.num_rows != expected["rows"] or parquet.num_row_groups != expected["row_groups"]:
        raise ValueError("live Parquet row/group counts differ from R3")
    if any(metadata.row_group(index).num_rows != 20_000 for index in range(parquet.num_row_groups)):
        raise ValueError("live Parquet contains an incomplete sequence group")

    names = set(parquet.schema.names)
    required = {
        "seq_ix",
        "step_in_seq",
        "need_prediction",
        "is_scored",
        "t0",
        "t1",
        *feature_columns,
    }
    if not required <= names:
        raise ValueError("live Parquet lacks required feature/evaluation columns")
    names_sha = hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()
    old = preflight["metadata_only_validation"]
    if len(names) != old["schema_names_count"] or names_sha != old["schema_names_sha256"]:
        raise ValueError("live Parquet column names differ from R3 preflight")

    schema_text = str(parquet.schema)
    header, separator, body = schema_text.partition("\n")
    if not separator or PARQUET_SCHEMA_HEADER.fullmatch(header) is None:
        raise ValueError("unexpected PyArrow ParquetSchema header")
    stable_sha = hashlib.sha256((STABLE_SCHEMA_HEADER + separator + body).encode()).hexdigest()
    if stable_sha != EXPECTED_STABLE_SCHEMA_SHA256:
        raise ValueError("stable Parquet schema content differs from the R5 binding")
    raw_sha = hashlib.sha256(schema_text.encode()).hexdigest()
    old_raw_sha = old["schema_sha256"]
    return {
        "legacy_repr_sha256_expected_r3": old_raw_sha,
        "legacy_repr_sha256_live": raw_sha,
        "legacy_repr_hash_status": "MATCH"
        if raw_sha == old_raw_sha
        else "MISMATCH_PROCESS_ADDRESS_IN_HEADER",
        "stable_schema_sha256": stable_sha,
        "schema_names_sha256": names_sha,
    }
