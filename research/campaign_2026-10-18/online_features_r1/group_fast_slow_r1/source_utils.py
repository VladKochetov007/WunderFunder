"""Pinned-source and resource-safe helpers for the innovation incrementality screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SCRATCH_ROOT = Path("/home/vlad/development/wunder/research_tmp").resolve()


def sha256_file(path: Path) -> str:
    """Hash a file incrementally, including large immutable sources."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write one small JSON receipt atomically without replacing prior output."""
    temporary = path.with_name(f".{path.name}.pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_recipe_sources(root: Path, recipe: dict[str, Any]) -> dict[str, str]:
    """Verify all Python and metric/data implementation files before use."""
    expected = recipe["source_hashes"]
    observed = {str(name): sha256_file(root / str(name)) for name in expected}
    if observed != expected:
        raise ValueError("experiment source hashes differ from the frozen recipe")
    return observed


def verify_train_footer(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    """Check file identity and hash the pinned Parquet footer before PyArrow opens it."""
    before = path.stat()
    if before.st_size != int(expected["bytes"]) or before.st_mtime_ns != int(expected["mtime_ns"]):
        raise ValueError("train Parquet size/mtime differs from the pinned source")

    footer_bytes = int(expected["footer_bytes"])
    digest = hashlib.sha256()
    with path.open("rb") as source:
        source.seek(-8, 2)
        trailer = source.read(8)
        if trailer[4:] != b"PAR1" or int.from_bytes(trailer[:4], "little") != footer_bytes:
            raise ValueError("train Parquet trailer differs from the pinned source")
        source.seek(-(footer_bytes + 8), 2)
        remaining = footer_bytes + 8
        while remaining:
            chunk = source.read(min(1 << 20, remaining))
            if not chunk:
                raise ValueError("train Parquet footer ended before the pinned range")
            digest.update(chunk)
            remaining -= len(chunk)

    after = path.stat()
    observed = {
        "bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "footer_bytes": footer_bytes,
        "footer_sha256": digest.hexdigest(),
    }
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise RuntimeError("train Parquet changed while its footer was checked")
    if observed != expected:
        raise ValueError("train Parquet footer hash differs from the pinned source")
    return observed


def load_members_and_folds(
    root: Path, recipe: dict[str, Any]
) -> tuple[list[dict[str, int]], dict[int, int], dict[str, str]]:
    """Load the exact 32-sequence sample and create four component folds."""
    selection = recipe["selection"]
    sample_source = selection["sample_source"]
    receipt_path = root / sample_source["path"]
    if sha256_file(receipt_path) != sample_source["sha256"]:
        raise ValueError("source feature-only receipt hash differs from the frozen recipe")
    sample_receipt = json.loads(receipt_path.read_text())
    members = [
        {key: int(item[key]) for key in ("component", "row_group", "seq_ix")}
        for item in sample_receipt["sample"]
    ]
    if len(members) != int(recipe["cv"]["component_count"]):
        raise ValueError("sample source has an unexpected component count")
    if len({item["component"] for item in members}) != len(members):
        raise ValueError("component sample contains duplicates")

    split_path = root / selection["split_path"]
    split_bytes = split_path.read_bytes()
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    if split_sha256 != selection["split_sha256"]:
        raise ValueError("source-qualified split hash differs from the frozen recipe")
    split = json.loads(split_bytes)
    fit_groups = set(map(int, split["fit"]["train"]))
    by_row_group = {int(record["row_group"]): record for record in split["sequences"]["train"]}
    for member in members:
        group = member["row_group"]
        record = by_row_group.get(group)
        if group not in fit_groups or record is None:
            raise ValueError(f"sample row group {group} is outside fold0.fit.train")
        if any(int(record[key]) != member[key] for key in ("component", "seq_ix")):
            raise ValueError(f"sample identity changed for row group {group}")

    ordered_components = np.asarray(sorted(item["component"] for item in members))
    rng = np.random.default_rng(int(recipe["cv"]["fold_seed"]))
    shuffled = rng.permutation(ordered_components)
    fold_count = int(recipe["cv"]["fold_count"])
    fold_by_component = {
        int(component): fold
        for fold, chunk in enumerate(np.array_split(shuffled, fold_count))
        for component in chunk
    }
    expected_size = int(recipe["cv"]["components_per_fold"])
    if any(
        sum(value == fold for value in fold_by_component.values()) != expected_size
        for fold in range(fold_count)
    ):
        raise ValueError("component folds do not match the frozen balanced design")

    return (
        members,
        fold_by_component,
        {
            "sample_receipt_sha256": str(sample_source["sha256"]),
            "split_sha256": split_sha256,
        },
    )


def require_physical_scratch(path: Path) -> Path:
    """Resolve an output path and keep it on the approved physical scratch volume."""
    resolved = path.resolve()
    if not resolved.is_relative_to(SCRATCH_ROOT):
        raise ValueError(f"outputs must remain under physical scratch: {SCRATCH_ROOT}")
    return resolved
