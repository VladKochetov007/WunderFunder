"""Validate final-fit W18 distillation caches without independent scoring semantics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from wnn.distillation_cache import (
    CACHE_FORMAT,
    CACHE_VERSION,
    FAMILY_ORDER,
    MAIN_VARIANTS,
    SEQUENCE_ROWS,
    file_sha256,
    row_groups_sha256,
)
from wnn.final_submit_manifest import FinalFitMembership
from wnn.training.final_submit_bridge import final_fit_identity
from wnn.training.final_submit_bridge_contract import validate_exact_final_w18_teachers

MAX_FINAL_CACHE_METADATA_BYTES = 2 << 20


def _read_final_cache_metadata(path: Path) -> tuple[dict[str, object], bytes]:
    """Read bounded final-only metadata without changing the generic cache limit."""
    size = path.stat().st_size
    if size <= 0 or size > MAX_FINAL_CACHE_METADATA_BYTES:
        raise ValueError("final-fit distillation metadata exceeds its bounded read size")
    with path.open("rb") as stream:
        raw = stream.read(MAX_FINAL_CACHE_METADATA_BYTES + 1)
    if len(raw) != size or len(raw) > MAX_FINAL_CACHE_METADATA_BYTES:
        raise ValueError("final-fit distillation metadata changed while being read")
    metadata = json.loads(raw)
    if not isinstance(metadata, dict):
        raise ValueError("final-fit distillation metadata must be an object")
    return metadata, raw


def _validate_teacher_sources(
    metadata: Mapping[str, object], membership: FinalFitMembership
) -> None:
    """Require each weighted source record to identify its final-fit checkpoint and rows."""
    validate_exact_final_w18_teachers(metadata)
    coefficients, sources = metadata.get("source_coefficients"), metadata.get("sources")
    if not isinstance(coefficients, list) or not isinstance(sources, list) or len(sources) < 4:
        raise ValueError("final-fit distillation cache lacks teacher source records")
    if tuple(metadata.get("family_order", ())) != FAMILY_ORDER:
        raise ValueError("final-fit distillation family order differs")
    if {item.get("family") for item in coefficients if isinstance(item, dict)} != set(FAMILY_ORDER):
        raise ValueError("final-fit teacher mixture must cover the four declared families")
    coefficient_sum = math.fsum(float(item["coefficient"]) for item in coefficients)
    if (
        not math.isfinite(coefficient_sum)
        or abs(coefficient_sum - 0.8125) > 1e-7
        or abs(float(metadata.get("coefficient_sum", 0.0)) - coefficient_sum) > 1e-7
    ):
        raise ValueError("final-fit teacher coefficients must retain the frozen 0.8125 sum")
    source_records = {item.get("run"): item for item in sources if isinstance(item, dict)}
    declared_runs = {item.get("run") for item in coefficients if isinstance(item, dict)}
    if len(source_records) != len(sources) or set(source_records) != declared_runs:
        raise ValueError("final-fit teacher records do not match the declared coefficients")
    expected_groups = {
        "train": list(membership.train_groups),
        "valid": list(membership.valid_groups),
    }
    for source in source_records.values():
        if (
            source.get("final_fit_manifest_sha256") != membership.manifest_sha256
            or source.get("final_fit_members_sha256") != membership.fit_members_sha256
            or source.get("source_validation_sha256") != membership.source_validation_sha256
            or source.get("prediction_role") != "fitting-only teacher targets; no independent score"
            or source.get("independent_score_claim") != "forbidden"
            or source.get("fit_group_counts")
            != {name: len(groups) for name, groups in expected_groups.items()}
        ):
            raise ValueError("final-fit teacher source belongs to a different data lineage")
        predictions = source.get("predictions")
        if not isinstance(predictions, Mapping):
            raise ValueError("final-fit teacher source omits source-index exports")
        checkpoint_sha256 = source.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
            raise ValueError("final-fit teacher source checkpoint hash is malformed")
        for split_name, groups in expected_groups.items():
            record = predictions.get(split_name)
            if (
                not isinstance(record, Mapping)
                or record.get("checkpoint_sha256") != checkpoint_sha256
                or record.get("row_groups_sha256") != row_groups_sha256(groups)
            ):
                raise ValueError(f"final-fit teacher {split_name} export order differs")
        closure = source.get("ancestor_closure")
        if not isinstance(closure, list) or not closure:
            raise ValueError("final-fit teacher source omits its ancestor closure")
        if any(
            not isinstance(parent, Mapping)
            or parent.get("final_fit_manifest_sha256") != membership.manifest_sha256
            or parent.get("final_fit_members_sha256") != membership.fit_members_sha256
            or parent.get("source_validation_sha256") != membership.source_validation_sha256
            for parent in closure
        ):
            raise ValueError("final-fit teacher ancestor closure mixes data lineages")
    for item in coefficients:
        record = source_records[item["run"]]
        if (
            record.get("family") != item.get("family")
            or record.get("epoch") != item.get("epoch")
            or abs(float(record.get("coefficient", 0.0)) - float(item["coefficient"])) > 1e-7
        ):
            raise ValueError(f"final-fit teacher source record differs: {item.get('run')}")


def validate_final_distillation_cache(
    cache_root: Path,
    membership: FinalFitMembership,
    variant: str,
    *,
    require_families: bool,
) -> dict[str, object]:
    """Verify an all-fit W18 cache, source identities, fixed teachers and payload bytes."""
    if variant not in MAIN_VARIANTS:
        raise ValueError(f"unknown distillation target variant: {variant}")
    root = cache_root.resolve(strict=True)
    metadata_path = root / "metadata.json"
    metadata, raw = _read_final_cache_metadata(metadata_path)
    identity = final_fit_identity(membership)
    if (
        metadata.get("format") != CACHE_FORMAT
        or metadata.get("version") != CACHE_VERSION
        or metadata.get("final_fit") != identity
        or metadata.get("split_manifest_sha256") is not None
        or metadata.get("fit_groups") != identity["fit_row_groups"]
        or metadata.get("evaluation_valid_groups") != []
    ):
        raise ValueError("distillation cache is not bound to this no-evaluation final fit")
    _validate_teacher_sources(metadata, membership)
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise ValueError("final-fit distillation cache omits its file manifest")
    expected_groups = identity["fit_row_groups"]
    verified: dict[str, str] = {}
    for split_name in ("train", "valid"):
        groups = expected_groups[split_name]
        keys = [f"main_{variant}_{split_name}"]
        if require_families:
            keys.append(f"families_{split_name}")
        for key in keys:
            entry = files.get(key)
            if not isinstance(entry, dict):
                raise ValueError(f"final-fit cache manifest is missing {key}")
            relative = Path(str(entry.get("path", "")))
            if relative.is_absolute() or len(relative.parts) != 1:
                raise ValueError(f"final-fit cache filename escapes its root: {key}")
            path = (root / relative).resolve(strict=True)
            if path.parent != root or not path.is_file():
                raise ValueError(f"final-fit cache payload is unsafe: {key}")
            row_groups = entry.get("row_groups")
            if row_groups != groups or entry.get("row_groups_sha256") != row_groups_sha256(groups):
                raise ValueError(f"final-fit cache row order differs for {key}")
            width = 2 if key.startswith("main_") else 8
            shape = (len(groups), SEQUENCE_ROWS, width)
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if (
                values.shape != shape
                or values.dtype != np.float32
                or entry.get("shape") != list(shape)
                or entry.get("dtype") != "float32"
            ):
                raise ValueError(f"final-fit cache payload shape or dtype differs for {key}")
            digest = file_sha256(path)
            if digest != entry.get("sha256") or path.stat().st_size != entry.get("bytes"):
                raise ValueError(f"final-fit cache payload bytes differ for {key}")
            for start in range(0, len(values), 64):
                if not np.isfinite(values[start : start + 64]).all():
                    raise ValueError(f"final-fit cache contains nonfinite values for {key}")
            verified[key] = digest
    return {
        "metadata": metadata,
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "verified_files": verified,
    }
