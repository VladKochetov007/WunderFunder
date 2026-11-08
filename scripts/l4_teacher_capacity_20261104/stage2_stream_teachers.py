"""Checkpoint, panel, and prediction ancestry checks for stage-two target streaming."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.l4_teacher_capacity_20261104.stage2_stream_common import (
    TRAIN_SEQUENCES,
    read_json,
)
from wnn.distillation_cache import SEQUENCE_ROWS, file_sha256, row_groups_sha256
from wnn.splits import CampaignSplit
from wnn.training.sequence_cache_ensemble_lineage import validate_producer
from wnn.training.sequence_cache_precision import validate_precision_contract
from wnn.training.sequence_config import load_config
from wnn.training.sequence_features import sequence_input_schema, sequence_input_width
from wnn.training.sequence_normalization import (
    build_normalization_identity,
    moments_sha256,
    normalization_identity_sha256,
)
from wnn.training.sequence_provenance import canonical_sha256, validate_profile_receipt


def _panel_cell(panel_path: Path, run: str, project_root: Path) -> tuple[dict[str, Any], str]:
    """Resolve the exact TF config and approved profile receipt from its frozen panel."""
    panel = read_json(panel_path)
    cells = panel.get("cells")
    if not isinstance(cells, list):
        raise ValueError("Transformer teacher panel has no cell list")
    matches = [cell for cell in cells if isinstance(cell, dict) and cell.get("run_name") == run]
    if len(matches) != 1:
        raise ValueError(f"Transformer panel must identify exactly one run: {run}")
    cell = matches[0]
    if (
        Path(str(cell.get("fit_output", ""))).resolve()
        != (project_root / "artifacts" / "runs" / run).resolve()
    ):
        raise ValueError(f"Transformer panel fit output differs from actual run path: {run}")
    config_path = Path(str(cell.get("config_path", "")))
    if not config_path.is_file():
        raise ValueError(f"Transformer config file/hash differs from its panel: {run}")
    expected_file_hash = cell.get("config_sha256")
    if expected_file_hash is not None and file_sha256(config_path) != expected_file_hash:
        raise ValueError(f"Transformer config file bytes differ from its panel: {run}")
    profile_output = Path(str(cell.get("profile_output", "")))
    profile_path = profile_output / "profile_receipt.json"
    if not profile_path.is_file():
        raise FileNotFoundError(f"approved Transformer profile receipt is missing: {run}")
    return cell, file_sha256(panel_path)


def _validate_sequence_sources(
    payload: dict[str, Any], split: CampaignSplit, producer: dict[str, Any]
) -> None:
    """Verify event211 source membership, recursive teacher ancestry, and normalizer bytes."""
    data = payload.get("data", {})
    config = payload.get("config", {})
    model = config.get("model", {})
    mode = model.get("input_feature_mode")
    schema = sequence_input_schema(mode)
    width = sequence_input_width(mode)
    members = [("train", group) for group in split.train_groups]
    member_lists = [list(member) for member in members]
    split_document = read_json(split.path)
    sources = data.get("sources")
    if (
        producer.get("fit_members") != member_lists
        or data.get("fit_members") != member_lists
        or data.get("evaluation_members")
        != [[split.evaluation_source, group] for group in split.evaluation_groups]
        or data.get("split_sha256") != split.sha256
        or data.get("primary_split_sha256") != split.primary_split_sha256
        or data.get("input_schema") != schema
        or schema != "raw112-event211-v1"
        or not isinstance(sources, dict)
        or sources != split_document.get("sources")
        or data.get("teacher") is not None
        or data.get("initializer") is not None
    ):
        raise ValueError("Transformer data membership/source/initializer ancestry differs")
    validate_producer(
        producer,
        split.sha256,
        members,
        evaluation_members=[(split.evaluation_source, group) for group in split.evaluation_groups],
        primary_split_sha256=split.primary_split_sha256,
        expected_sources=sources,
    )
    normalization = data.get("normalization", {})
    scope = config.get("normalization_scope", "prefix")
    if scope == "all_fit":
        selected = members
    elif scope == "prefix":
        limit = config.get("normalization_sequences", 32)
        if type(limit) is not int or limit < 1:
            raise ValueError("Transformer normalization prefix recipe is invalid")
        selected = members[:limit]
    else:
        raise ValueError("Transformer normalization scope is unsupported")
    selected_sources = {f"datasets/{source}.parquet" for source, _ in selected}
    expected_fingerprints = {name: sources[name] for name in sorted(selected_sources)}
    if (
        normalization.get("scope") != scope
        or normalization.get("sequence_count") != len(selected)
        or normalization.get("source_fingerprints") != expected_fingerprints
        or normalization.get("input_schema") != schema
    ):
        raise ValueError("Transformer normalization sources/membership differ from its recipe")
    normalizer_identity = build_normalization_identity(
        normalization["source_fingerprints"],
        split.sha256,
        selected,
        scope,
        width,
        input_schema=schema,
    )
    if normalization.get("identity_sha256") != normalization_identity_sha256(normalizer_identity):
        raise ValueError("Transformer normalization identity does not bind fitting members")
    actual_normalization = payload.get("provenance", {}).get("normalization", {})
    mean = np.asarray(actual_normalization.get("mean"), dtype=np.float32)
    std = np.asarray(actual_normalization.get("std"), dtype=np.float32)
    if (
        mean.shape != (width,)
        or std.shape != (width,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
        or normalization.get("moments_sha256") != moments_sha256(mean, std)
    ):
        raise ValueError("Transformer checkpoint normalizer moments are invalid or unbound")
    identity = payload.get("provenance", {}).get("identity", {})
    runtime = identity.get("runtime", {})
    if (
        identity.get("data") != data
        or identity.get("data_sha256") != canonical_sha256(data)
        or identity.get("config_sha256") != canonical_sha256(config)
    ):
        raise ValueError("Transformer checkpoint identity does not bind full config/data")
    validate_precision_contract(config, payload["provenance"], runtime)


def verify_sequence_teacher(
    run: str,
    split: CampaignSplit,
    project_root: Path,
    export_root: Path,
    panel_path: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Validate one TF checkpoint against the actual profiled config and train export."""
    checkpoint_path = project_root / "artifacts" / "runs" / run / "model.pt"
    manifest_path = export_root / "manifest.json"
    if not checkpoint_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"sequence teacher checkpoint/export is incomplete: {run}")
    cell, panel_hash = _panel_cell(panel_path, run, project_root)
    checkpoint_sha = file_sha256(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    manifest = read_json(manifest_path)
    data = payload.get("data", {})
    config = payload.get("config", {})
    model = config.get("model", {})
    producer = manifest.get("producer", {})
    approved_config = load_config(Path(cell["config_path"]))
    config_file_hash = file_sha256(Path(cell["config_path"]))
    config_identity_hash = canonical_sha256(approved_config)
    provenance = payload.get("provenance", {})
    profile_ref = provenance.get("profile_receipt", {})
    profile_path = Path(str(cell["profile_output"])) / "profile_receipt.json"
    profile_sha = file_sha256(profile_path)
    identity = provenance.get("identity", {})
    expected_seed = int(cell["seed"])
    if (
        payload.get("checkpoint_schema") != "wnn.sequence-fit.v1"
        or payload.get("epoch") != 3
        or payload.get("completed_epochs") != 4
        or config != approved_config
        or cell.get("normalized_config_sha256") != config_identity_hash
        or config.get("seed") != expected_seed
        or model.get("architecture") != "transformer"
        or (model.get("hidden"), model.get("layers"), model.get("heads")) != (384, 8, 8)
        or model.get("context_rows") != 1024
        or model.get("hint_dim") != 0
        or config.get("precision_mode") != "fp32"
        or data.get("split_sha256") != split.sha256
        or data.get("initializer") is not None
        or profile_ref.get("path") != str(profile_path)
        or profile_ref.get("sha256") != profile_sha
        or identity.get("world_size") != 1
        or manifest.get("schema") != "sequence-teacher-cache-v1"
        or manifest.get("complete") is not True
        or manifest.get("membership") != "fit"
        or manifest.get("split_sha256") != split.sha256
        or manifest.get("sequence_length") != SEQUENCE_ROWS
        or manifest.get("hint_dim") != 0
        or manifest.get("input_schema") != data.get("input_schema")
        or producer.get("checkpoint_sha256") != checkpoint_sha
        or producer.get("split_sha256") != split.sha256
        or producer.get("fit_members") != data.get("fit_members")
        or producer.get("teacher") != data.get("teacher")
        or producer.get("config") != config
        or producer.get("provenance") != provenance
    ):
        raise ValueError(f"Transformer checkpoint/export differs from approved recipe: {run}")
    config_path = Path(str(cell["config_path"]))
    if cell.get("config_sha256") != config_file_hash:
        raise ValueError(f"Transformer approved config file hash changed: {run}")
    validate_profile_receipt(profile_path, identity)
    runtime = identity.get("runtime", {})
    devices = runtime.get("devices", [])
    if (
        len(devices) != 1
        or devices[0].get("uuid") != cell.get("gpu_uuid")
        or Path(str(profile_ref.get("path"))).resolve() != profile_path.resolve()
    ):
        raise ValueError(f"Transformer profile device/receipt differs from frozen panel: {run}")
    _validate_sequence_sources(payload, split, producer)
    entries = manifest.get("sequences")
    if not isinstance(entries, list) or len(entries) != TRAIN_SEQUENCES:
        raise ValueError(f"sequence teacher export must contain all train groups: {run}")
    by_group: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("split") != "train":
            raise ValueError(f"sequence teacher export contains a non-train member: {run}")
        group = entry.get("row_group")
        if type(group) is not int or group in by_group:
            raise ValueError(f"sequence teacher export has an invalid/duplicate group: {run}")
        by_group[group] = entry
    if list(by_group) != list(split.train_groups):
        raise ValueError(f"sequence teacher export order differs from official train: {run}")
    record = {
        "run": run,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_file": checkpoint_path.name,
        "epoch": 3,
        "architecture": model["architecture"],
        "config_path": str(config_path),
        "config_file_sha256": config_file_hash,
        "normalized_config_sha256": config_identity_hash,
        "config_identity_sha256": identity["config_sha256"],
        "profile_receipt_path": str(profile_path),
        "profile_receipt_sha256": profile_sha,
        "panel_sha256": panel_hash,
        "source_identity_sha256": identity["source_sha256"],
        "data_identity_sha256": identity["data_sha256"],
        "run_identity_sha256": canonical_sha256(identity),
        "provenance_sha256": canonical_sha256(provenance),
        "fit_groups": len(split.train_groups),
        "fit_row_groups_sha256": row_groups_sha256(list(split.train_groups)),
        "prediction_manifest_path": str(manifest_path),
        "prediction_manifest_sha256": file_sha256(manifest_path),
        "prediction_root": str(export_root.resolve()),
    }
    return record, by_group
