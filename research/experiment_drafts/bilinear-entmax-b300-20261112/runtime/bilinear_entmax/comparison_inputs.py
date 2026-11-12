"""Admit source-pinned Bilinear Entmax score receipts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from bilinear_entmax.bindings import (
    PACKET,
    RUNTIME_COMMIT,
    RUNTIME_MANIFEST_SHA256,
    SEEDS,
    SPLIT_SHA256,
    VARIANTS,
    canonical,
    digest,
    load_cell,
    read_object,
)
from bilinear_entmax.fitting_admission import admit_full_shape_identity

SCORED_ROWS = 3_528_273
VALIDATION_SEQUENCES = 1_873
METRIC_ID = "global_wp_clip2_v1"


def _number(value: object, label: str) -> float:
    """Require a finite JSON number, excluding booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _current_source_files() -> dict[str, str]:
    """Join the pinned PR75 source manifest with current packet runtime hashes."""
    manifest_path = PACKET / "sources/runtime_source_files.json"
    if digest(manifest_path) != RUNTIME_MANIFEST_SHA256:
        raise ValueError("pinned canonical runtime file manifest has changed")
    manifest = read_object(manifest_path)
    canonical_files = manifest.get("files")
    if (
        manifest.get("runtime_commit") != RUNTIME_COMMIT
        or not isinstance(canonical_files, dict)
        or not canonical_files
        or manifest.get("source_sha256") != canonical(canonical_files)
    ):
        raise ValueError("pinned canonical runtime source manifest is malformed")
    files = dict(canonical_files)
    if any(
        not isinstance(name, str) or not isinstance(value, str) or len(value) != 64
        for name, value in files.items()
    ):
        raise ValueError("pinned runtime manifest contains an invalid source digest")
    for path in sorted((PACKET / "runtime").rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"packet source must be a regular file: {path}")
        files[f"bilinear_entmax/{path.relative_to(PACKET).as_posix()}"] = digest(path)
    return files


def _identity(producer: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate producer config, data and pinned source identities from one score."""
    config = producer.get("config")
    provenance = producer.get("provenance")
    if not isinstance(config, dict) or not isinstance(provenance, dict):
        raise ValueError("score producer lacks its canonical config or provenance")
    identity = provenance.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("score producer provenance lacks the fit identity")
    config_sha256 = identity.get("config_sha256")
    data = identity.get("data")
    if (
        config_sha256 != canonical(config)
        or not isinstance(data, dict)
        or identity.get("data_sha256") != canonical(data)
        or identity.get("source_sha256") != canonical(identity.get("source_files"))
    ):
        raise ValueError("score producer config, data, or source digest is inconsistent")
    expected_sources = _current_source_files()
    if identity.get("source_files") != expected_sources:
        raise ValueError("score source files differ from pinned PR75 and current packet sources")
    if identity.get("source_sha256") != canonical(expected_sources):
        raise ValueError("score source identity differs from the reviewed runtime map")
    if identity.get("world_size") != 2:
        raise ValueError("Bilinear Entmax fits require the pinned two-rank runtime")
    if config.get("precision_mode") != "bf16_bf16x9" or config.get("epochs") != 4:
        raise ValueError("score config differs from BF16x9 fixed E4")
    if (
        provenance.get("completed_epochs") != 4
        or provenance.get("fixed_checkpoint_policy") != "configured-final-epoch"
        or provenance.get("checkpoint_endpoint_policy") != "configured_final_epoch"
        or provenance.get("checkpoint_purpose") != "fixed_configured_final_endpoint"
    ):
        raise ValueError("producer provenance does not identify the fixed configured E4 endpoint")
    if identity.get("data", {}).get("split_sha256") != SPLIT_SHA256:
        raise ValueError("score fit ancestry differs from the official split pin")
    return config, identity


def _runtime_compatibility(runtime: object) -> dict[str, Any]:
    """Compare hardware/software while allowing distinct physical B300 device UUIDs."""
    if not isinstance(runtime, dict):
        raise ValueError("fit identity lacks runtime precision metadata")
    compatible = dict(runtime)
    devices = runtime.get("devices")
    if not isinstance(devices, list) or len(devices) != 2:
        raise ValueError("fit runtime must identify exactly two accelerator ranks")
    compatible_devices: list[dict[str, Any]] = []
    for rank, item in enumerate(devices):
        if not isinstance(item, dict) or item.get("rank") != rank:
            raise ValueError("fit runtime device records are missing or misordered")
        if "B300" not in str(item.get("name", "")):
            raise ValueError("fit runtime device is not a B300")
        compatible_devices.append({key: value for key, value in item.items() if key != "uuid"})
    compatible["devices"] = compatible_devices
    return compatible


def _admit_score(path: Path) -> dict[str, Any]:
    """Admit one complete canonical fixed-final score and bind it to one cell."""
    value = read_object(path)
    metric = value.get("score")
    producer = value.get("producer")
    if (
        value.get("split_sha256") != SPLIT_SHA256
        or value.get("sequences") != VALIDATION_SEQUENCES
        or not isinstance(metric, dict)
        or metric.get("metric_id") != METRIC_ID
        or metric.get("selected_rows") != SCORED_ROWS
        or not isinstance(producer, dict)
    ):
        raise ValueError(f"incomplete official-valid score: {path}")
    if producer.get("split_sha256") != SPLIT_SHA256:
        raise ValueError(f"producer split differs from score split: {path}")
    checkpoint_sha256 = producer.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
    ):
        raise ValueError(f"producer checkpoint digest is malformed: {path}")
    config, identity = _identity(producer)
    extension = config.get("bilinear_entmax")
    if not isinstance(extension, dict) or set(extension) != {
        "schema",
        "cell",
        "preprocessing",
        "preprocessing_sha256",
    }:
        raise ValueError(f"score lacks Bilinear Entmax model identity: {path}")
    cell = extension.get("cell")
    if not isinstance(cell, dict):
        raise ValueError(f"score cell identity is malformed: {path}")
    cell_id, variant, seed = cell.get("id"), cell.get("variant"), cell.get("seed")
    if variant not in VARIANTS or type(seed) is not int or seed not in SEEDS:
        raise ValueError(f"score cell is outside the frozen eight-cell panel: {path}")
    if cell_id != f"be_{variant}_s{seed}" or config.get("seed") != seed:
        raise ValueError(f"score cell identifier and recipe seed differ: {path}")
    declared_cell, _ = load_cell(PACKET / "configs" / f"{cell_id}.json")
    if cell != declared_cell or extension.get("schema") != "bilinear-entmax.model.v1":
        raise ValueError(f"score cell differs from the immutable packet descriptor: {path}")
    if identity.get("bilinear_entmax_cell") != cell:
        raise ValueError(f"score run identity does not bind its declared cell: {path}")
    admit_full_shape_identity(identity, producer["provenance"], cell, path)
    preprocessing_sha256 = extension.get("preprocessing_sha256")
    if (
        not isinstance(preprocessing_sha256, str)
        or len(preprocessing_sha256) != 64
        or any(character not in "0123456789abcdef" for character in preprocessing_sha256)
        or identity.get("data", {}).get("bilinear_entmax_preprocessing_sha256")
        != preprocessing_sha256
    ):
        raise ValueError(f"score preprocessing identity is inconsistent: {path}")
    fit_members = producer.get("fit_members")
    if (
        not isinstance(fit_members, list)
        or len(fit_members) != 10_325
        or any(
            not isinstance(member, list)
            or len(member) != 2
            or member[0] != "train"
            or type(member[1]) is not int
            for member in fit_members
        )
        or len({(member[0], member[1]) for member in fit_members}) != len(fit_members)
    ):
        raise ValueError(f"score producer fit roster is incomplete or duplicated: {path}")
    if identity.get("data", {}).get("fit_members") != fit_members:
        raise ValueError(f"score fit roster differs from its provenance identity: {path}")
    evaluation_members = identity.get("data", {}).get("evaluation_members")
    if (
        not isinstance(evaluation_members, list)
        or len(evaluation_members) != VALIDATION_SEQUENCES
        or any(
            not isinstance(member, list)
            or len(member) != 2
            or member[0] != "valid"
            or type(member[1]) is not int
            for member in evaluation_members
        )
        or len({(member[0], member[1]) for member in evaluation_members}) != VALIDATION_SEQUENCES
        or identity.get("data", {}).get("fit_members") != fit_members
    ):
        raise ValueError(f"score data identity omits full fit or validation membership: {path}")
    scores = {name: _number(metric.get(name), f"{path}:{name}") for name in ("t0", "t1", "wp")}
    if not math.isclose(scores["wp"], (scores["t0"] + scores["t1"]) / 2, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"pooled score does not equal the mean of target scores: {path}")
    return {
        "cell_id": cell_id,
        "variant": variant,
        "seed": seed,
        "score": scores,
        "checkpoint_sha256": checkpoint_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "config": config,
        "identity": identity,
        "score_sha256": digest(path),
        "runtime_compatibility": _runtime_compatibility(identity.get("runtime")),
    }
