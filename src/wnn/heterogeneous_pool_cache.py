from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from wnn.heterogeneous_pool_sources import (
    FORMAT,
    INPUT_FORMAT,
    SPLIT_SHA256,
    JsonObject,
    groups_hash,
    ordered_groups,
    read_object,
    validate_profile,
    validate_source,
)
from wnn.splits import (
    CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    CampaignSplit,
    file_sha256,
    load_campaign_split,
)

# Dynamic metadata is checked at its producer/consumer boundary before array use.
STUDENT_CONTRACT = {
    "arch": "gru",
    "cell": "gru",
    "enc": "linear",
    "hidden": 256,
    "layers": 2,
    "precision": "fp32",
    "train_source": "train",
    "train_seqs": 10325,
    "epochs": 4,
    "checkpoint_policy": "fixed",
    "checkpoint_epoch": 3,
    "batch": 32,
    "chunk": 1000,
    "optim": "adamw",
    "lr": 0.001,
    "loss": "wmse",
    "loss_norm": "chunk",
    "distill_alpha": 0.5,
    "distill_boot_h": 0,
    "distill_alpha_far": None,
    "distill_family_weight": 0,
    "distill_cache_variant": "source",
    "distill_cache_scale": 1,
    "ema_teacher": 1,
    "ema_decay": 0.999,
    "eval_ema": True,
    "since_weight": 32,
    "since_floor": 0.25,
    "aux_next": 0.3,
    "aux_struct": 0.3,
    "init": "",
    "distill": "",
    "raw_only": False,
}


def validate_pool_student_options(args: Any) -> None:
    """Reject scientific-recipe drift only for this newly declared pool format."""
    for name, expected in STUDENT_CONTRACT.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"heterogeneous-pool student requires {name}={expected!r}")


def _require_split(split: CampaignSplit) -> None:
    if (
        split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
        or split.sha256 != SPLIT_SHA256
        or len(split.train_groups) != 10325
        or split.valid_groups
        or len(split.evaluation_groups) != 1873
    ):
        raise ValueError("pool cache requires the frozen10325/0/1873 primary split")


def _fit_order(data_root: Path, split: CampaignSplit) -> list[int]:
    with np.load(data_root / "cache/train_targets.npz", allow_pickle=False) as targets:
        raw = np.asarray(targets["row_group"])
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError("training target index is not a one-dimensional integer array")
    all_rows = ordered_groups(raw.tolist())
    fitted = set(split.train_groups)
    rows = [group for group in all_rows if group in fitted]
    if set(rows) != fitted:
        raise ValueError("training target index omits fitting members")
    return rows


def mix_source_rows(
    arrays: list[np.ndarray],
    indexes: list[dict[int, int]],
    coefficients: list[float],
    groups: list[int],
) -> np.ndarray:
    """Join each teacher by actual row-group identity and sum raw FP32 predictions."""
    if not arrays or not (len(arrays) == len(indexes) == len(coefficients)):
        raise ValueError("pool mixing requires aligned source records")
    result = None
    for array, index, coefficient in zip(arrays, indexes, coefficients, strict=True):
        if any(group not in index for group in groups):
            raise ValueError("teacher prediction index omits a fitting group")
        values = np.asarray(array[[index[group] for group in groups]], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("teacher fitting predictions contain nonfinite values")
        weighted = np.float32(coefficient) * values
        if result is None:
            result = weighted
        elif result.shape != values.shape:
            raise ValueError("teacher fitting predictions have inconsistent shapes")
        else:
            result += weighted
    if result is None or not np.isfinite(result).all():
        raise ValueError("teacher pool produced nonfinite values")
    return result


def _entry(path: Path, groups: list[int], values: np.ndarray) -> JsonObject:
    return {
        "path": path.name,
        "shape": list(values.shape),
        "dtype": "float32",
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "row_groups": groups,
        "row_groups_sha256": groups_hash(groups),
    }


def build_pool_cache(
    inputs: Path,
    data_root: Path,
    split_manifest: Path,
    output: Path,
    *,
    chunk_sequences: int = 8,
    min_free_bytes: int = 12 * 1024**3,
) -> Path:
    """Build one create-only fitting cache after verifying original source evidence."""
    if chunk_sequences < 1 or min_free_bytes < 0:
        raise ValueError("invalid pool-cache resource bound")
    manifest = read_object(inputs)
    if manifest.get("format") != INPUT_FORMAT:
        raise ValueError("unsupported pool-cache input manifest")
    split = load_campaign_split(split_manifest, data_root)
    _require_split(split)
    sources = manifest.get("sources")
    if not isinstance(sources, list) or any(not isinstance(item, dict) for item in sources):
        raise ValueError("pool-cache inputs lack explicit source records")
    profile = manifest.get("profile")
    validate_profile(profile, sources)
    qualified = [validate_source(record, split) for record in sources]
    groups = _fit_order(data_root, split)
    output = output.resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError("preserve existing pool-cache attempts; use a fresh output")
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = len(groups) * 20000 * 2 * 4 + 4096
    if shutil.disk_usage(output.parent).free < expected_bytes + min_free_bytes:
        raise OSError("insufficient disk for the pool cache and required headroom")
    partial.mkdir()
    train_path = partial / "main_source_train.npy"
    train = np.lib.format.open_memmap(
        train_path, mode="w+", dtype=np.float32, shape=(len(groups), 20000, 2)
    )
    arrays, indexes = [item[0] for item in qualified], [item[1] for item in qualified]
    for start in range(0, len(groups), chunk_sequences):
        selected = groups[start : start + chunk_sequences]
        train[start : start + len(selected)] = mix_source_rows(
            arrays, indexes, [item["coefficient"] for item in sources], selected
        )
    train.flush()
    files = {"main_source_train": _entry(train_path, groups, train)}
    del train
    valid_path = partial / "main_source_valid.npy"
    empty = np.empty((0, 20000, 2), dtype=np.float32)
    with valid_path.open("xb") as stream:
        np.save(stream, empty, allow_pickle=False)
    files["main_source_valid"] = _entry(valid_path, [], empty)
    metadata = {
        "format": FORMAT,
        "version": 1,
        "profile": profile,
        "fold": split.fold,
        "split_manifest_sha256": split.sha256,
        "fit_groups": {"train": list(split.train_groups), "valid": []},
        "evaluation_valid_groups": list(split.evaluation_groups),
        "source_coefficients": [
            {key: item[key] for key in ("run", "family", "epoch", "coefficient")}
            for item in sources
        ],
        "sources": sources,
        "coefficient_sum": 1.0,
        "family_order": list(dict.fromkeys(item["family"] for item in sources)),
        "homogeneous_family": "gru" if profile == "gru-only" else None,
        "epoch_convention": {
            "zero_based": True,
            "student_epochs": 4,
            "student_checkpoint_epoch": 3,
        },
        "student_contract": STUDENT_CONTRACT,
        "target_contract": "ordered raw FP32 weighted sum; coefficients total1; no cache-time clipping",
        "files": files,
        "inputs_sha256": file_sha256(inputs),
    }
    raw = json.dumps(metadata, indent=2, allow_nan=False).encode()
    if len(raw) > 512000:
        raise ValueError("pool-cache metadata exceeds the consumer's512k bound")
    (partial / "metadata.json").write_bytes(raw + b"\n")
    validate_pool_cache(partial, data_root, split, "source", require_families=False)
    if output.exists():
        raise FileExistsError("another writer claimed the pool-cache output")
    os.rename(partial, output)
    return output


def validate_pool_cache(
    cache_root: Path,
    data_root: Path,
    split: CampaignSplit,
    variant: str,
    *,
    require_families: bool,
) -> JsonObject:
    """Independently bind consumed cache bytes, original teachers and fitting row order."""
    _require_split(split)
    if variant != "source" or require_families:
        raise ValueError("teacher-pool cache supports source targets without family loss only")
    metadata_path = cache_root / "metadata.json"
    if metadata_path.stat().st_size > 512000:
        raise ValueError("pool-cache metadata exceeds512k")
    metadata = read_object(metadata_path)
    if (
        metadata.get("format") != FORMAT
        or metadata.get("version") != 1
        or metadata.get("split_manifest_sha256") != split.sha256
        or metadata.get("fold") != split.fold
        or metadata.get("fit_groups") != {"train": list(split.train_groups), "valid": []}
        or metadata.get("evaluation_valid_groups") != list(split.evaluation_groups)
        or metadata.get("student_contract") != STUDENT_CONTRACT
        or metadata.get("coefficient_sum") != 1.0
    ):
        raise ValueError("pool-cache metadata changes the frozen contract")
    sources = metadata.get("sources")
    if not isinstance(sources, list):
        raise ValueError("pool-cache source records are missing")
    validate_profile(metadata.get("profile"), sources)
    coefficients = [
        {key: item[key] for key in ("run", "family", "epoch", "coefficient")} for item in sources
    ]
    if metadata.get("source_coefficients") != coefficients:
        raise ValueError("pool-cache coefficient summary disagrees with source records")
    for record in sources:
        validate_source(record, split)
    verified = {}
    for split_name, groups in (("train", _fit_order(data_root, split)), ("valid", [])):
        key = f"main_source_{split_name}"
        entry = metadata.get("files", {}).get(key, {})
        path = cache_root / f"{key}.npy"
        if path.is_symlink() or entry.get("path") != path.name:
            raise ValueError("pool-cache target path differs or is a symlink")
        values = np.load(path, mmap_mode="r" if groups else None, allow_pickle=False)
        expected_shape = (len(groups), 20000, 2)
        if (
            values.shape != expected_shape
            or values.dtype != np.float32
            or entry.get("row_groups") != groups
            or entry.get("row_groups_sha256") != groups_hash(groups)
            or entry.get("shape") != list(expected_shape)
            or entry.get("dtype") != "float32"
            or path.stat().st_size != entry.get("bytes")
            or file_sha256(path) != entry.get("sha256")
        ):
            raise ValueError("pool-cache payload identity/row alignment differs")
        for start in range(0, len(values), 64):
            if not np.isfinite(values[start : start + 64]).all():
                raise ValueError("pool-cache targets are nonfinite")
        verified[key] = entry["sha256"]
    return {
        "metadata": metadata,
        "metadata_sha256": file_sha256(metadata_path),
        "verified_files": verified,
    }
