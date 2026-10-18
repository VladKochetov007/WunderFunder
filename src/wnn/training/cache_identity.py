from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import numpy as np
import torch

from wnn.data import CORRUPT_TRAIN_ROW_GROUPS, SequenceFile
from wnn.privileged import validate_crossfit_cache
from wnn.propensity import (
    LEGACY_PROFILE,
    PROPENSITY_FIT_SOURCE_FILES,
    STRICT_FP32_PROFILE,
    ordered_groups_sha256,
)
from wnn.splits import CampaignSplit, file_sha256
from wnn.training.environment import TrainingEnvironment


def bind_fold_teachers(
    teachers: list[tuple[str, float]] | None,
    campaign_split: CampaignSplit,
    splits: Iterable[str],
    *,
    environment: TrainingEnvironment | None = None,
) -> list[dict[str, Any]]:
    """Distillation targets under a clean fold must come from teachers fitted on that same fold.

    Each cached prediction needs a sidecar naming the teacher run, split and the student's split-manifest hash,
    and its bytes must match the recorded prediction hash; the evaluation fold never reached the teacher's fit.
    """
    environment = environment if environment is not None else TrainingEnvironment()
    bound = []
    for name, scale in teachers:
        for split in splits:
            pred = environment.root / "cache" / f"pred_{name}_{split}.npy"
            side = json.loads(pred.with_suffix(".json").read_text())
            if side.get("run") != name or side.get("split") != split:
                raise ValueError(f"teacher sidecar does not describe {name}/{split}")
            if side.get("split_manifest_sha256") != campaign_split.sha256:
                raise ValueError(f"teacher {name} was not fitted on the student's clean fold")
            order = np.load(environment.root / "cache" / f"{split}_targets.npz")["row_group"]
            if list(map(int, side.get("row_groups", []))) != order.tolist():
                raise ValueError(f"teacher {name}/{split} rows are not in cache target order")
            digest = file_sha256(pred)
            if digest != side.get("prediction_sha256"):
                raise ValueError(f"teacher prediction bytes changed since export: {pred}")
            if "crossfit_format" in side:
                validate_crossfit_cache(side, campaign_split, environment.root, split)
            bound.append(
                {
                    "run": name,
                    "split": split,
                    "scale": scale,
                    "epoch": side.get("epoch"),
                    "checkpoint_sha256": side.get("checkpoint_sha256"),
                    "prediction_sha256": digest,
                }
            )
    return bound


def prop_cache_path(
    name: str,
    split: str,
    *,
    environment: TrainingEnvironment | None = None,
    campaign_split: CampaignSplit | None = None,
) -> Path:
    """Propensity cache written by scripts/cache_propensity.py; refuses a cache built from other weights."""
    if not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("propensity model name must be a single path component")
    environment = environment if environment is not None else TrainingEnvironment()
    path = environment.root / "cache" / f"propcache_{name}_{split}.npy"
    meta = path.with_suffix(".json")
    if not path.exists() or not meta.exists():
        raise FileNotFoundError(f"{path} missing: run scripts/cache_propensity.py {name}")
    model_path = environment.root / "artifacts" / "runs" / name / "model.pt"
    want = file_sha256(model_path)
    metadata_size = meta.stat().st_size
    if metadata_size <= 0 or metadata_size > 64000:
        raise ValueError("propensity cache sidecar exceeds its 64,000-byte read bound")
    metadata_raw = meta.read_bytes()
    if len(metadata_raw) != metadata_size:
        raise ValueError("propensity cache sidecar changed while being read")
    got = json.loads(metadata_raw)
    if not isinstance(got, dict):
        raise ValueError("propensity cache sidecar must be a JSON object")
    if got.get("model_sha256") != want:
        raise RuntimeError(f"{path}: propensity weights changed since the cache was built")
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    checkpoint_provenance = checkpoint.get("provenance", {})
    profile = checkpoint_provenance.get(
        "precision_profile", checkpoint.get("args", {}).get("precision_profile", LEGACY_PROFILE)
    )
    if profile == STRICT_FP32_PROFILE:
        _validate_strict_propensity_cache(
            path,
            got,
            name=name,
            split=split,
            model_sha256=want,
            checkpoint=checkpoint,
            checkpoint_provenance=checkpoint_provenance,
            campaign_split=campaign_split,
            environment=environment,
        )
    elif profile != LEGACY_PROFILE:
        raise ValueError(f"unsupported propensity checkpoint profile: {profile}")
    return path


def _validate_strict_propensity_cache(
    path: Path,
    metadata: dict[str, Any],
    *,
    name: str,
    split: str,
    model_sha256: str,
    checkpoint: dict[str, Any],
    checkpoint_provenance: dict[str, Any],
    campaign_split: CampaignSplit | None,
    environment: TrainingEnvironment,
) -> None:
    """Bind strict cache bytes, source, split, fit membership and output row order."""
    if campaign_split is None:
        raise ValueError("strict propensity caches require the matching campaign split")
    if split not in {"train", "valid"}:
        raise ValueError(f"unsupported propensity cache split: {split}")
    if metadata.get("model") != name or metadata.get("split") != split:
        raise ValueError("strict propensity cache names the wrong model or dataset split")
    if metadata.get("model_sha256") != model_sha256:
        raise ValueError("strict propensity cache checkpoint hash is invalid")
    if metadata.get("schema") != "wnn.propensity_cache.strict_fp32.v1":
        raise ValueError("strict propensity cache sidecar has the wrong schema")
    if (
        metadata.get("precision_profile") != STRICT_FP32_PROFILE
        or metadata.get("inference_precision") != "float32_no_autocast_tf32_disabled"
    ):
        raise ValueError("strict propensity cache precision binding is invalid")
    if metadata.get("split_manifest_sha256") != campaign_split.sha256:
        raise ValueError("strict propensity cache belongs to a different campaign split")
    if checkpoint_provenance.get("split_manifest_sha256") != campaign_split.sha256:
        raise ValueError("strict propensity checkpoint belongs to a different campaign split")
    fit_groups = checkpoint_provenance.get("fit_row_groups", {}).get("valid")
    expected_fit_groups = list(campaign_split.valid_groups)
    if checkpoint_provenance.get("fit_row_groups") != {"valid": expected_fit_groups}:
        raise ValueError("strict propensity fit membership differs from the campaign manifest")
    if fit_groups != expected_fit_groups:
        raise ValueError("strict propensity fit membership differs from the campaign manifest")
    evaluation_groups = checkpoint_provenance.get("evaluation_row_groups")
    if evaluation_groups != list(campaign_split.evaluation_groups):
        raise ValueError(
            "strict propensity evaluation membership differs from the campaign manifest"
        )
    fit_digest = ordered_groups_sha256(fit_groups)
    if checkpoint_provenance.get("fit_row_groups_sha256") != {"valid": fit_digest}:
        raise ValueError("strict propensity checkpoint fit-membership digest is invalid")
    if metadata.get("fit_row_groups_sha256") != fit_digest:
        raise ValueError("strict propensity cache fit-membership digest is invalid")
    if metadata.get("fit_row_groups_count") != len(fit_groups):
        raise ValueError("strict propensity cache fit-membership count is invalid")
    normalization = checkpoint_provenance.get("normalization")
    if not isinstance(normalization, dict) or normalization.get("method") != (
        "ordered_fit_population_welford_float64_v1"
    ):
        raise ValueError("strict propensity checkpoint lacks its declared fit normalization")
    if metadata.get("checkpoint_normalization") != normalization:
        raise ValueError("strict propensity cache normalization differs from the checkpoint")
    if normalization.get("count") != len(fit_groups) * 20_000:
        raise ValueError("strict propensity normalization count differs from its fit rows")
    state = checkpoint.get("model", {})
    for key in ("mean", "std"):
        tensor = state.get(key)
        if tensor is None or tensor.dtype != torch.float32:
            raise ValueError("strict propensity checkpoint lacks float32 normalization buffers")
        if not torch.isfinite(tensor).all().item() or (key == "std" and torch.any(tensor <= 0)):
            raise ValueError("strict propensity checkpoint normalization is invalid")
        digest = hashlib.sha256(
            tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
        ).hexdigest()
        if normalization.get(f"{key}_sha256") != digest:
            raise ValueError(f"strict propensity {key} buffer differs from its provenance")

    source_root = environment.source_root
    checkpoint_sources = checkpoint_provenance.get("source_sha256")
    if not isinstance(checkpoint_sources, dict) or set(checkpoint_sources) != set(
        PROPENSITY_FIT_SOURCE_FILES
    ):
        raise ValueError("strict propensity checkpoint source bindings are incomplete")
    if any(
        checkpoint_sources.get(relative) != file_sha256(source_root / relative)
        for relative in PROPENSITY_FIT_SOURCE_FILES
    ):
        raise ValueError("strict propensity training/feature sources changed")

    rows = len(SequenceFile(environment.root / "datasets" / f"{split}.parquet"))
    if metadata.get("rows") != rows or metadata.get("cache_shape") != [rows, 20000]:
        raise ValueError("strict propensity cache row count or shape is invalid")
    if metadata.get("cache_dtype") != "float32":
        raise ValueError("strict propensity cache dtype is invalid")
    if metadata.get("ordered_output_row_groups_sha256") != ordered_groups_sha256(range(rows)):
        raise ValueError("strict propensity cache row-group order is not canonical")
    filled = list(range(rows))
    if split == "train":
        filled = [group for group in filled if group not in CORRUPT_TRAIN_ROW_GROUPS]
    if metadata.get("populated_row_groups_sha256") != ordered_groups_sha256(filled):
        raise ValueError("strict propensity cache populated row groups are invalid")
    if metadata.get("n_filled") != len(filled):
        raise ValueError("strict propensity cache populated row-group count is invalid")

    cache = np.load(path, mmap_mode="r", allow_pickle=False)
    if cache.shape != (rows, 20000) or cache.dtype != np.float32:
        raise ValueError("strict propensity cache array shape/dtype differs from its contract")
    for start in range(0, rows, 64):
        chunk = np.asarray(cache[start : start + 64])
        if not np.isfinite(chunk).all() or np.any((chunk < 0) | (chunk > 1)):
            raise ValueError("strict propensity cache contains invalid probability values")
    del cache
    if metadata.get("cache_sha256") != file_sha256(path):
        raise ValueError("strict propensity cache bytes differ from the recorded digest")
    source_sha256 = metadata.get("source_sha256")
    if not isinstance(source_sha256, dict):
        raise ValueError("strict propensity cache source bindings are missing")
    cache_sources = (*PROPENSITY_FIT_SOURCE_FILES, "scripts/cache_propensity.py")
    if not isinstance(source_sha256, dict) or set(source_sha256) != set(cache_sources):
        raise ValueError("strict propensity cache source inventory is incomplete")
    if any(
        source_sha256.get(relative) != file_sha256(source_root / relative)
        for relative in cache_sources
    ):
        raise ValueError("strict propensity cache producer/feature source bindings changed")
