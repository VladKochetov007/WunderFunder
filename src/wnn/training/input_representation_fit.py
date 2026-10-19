from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from wnn.data import N_FEATURES, SEQUENCE_LENGTH, WARMUP
from wnn.training.input_representation import (
    FittedInputRepresentation,
    deserialize_input_representation,
    serialize_input_representation,
)
from wnn.training.input_representation_dictionary import (
    ARTIFACT_SCHEMA,
    COMPONENT_SEED,
    FIT_COMPONENTS,
    REPRESENTATION_RECIPE,
    STRIDE,
    fit_dictionary,
    raw_normalizer,
    representation_recipe_sha256,
)

if TYPE_CHECKING:
    from wnn.splits import CampaignSplit
    from wnn.training.environment import TrainingEnvironment


def select_fit_members(
    train_groups: np.ndarray, split: CampaignSplit, count: int = FIT_COMPONENTS
) -> list[dict[str, int | str]]:
    """Seed-select one lowest train row-group from each fit-only component."""
    groups = np.asarray(train_groups, dtype=np.int64)
    if groups.ndim != 1 or groups.size == 0 or len(set(groups.tolist())) != groups.size:
        raise ValueError("dictionary training groups must be unique and one-dimensional")
    split.assert_fit_groups(groups, ())
    allowed = set(map(int, groups.tolist()))
    by_component: dict[int, list[int]] = defaultdict(list)
    for source, group, component in split.components:
        if source == "train" and int(group) in allowed:
            by_component[int(component)].append(int(group))
    components = np.asarray(sorted(by_component), dtype=np.int64)
    if count < 1 or components.size < count:
        raise ValueError(f"dictionary fitting needs {count} fit-only train components")
    rng = np.random.default_rng(COMPONENT_SEED)
    chosen = np.sort(rng.choice(components, size=count, replace=False))
    return [
        {"source": "train", "group": min(by_component[int(component)]), "component": int(component)}
        for component in chosen
    ]


def _member_identity(members: list[dict[str, int | str]]) -> str:
    encoded = json.dumps(members, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def validate_fitted_representation(
    fitted: FittedInputRepresentation,
    split_sha256: str,
    members: list[dict[str, int | str]],
) -> None:
    """Fail closed if embedded transform metadata differs from the fixed fold recipe."""
    metadata = fitted.fit_metadata
    if (
        metadata.get("split_manifest_sha256") != split_sha256
        or metadata.get("fit_members") != members
        or metadata.get("fit_members_sha256") != _member_identity(members)
        or metadata.get("recipe") != REPRESENTATION_RECIPE
        or metadata.get("recipe_sha256") != representation_recipe_sha256()
    ):
        raise ValueError("fitted representation metadata differs from the fixed fold recipe")
    if any(
        value.dtype != torch.float32
        for value in (
            fitted.raw_mean,
            fitted.raw_scale,
            fitted.encoder_weight,
            fitted.decoder_weight,
        )
    ):
        raise ValueError("fitted representation tensors must use FP32 storage")


def _fit_rows(members: list[dict[str, int | str]], environment: TrainingEnvironment) -> np.ndarray:
    """Read complete fitting sequences, retaining only post-warmup stride-40 rows."""
    row_indices = np.arange(0, SEQUENCE_LENGTH, STRIDE, dtype=np.int64)
    row_indices = row_indices[row_indices >= WARMUP]
    expected_rows = len(members) * len(row_indices)
    matrix = np.empty((expected_rows, N_FEATURES), dtype=np.float32)
    source = environment.sequence_file(environment.root / "datasets" / "train.parquet")
    offset = 0
    for member in members:
        sequence = source.read(int(member["group"]), targets=False, mask=False)
        features = np.asarray(sequence.features)
        if features.shape != (SEQUENCE_LENGTH, N_FEATURES) or features.dtype != np.float32:
            raise ValueError("dictionary source sequence does not match canonical F0 schema")
        selected = features[row_indices]
        if not np.isfinite(selected).all():
            raise ValueError("dictionary fitting inputs contain nonfinite raw features")
        matrix[offset : offset + len(selected)] = selected
        offset += len(selected)
    if offset != expected_rows:
        raise RuntimeError("dictionary fitting row count changed during full-sequence extraction")
    return matrix


def _artifact_payload(
    fitted: FittedInputRepresentation,
    split_sha256: str,
    members: list[dict[str, int | str]],
) -> dict[str, object]:
    """Bind dictionary tensors to exact fit membership and fixed recipe metadata."""
    return {
        "schema": ARTIFACT_SCHEMA,
        "split_manifest_sha256": split_sha256,
        "fit_members": members,
        "fit_members_sha256": _member_identity(members),
        "recipe": REPRESENTATION_RECIPE,
        "recipe_sha256": representation_recipe_sha256(),
        "representation": serialize_input_representation(fitted),
    }


def _load_artifact(
    path: Path,
    split_sha256: str,
    members: list[dict[str, int | str]],
    device: torch.device,
) -> FittedInputRepresentation:
    """Load only a matching same-fold fit artifact with safe tensor deserialization."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != ARTIFACT_SCHEMA
        or payload.get("split_manifest_sha256") != split_sha256
        or payload.get("fit_members") != members
        or payload.get("fit_members_sha256") != _member_identity(members)
        or payload.get("recipe") != REPRESENTATION_RECIPE
        or payload.get("recipe_sha256") != representation_recipe_sha256()
    ):
        raise ValueError("representation cache does not match current fold fitting membership")
    representation = deserialize_input_representation(payload.get("representation"), device)
    validate_fitted_representation(representation, split_sha256, members)
    return representation


def prepare_input_representation(
    train_groups: np.ndarray,
    split: CampaignSplit | None,
    environment: TrainingEnvironment,
    device: torch.device,
    *,
    cache_path: Path | None,
    output_path: Path,
) -> FittedInputRepresentation:
    """Load the shared fold dictionary or fit it from selected train groups only."""
    if split is None:
        raise ValueError("experimental representations require a source-qualified campaign split")
    members = select_fit_members(train_groups, split)
    if cache_path is not None:
        cache_path = cache_path.expanduser().resolve()
        if cache_path.is_file():
            return _load_artifact(cache_path, split.sha256, members, device)
        destination = cache_path
    else:
        destination = output_path / "input_representation.pt"
    raw = _fit_rows(members, environment)
    mean, scale = raw_normalizer(raw)
    mean_fp32 = mean.astype(np.float32)
    scale_fp32 = scale.astype(np.float32)
    normalized = (raw - mean_fp32) / scale_fp32
    encoder, decoder, training = fit_dictionary(normalized)
    metadata: dict[str, object] = {
        "fit_members": members,
        "fit_members_sha256": _member_identity(members),
        "split_manifest_sha256": split.sha256,
        "recipe": REPRESENTATION_RECIPE,
        "recipe_sha256": representation_recipe_sha256(),
        "component_selection_seed": COMPONENT_SEED,
        "component_selection": "uniform_without_replacement_sorted_fit_train_components; lowest_train_row_group_per_component",
        "fit_rows": "all train rows at canonical stride40 with row_index>=99; train source has no scored mask",
        "fit_row_count": int(len(raw)),
        "raw_normalizer": "float64_population_moments_fit_rows_only; std_floor=1e-6; accumulated in 16000-row blocks; stored_buffers_and_application_fp32",
        "raw_mean": mean.tolist(),
        "raw_scale": scale.tolist(),
        "dictionary_training": training,
    }
    fitted_cpu = FittedInputRepresentation(
        raw_mean=torch.from_numpy(mean_fp32),
        raw_scale=torch.from_numpy(scale_fp32),
        encoder_weight=encoder,
        decoder_weight=decoder,
        fit_metadata=metadata,
    )
    payload = _artifact_payload(fitted_cpu, split.sha256, members)
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
    return fitted_cpu.on_device(device)
