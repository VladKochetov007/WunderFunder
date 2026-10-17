from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from wnn.data import N_FEATURES
from wnn.numeric_ple import fit_training_quantile_knots
from wnn.resources import cpu_threads
from wnn.splits import file_sha256
from wnn.training.environment import TrainingEnvironment
from wnn.training.inputs import build_inputs


@torch.no_grad()
def fit_normalization(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    batches: int = 2,
    *,
    environment: TrainingEnvironment | None = None,
) -> None:
    """Fit normalization."""
    environment = environment if environment is not None else TrainingEnvironment()
    feats = []
    for b, batch in enumerate(loader):
        if b == 0:
            torch.set_num_threads(cpu_threads())
        if b >= batches:
            break
        f = build_inputs(batch[0].to(device), environment=environment)
        feats.append(f.reshape(-1, f.shape[-1])[::7])
    f = torch.cat(feats)
    model.mean.copy_(f.mean(0))
    model.std.copy_(f.std(0).clamp_min(0.001))


def iter_e540_fit_features(
    train_groups: list[int],
    valid_groups: list[int],
    *,
    environment: TrainingEnvironment | None = None,
) -> Iterable[np.ndarray]:
    """Read feature-only sequences in the same order as E540 fit membership."""
    environment = environment if environment is not None else TrainingEnvironment()
    for split, groups in (("train", train_groups), ("valid", valid_groups)):
        source = environment.sequence_file(environment.root / "datasets" / f"{split}.parquet")
        for group in groups:
            sequence = source.read(group, targets=False, mask=False)
            yield sequence.features


def fit_e540_numeric_preprocessing(
    model: torch.nn.Module,
    train_groups: list[int],
    valid_groups: list[int],
    split_manifest_sha256: str,
    *,
    environment: TrainingEnvironment | None = None,
) -> dict[str, object] | None:
    """Fit and bind PLE knots from exact feature-only fitting membership."""
    environment = environment if environment is not None else TrainingEnvironment()
    numeric_map = getattr(model, "numeric_feature_map", None)
    if numeric_map is None:
        return None
    members = [("train", group) for group in train_groups]
    members.extend(("valid", group) for group in valid_groups)
    result: dict[str, object] = {
        "mode": numeric_map.mode,
        "n_features": N_FEATURES,
        "base_input_width": numeric_map.input_width,
        "mapped_input_width": numeric_map.output_width,
        "n_intervals": numeric_map.n_intervals,
        "feature_order": "raw112 supplied channel order; no physical semantics inferred",
        "raw_bypass": "one unbounded standardized raw coordinate per channel",
        "event_features": "passed through after the expanded raw block",
        "split_manifest_sha256": split_manifest_sha256,
        "fit_members": {"train": train_groups, "valid": valid_groups},
        "fit_members_sha256": hashlib.sha256(
            json.dumps(sorted(members), separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "module_source_sha256": file_sha256(environment.source_root / "src/wnn/numeric_ple.py"),
    }
    if numeric_map.mode == "linear":
        result["control_definition"] = "same K+1 slots; only raw bypass coordinate is active"
        return result
    center = model.mean[:N_FEATURES].detach().cpu().numpy().astype(np.float32, copy=True)
    scale = model.std[:N_FEATURES].detach().cpu().numpy().astype(np.float32, copy=True)
    knots, counts, metadata = fit_training_quantile_knots(
        iter_e540_fit_features(train_groups, valid_groups, environment=environment),
        members,
        center,
        scale,
        n_intervals=numeric_map.n_intervals,
    )
    numeric_map.set_knots(knots, counts)
    normalization_hash = hashlib.sha256(
        center.astype("<f4", copy=False).tobytes() + scale.astype("<f4", copy=False).tobytes()
    ).hexdigest()
    result.update(metadata)
    result["training_raw_normalization_sha256"] = normalization_hash
    result["knot_counts"] = counts.tolist()
    result["knot_buffer"] = "model.numeric_feature_map.knots"
    return result
