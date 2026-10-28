from __future__ import annotations

import json
import random
import re
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.data import N_FEATURES, SEQUENCE_LENGTH
from wnn.splits import CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA, file_sha256, load_campaign_split
from wnn.training.sequence_epoch_prediction import digest, epoch_admission
from wnn.training.sequence_predict import load_endpoint
from wnn.training.sequence_records import SequenceRecords

# Heterogeneous checkpoint/config provenance is validated at this deserialization boundary.


def validate_precision_environment(environment: Mapping[str, str]) -> None:
    """Reject process overrides that can supersede the explicit strict-FP32 flags."""
    if environment.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") not in (None, "0"):
        raise ValueError("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE must be unset or 0")


def runtime_identity(expected_commit: str) -> dict[str, str]:
    """Reject an unreviewed or dirty receiver checkout before loading any model."""
    root = Path(__file__).resolve().parents[3]
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("expected-commit requires a complete Git SHA")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True
    ).strip()
    if commit != expected_commit or dirty:
        raise ValueError("receiver commit mismatch, modified sources or untracked files")
    return {"commit": commit, "root": str(root)}


def select_members(fitting: list[tuple[str, int]], count: int, seed: int) -> list[tuple[str, int]]:
    """Freeze a train-only seeded subset without inspecting inputs or labels."""
    if not fitting or any(source != "train" for source, _ in fitting):
        raise ValueError("calibration membership must contain only fitting train sequences")
    if len(set(fitting)) != len(fitting) or not 1 <= count <= len(fitting):
        raise ValueError("invalid calibration count or duplicate fitting membership")
    return sorted(random.Random(seed).sample(sorted(fitting), count))


def load_inputs(
    checkpoint: Path,
    checkpoint_sha256: str,
    index: Path,
    index_sha256: str,
    training_source_sha256: str,
    split_manifest: Path,
    data_root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], SequenceRecords, SequenceRecords, dict[str, Any]]:
    """Admit an immutable retained epoch against source/config/data and train-only split."""
    if checkpoint.is_symlink() or file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("checkpoint digest mismatch or symlink")
    admission = epoch_admission(checkpoint, index, index_sha256)
    identity = admission["index"]["identity"]
    if identity.get("source_sha256") != training_source_sha256:
        raise ValueError("training source does not match the explicitly pinned producer")
    if digest(identity.get("source_files")) != training_source_sha256:
        raise ValueError("training source file inventory is inconsistent")
    split = load_campaign_split(split_manifest, data_root)
    if split.schema != CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA or split.valid_groups:
        raise ValueError("BN diagnosis requires train-only/full-valid primary source split")
    model, payload = load_endpoint(checkpoint, device, admission)
    config, data = payload["config"], payload["data"]
    fitting = [("train", group) for group in split.train_groups]
    evaluation = [("valid", group) for group in split.evaluation_groups]
    manifest = json.loads(split_manifest.read_text())
    if (
        data.get("split_sha256") != split.sha256
        or data.get("fit_members") != [list(member) for member in fitting]
        or data.get("evaluation_members") != [list(member) for member in evaluation]
        or data.get("sources") != manifest["sources"]
        or digest(data) != identity.get("data_sha256")
        or data.get("teacher") is not None
        or config.get("kd_weight") != 0
        or config.get("hint_weight") != 0
    ):
        raise ValueError("checkpoint data/source/ancestry differs from train-only split")
    settings = config["model"]
    if (
        settings.get("architecture") != "galam"
        or settings.get("precision_mode", "fp32") != "fp32"
        or settings.get("input_feature_mode", "raw112") != "raw112"
        or settings["galam"].get("execution_backend", "step") != "step"
        or config.get("precision_mode", "fp32") != "fp32"
    ):
        raise ValueError("BN diagnostic requires the original FP32 raw112 step GALAM")
    fit = SequenceRecords(data_root, fitting, 0)
    valid = SequenceRecords(data_root, evaluation, 0)
    valid.validate_source_schema()
    binding = {
        "checkpoint_sha256": checkpoint_sha256,
        "epoch_index_sha256": index_sha256,
        "training_source_sha256": training_source_sha256,
        "config_sha256": digest(config),
        "split_sha256": split.sha256,
        "data_sha256": digest(data),
        "sources": data["sources"],
        "fit_members": data["fit_members"],
        "evaluation_members": data["evaluation_members"],
        "epoch_admission": admission,
    }
    return model, payload, fit, valid, binding


def feature_batches(
    dataset: SequenceRecords, members: list[tuple[str, int]], batch_size: int
) -> Iterator[torch.Tensor]:
    """Read calibration features only; never load targets or scored-row masks."""
    if batch_size < 1 or not set(members).issubset(dataset.members):
        raise ValueError("invalid calibration batch or membership")
    if any(source != "train" for source, _ in members):
        raise ValueError("validation inputs cannot enter BN calibration")
    source = dataset.sources["train"]
    if source._file is None:
        source._file = source.environment.sequence_file(source.path)
    for start in range(0, len(members), batch_size):
        rows = []
        for _, group in members[start : start + batch_size]:
            features = source._file.read(group, targets=False, mask=False).features
            if features.shape != (SEQUENCE_LENGTH, N_FEATURES) or not np.isfinite(features).all():
                raise ValueError("invalid calibration input sequence")
            rows.append(torch.from_numpy(np.array(features, dtype=np.float32, copy=True)))
        yield torch.stack(rows)
