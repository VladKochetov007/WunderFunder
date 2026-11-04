from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from wnn.training.sequence_cache_ensemble_provenance import canonical_sha256
from wnn.training.sequence_cache_weighted_pool_lineage import _validate_initializer
from wnn.training.sequence_initializer import load_fitting_initializer
from wnn.training.sequence_normalization import (
    build_normalization_identity,
    moments_sha256,
    normalization_identity_sha256,
)
from wnn.training.sequence_precision import precision_identity, runtime_precision_identity


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(112, dtype=torch.float32))
        self.register_buffer("std", torch.ones(112, dtype=torch.float32))
        self.linear = torch.nn.Linear(112, 1)


def _parent_checkpoint(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    model = _TinyModel()
    with torch.no_grad():
        model.linear.weight.fill_(0.25)
        model.linear.bias.fill_(-0.5)
    mean, scale = np.zeros(112, dtype=np.float32), np.ones(112, dtype=np.float32)
    source_fingerprints = {"datasets/train.parquet": {"sha256": "c" * 64}}
    normalization_data: dict[str, object] = {
        "scope": "prefix",
        "sequence_count": 1,
        "fit_prefix_sequences": 1,
        "source_fingerprints": source_fingerprints,
    }
    normalizer_identity = build_normalization_identity(
        source_fingerprints, "a" * 64, [("train", 0)], "prefix", 112
    )
    normalization_data["identity_sha256"] = normalization_identity_sha256(normalizer_identity)
    normalization_data["moments_sha256"] = moments_sha256(mean, scale)
    data: dict[str, object] = {
        "split_sha256": "a" * 64,
        "primary_split_sha256": "b" * 64,
        "fit_members": [["train", 0]],
        "evaluation_members": [["train", 1]],
        "evaluation_role": "control",
        "evaluation_source": "train",
        "evaluation_mask": "all_after_warmup",
        "training_mask": "all_after_warmup",
        "mask": "all_after_warmup AND row_index>=99",
        "input_schema": "raw112-v1",
        "normalization": normalization_data,
        "sources": {"datasets/train.parquet": {"sha256": "c" * 64}},
        "teacher": None,
        "initializer": None,
    }
    parent_config: dict[str, object] = {
        "epochs": 3,
        "precision_mode": "fp32",
        "lr": 0.001,
        "kd_weight": 0.0,
        "hint_weight": 0.0,
        "normalization_scope": "prefix",
        "normalization_sequences": 1,
        "model": {
            "architecture": "gru",
            "hidden": 2,
            "layers": 1,
            "hint_dim": 0,
            "precision_mode": "fp32",
            "attention_backend": "math",
            "compile_blocks": False,
        },
    }
    source_files = {"scripts/train_sequence.py": "d" * 64}
    runtime = runtime_precision_identity(
        parent_config,
        {
            "world_size": 1,
            "matmul_precision": "highest",
            "matmul_tf32": False,
            "cudnn_tf32": False,
            "autocast": False,
            "amp_scaler": False,
            "devices": [
                {
                    "rank": 0,
                    "local_rank": 0,
                    "name": "NVIDIA B300",
                    "capability": [10, 0],
                    "total_memory_bytes": 100,
                }
            ],
        },
    )
    identity = {
        "schema": "wnn.sequence-run-identity.v1",
        "config_sha256": canonical_sha256(parent_config),
        "data_sha256": canonical_sha256(data),
        "data": data,
        "source_sha256": canonical_sha256(source_files),
        "source_files": source_files,
        "runtime": runtime,
        "world_size": 1,
    }
    normalization: dict[str, object] = {"mean": mean.tolist(), "std": scale.tolist()}
    provenance = {
        "schema": "wnn.sequence-provenance.v1",
        "identity": identity,
        "normalization": normalization,
        "precision": precision_identity("fp32", "math"),
        "fixed_checkpoint_policy": "configured-final-epoch",
        "completed_epochs": 3,
        "global_updates": 3,
    }
    torch.save(
        {
            "checkpoint_schema": "wnn.sequence-fit.v1",
            "config": parent_config,
            "data": data,
            "model": model.state_dict(),
            "completed_epochs": 3,
            "epoch": 2,
            "provenance": provenance,
        },
        path,
    )
    return data, normalization


def test_initializer_allows_learning_rate_change_and_loads_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    data, normalization = _parent_checkpoint(checkpoint)
    child = _TinyModel()
    child_config = {
        "epochs": 4,
        "precision_mode": "fp32",
        "lr": 0.0002,
        "model": {
            "architecture": "gru",
            "hidden": 2,
            "layers": 1,
            "hint_dim": 0,
            "precision_mode": "fp32",
            "attention_backend": "math",
            "compile_blocks": False,
        },
    }

    ancestry = load_fitting_initializer(checkpoint, child, child_config, data, normalization)

    assert ancestry["schema"] == "wnn.sequence-initializer.v1"
    assert ancestry["parent_split_sha256"] == data["split_sha256"]
    assert ancestry["primary_split_sha256"] == data["primary_split_sha256"]
    torch.testing.assert_close(child.linear.weight, torch.full_like(child.linear.weight, 0.25))
    torch.testing.assert_close(child.linear.bias, torch.full_like(child.linear.bias, -0.5))


def test_initializer_rejects_different_auxiliary_membership(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    data, normalization = _parent_checkpoint(checkpoint)
    child = _TinyModel()
    child_config = {
        "epochs": 4,
        "precision_mode": "fp32",
        "lr": 0.0002,
        "model": {
            "architecture": "gru",
            "hidden": 2,
            "layers": 1,
            "hint_dim": 0,
            "precision_mode": "fp32",
            "attention_backend": "math",
            "compile_blocks": False,
        },
    }
    changed_data = data | {"fit_members": [["train", 2]]}

    with pytest.raises(ValueError, match="fit_members"):
        load_fitting_initializer(checkpoint, child, child_config, changed_data, normalization)


def test_initializer_rejects_reduced_precision_parent_runtime(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    data, normalization = _parent_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["provenance"]["identity"]["runtime"]["autocast"] = True
    torch.save(payload, checkpoint)
    child = _TinyModel()
    child_config = {
        "epochs": 4,
        "precision_mode": "fp32",
        "lr": 0.0002,
        "model": {
            "architecture": "gru",
            "hidden": 2,
            "layers": 1,
            "hint_dim": 0,
            "precision_mode": "fp32",
            "attention_backend": "math",
            "compile_blocks": False,
        },
    }

    with pytest.raises(ValueError, match="strict FP32"):
        load_fitting_initializer(checkpoint, child, child_config, data, normalization)


def test_initializer_receipt_carries_a_valid_direct_parent_for_pool_lineage(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    data, normalization = _parent_checkpoint(checkpoint)
    child_config = {
        "epochs": 4,
        "precision_mode": "fp32",
        "lr": 0.0002,
        "model": {
            "architecture": "gru",
            "hidden": 2,
            "layers": 1,
            "hint_dim": 0,
            "precision_mode": "fp32",
            "attention_backend": "math",
            "compile_blocks": False,
        },
    }
    ancestry = load_fitting_initializer(checkpoint, _TinyModel(), child_config, data, normalization)

    _validate_initializer(
        ancestry,
        data["split_sha256"],
        data["primary_split_sha256"],
        data["fit_members"],
        data["evaluation_members"],
        child_config["model"],
        data["sources"],
    )
    assert ancestry["parent_producer"]["provenance"]["precision"]["mode"] == "fp32"


@pytest.mark.parametrize(
    "tamper",
    ("training_mask", "kd_weight", "teacher", "initializer", "compiler", "legacy_precision"),
)
def test_initializer_rejects_unbound_or_non_label_parent_ancestry(
    tmp_path: Path, tamper: str
) -> None:
    checkpoint = tmp_path / "model.pt"
    data, normalization = _parent_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    identity = payload["provenance"]["identity"]
    if tamper == "training_mask":
        for record in (payload["data"], identity["data"]):
            record["training_mask"] = "is_scored"
            record["mask"] = "is_scored AND row_index>=99"
        identity["data_sha256"] = canonical_sha256(identity["data"])
    elif tamper == "kd_weight":
        payload["config"]["kd_weight"] = 0.25
    elif tamper == "teacher":
        payload["data"]["teacher"] = {"producer": {}}
        identity["data"] = payload["data"]
        identity["data_sha256"] = canonical_sha256(identity["data"])
    elif tamper == "initializer":
        payload["data"]["initializer"] = {"parent_identity_sha256": "e" * 64}
        identity["data"] = payload["data"]
        identity["data_sha256"] = canonical_sha256(identity["data"])
    elif tamper == "legacy_precision":
        payload["provenance"]["precision"] = "strict_fp32_no_autocast_no_tf32_math_sdpa"
        identity["runtime"].pop("precision")
        identity["runtime"].pop("compiler")
    else:
        identity["runtime"]["compiler"]["enabled"] = True
    torch.save(payload, checkpoint)
    child_config = {
        "epochs": 4,
        "precision_mode": "fp32",
        "lr": 0.0002,
        "model": {
            "architecture": "gru",
            "hidden": 2,
            "layers": 1,
            "hint_dim": 0,
            "precision_mode": "fp32",
            "attention_backend": "math",
            "compile_blocks": False,
        },
    }

    with pytest.raises(ValueError):
        load_fitting_initializer(checkpoint, _TinyModel(), child_config, data, normalization)
