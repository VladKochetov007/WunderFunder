from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wnn.training.sequence_config import load_config
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_monitor_progress import MonitorProgress
from wnn.training.sequence_objectives import (
    objective_numerators,
    scaled_objective,
    window_denominators,
)
from wnn.training.sequence_pretraining import (
    FutureInnovationRecords,
    prepare_future_innovation_records,
)
from wnn.training.sequence_pretraining_contract import (
    FUTURE_INNOVATION_HORIZONS,
    FUTURE_INNOVATION_KIND,
    FUTURE_INNOVATION_WIDTH,
)


def _pretraining_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "epochs": 2,
        "batch_size": 32,
        "accumulate_chunks": 1,
        "chunk_rows": 1000,
        "seed": 203201,
        "lr": 0.0003,
        "weight_decay": 0.01,
        "optimizer": "adamw",
        "grad_clip": 1.0,
        "warmup_fraction": 0.05,
        "schedule": "warmup_cosine_v2",
        "min_lr_ratio": 0.1,
        "kd_weight": 0.0,
        "hint_weight": 1.0,
        "training_mask": "all_after_warmup",
        "normalization_scope": "all_fit",
        "normalization_sequences": 10325,
        "required_world_size": 1,
        "required_device": "L4",
        "precision_mode": "fp32",
        "profile_reload": True,
        "checkpoint_retention": "all_epochs",
        "checkpoint_endpoint_policy": "configured_final_epoch",
        "pretraining": {"kind": FUTURE_INNOVATION_KIND, "horizons": [16, 64, 256]},
        "model": {
            "architecture": "gru",
            "hidden": 256,
            "layers": 2,
            "heads": 4,
            "context_rows": 512,
            "dropout": 0.0,
            "checkpoint_blocks": False,
            "hint_dim": FUTURE_INNOVATION_WIDTH,
            "hint_projection": "learned",
            "attention_backend": "math",
            "compile_blocks": False,
        },
    }


def _write_config(tmp_path: Path, config: dict[str, object]) -> Path:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_pretraining_recipe_is_opt_in_and_exact(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, _pretraining_config()))
    assert config["pretraining"] == {"kind": FUTURE_INNOVATION_KIND, "horizons": [16, 64, 256]}
    assert config["model"]["hint_dim"] == 336

    invalid = _pretraining_config() | {"epochs": 3}
    with pytest.raises(ValueError, match="future-input pretraining"):
        load_config(_write_config(tmp_path, invalid))


def test_allocated_ssl_configs_load_with_shared_paired_seeds() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "research/experiment_drafts/l4-input-dynamics-20261110/configs"
    )
    for seed in (203201, 203202):
        parent = load_config(root / f"ssl_pretrain_s{seed}.json")
        child = load_config(root / f"ssl_finetune_s{seed}.json")
        assert parent["seed"] == child["seed"] == seed
        assert parent["epochs"] == 2 and child["epochs"] == 4
        assert parent["pretraining"]["horizons"] == [16, 64, 256]
        assert child["parent_initializer"] == FUTURE_INNOVATION_KIND
        assert child.get("supervised_target") is None


def test_supervised_child_requires_declared_parent_stage(tmp_path: Path) -> None:
    config = {
        **_pretraining_config(),
        "epochs": 4,
        "hint_weight": 0.0,
        "objective_mode": "standalone_labels_wmse_v2",
        "parent_initializer": FUTURE_INNOVATION_KIND,
        "checkpoint_retention": "all_epochs",
        "checkpoint_endpoint_policy": "configured_final_epoch",
        "monitoring": {"train_eval_sequences": 32, "eval_batch_size": 4},
        "model": _pretraining_config()["model"] | {"hint_dim": 0},
    }
    config.pop("pretraining")
    loaded = load_config(_write_config(tmp_path, config))
    assert loaded["parent_initializer"] == FUTURE_INNOVATION_KIND
    assert loaded.get("supervised_target") is None

    config["epochs"] = 5
    with pytest.raises(ValueError, match="four-epoch shared"):
        load_config(_write_config(tmp_path, config))

    config["epochs"] = 4
    config["supervised_target"] = 0
    with pytest.raises(ValueError, match="four-epoch shared"):
        load_config(_write_config(tmp_path, config))


class _FeatureOnlyFile:
    def __init__(self, features: np.ndarray) -> None:
        self.features = features
        self.read_calls = 0

    def read(self, group: int, *, targets: bool, mask: bool) -> SimpleNamespace:
        assert group == 0
        assert targets is False and mask is False
        self.read_calls += 1
        return SimpleNamespace(features=self.features)


def _records(features: np.ndarray) -> object:
    from wnn.training.sequence_data import SequenceRecords

    record = object.__new__(SequenceRecords)
    record.members = [("train", 0)]
    record.sources = {
        "train": SimpleNamespace(_file=_FeatureOnlyFile(features), environment=None, path=None)
    }
    record.positions = {("train", 0): 0}
    record.hint_dim = 0
    record.cache = None
    record.hint_members = None
    record.training_mask = "all_after_warmup"
    return record


def test_pretext_moments_and_targets_are_train_only_and_label_free(tmp_path: Path) -> None:
    from wnn.data import N_FEATURES, SEQUENCE_LENGTH

    row = np.arange(SEQUENCE_LENGTH, dtype=np.float32)[:, None]
    feature = np.arange(N_FEATURES, dtype=np.float32)[None, :]
    features = np.sin(row * 0.002 + feature * 0.07).astype(np.float32)
    base = _records(features)
    data = SimpleNamespace(
        fit=base,
        provenance={
            "split_sha256": "a" * 64,
            "sources": {
                "datasets/train.parquet": {"sha256": "b" * 64},
                "datasets/valid.parquet": {"sha256": "c" * 64},
            },
        },
    )
    contract = prepare_future_innovation_records(data, tmp_path)
    assert isinstance(data.fit, FutureInnovationRecords)
    assert contract["source_fingerprints"] == {"datasets/train.parquet": {"sha256": "b" * 64}}
    assert contract["validation_members_used"] is False
    assert contract["label_values_read"] is False

    item = data.fit[0]
    assert item["x"].shape == (SEQUENCE_LENGTH, N_FEATURES)
    assert item["hints"].shape == (SEQUENCE_LENGTH, FUTURE_INNOVATION_WIDTH)
    assert not item["y"].any() and not item["teacher"].any()
    assert not item["mask"][:99].any()
    assert item["mask"][99 : SEQUENCE_LENGTH - 256].all()
    assert not item["mask"][SEQUENCE_LENGTH - 256 :].any()
    assert not item["hints"][SEQUENCE_LENGTH - 256 :].any()
    for horizon_index, _horizon in enumerate(FUTURE_INNOVATION_HORIZONS):
        target = item["hints"][
            99 : SEQUENCE_LENGTH - 256, horizon_index * 112 : (horizon_index + 1) * 112
        ]
        torch.testing.assert_close(target.mean(dim=0), torch.zeros(112), atol=2e-5, rtol=0)
        torch.testing.assert_close(
            target.std(dim=0, unbiased=False), torch.ones(112), atol=2e-5, rtol=0
        )
    assert base.sources["train"]._file.read_calls == 2


def test_future_suffix_cannot_change_pretext_prediction_prefix() -> None:
    model = build_sequence_model(
        {
            "architecture": "gru",
            "hidden": 8,
            "layers": 1,
            "heads": 1,
            "context_rows": 512,
            "dropout": 0.0,
            "checkpoint_blocks": False,
            "hint_dim": FUTURE_INNOVATION_WIDTH,
            "hint_projection": "learned",
            "precision_mode": "fp32",
            "attention_backend": "math",
            "compile_blocks": False,
        }
    ).eval()
    generator = torch.Generator().manual_seed(42)
    values = torch.randn(1, 100, 112, generator=generator)
    changed = values.clone()
    changed[:, 40:] = torch.randn(1, 60, 112, generator=generator)
    prediction, hint, _ = model(values)
    changed_prediction, changed_hint, _ = model(changed)
    torch.testing.assert_close(prediction[:, :40], changed_prediction[:, :40], rtol=0, atol=0)
    torch.testing.assert_close(hint[:, :40], changed_hint[:, :40], rtol=0, atol=0)


def test_pretext_loss_has_no_label_head_gradient() -> None:
    mask = torch.ones((1, 3), dtype=torch.bool)
    target = torch.zeros((1, 3, 2))
    teacher = torch.zeros_like(target)
    hints = torch.randn((1, 3, FUTURE_INNOVATION_WIDTH))
    prediction = torch.randn((1, 3, 2), requires_grad=True)
    hint_prediction = torch.randn_like(hints, requires_grad=True)
    denominators = window_denominators(target, mask, hints)
    numerators = objective_numerators(
        prediction,
        hint_prediction,
        target,
        teacher,
        hints,
        mask,
    )
    loss = scaled_objective(numerators, denominators, 1, 0.0, 1.0)
    loss.backward()
    assert denominators[0].item() == 0
    assert numerators.label.item() == numerators.distillation.item() == 0
    assert prediction.grad is not None and not prediction.grad.any()
    assert hint_prediction.grad is not None and hint_prediction.grad.abs().sum() > 0


def test_pretext_progress_is_training_only(tmp_path: Path) -> None:
    progress = MonitorProgress(tmp_path)
    progress.pretraining_epoch(1, 12, 0.375)
    state = json.loads((tmp_path / "monitor_progress.json").read_text(encoding="utf-8"))
    progress.close()
    assert state["phase"] == "pretraining_epoch_completed"
    assert state["pretext_task"] == FUTURE_INNOVATION_KIND
    assert state["pretext_loss"] == 0.375
    assert state["metric_scope"] == "training_only"
