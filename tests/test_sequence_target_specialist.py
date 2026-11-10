from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from wnn.training.sequence_config import load_config, validate_runtime_requirements
from wnn.training.sequence_distributed import DistributedContext
from wnn.training.sequence_ensemble_config import STANDALONE_LABELS_MODE, objective_contract
from wnn.training.sequence_fit_admission import validate_fit_reference_inputs
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_objectives import (
    objective_numerators,
    scaled_objective,
    window_denominators,
)
from wnn.training.sequence_provenance import run_identity, validate_profile_receipt


def _write_config(tmp_path: Path, **changes: object) -> Path:
    raw: dict[str, object] = {
        "epochs": 6,
        "batch_size": 32,
        "accumulate_chunks": 1,
        "chunk_rows": 1000,
        "seed": 203201,
        "lr": 0.0003,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "warmup_fraction": 0.05,
        "schedule": "warmup_cosine_v2",
        "min_lr_ratio": 0.0,
        "normalization_scope": "all_fit",
        "training_mask": "all_after_warmup",
        "objective_mode": STANDALONE_LABELS_MODE,
        "checkpoint_retention": "all_epochs",
        "checkpoint_endpoint_policy": "configured_final_epoch",
        "required_world_size": 1,
        "required_device": "L4",
        "precision_mode": "fp32",
        "profile_reload": True,
        "monitoring": {
            "train_eval_sequences": 128,
            "eval_batch_size": 4,
            "train_eval_selection": "hash_rank_v1",
        },
        "model": {
            "architecture": "gru",
            "hidden": 256,
            "layers": 2,
            "heads": 4,
            "context_rows": 512,
            "dropout": 0.0,
            "checkpoint_blocks": False,
            "hint_dim": 0,
            "attention_backend": "math",
            "compile_blocks": False,
        },
    }
    raw.update(changes)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


@pytest.mark.parametrize("target_index", [0, 1])
def test_specialist_updates_one_label_with_shared_two_label_denominator(
    target_index: int,
) -> None:
    target = torch.tensor([[[3.0, -0.5], [0.2, -0.8], [-1.5, 2.5]]])
    prediction = torch.tensor([[[0.2, -0.3], [0.7, 0.1], [-0.6, 0.5]]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    empty = torch.empty(1, 3, 0)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)

    denominators = window_denominators(target, mask, empty)
    numerators = objective_numerators(
        prediction,
        empty,
        target,
        torch.zeros_like(target),
        empty,
        mask,
        objective_mode=STANDALONE_LABELS_MODE,
        context=context,
        supervised_target=target_index,
    )
    clipped = target.clamp(-2.0, 2.0)
    expected_denominator = (clipped.abs() * mask[..., None]).sum()
    expected_numerator = (
        (prediction[..., target_index] - clipped[..., target_index]).square()
        * clipped[..., target_index].abs()
        * mask
    ).sum()
    loss = scaled_objective(numerators, denominators, 1, 0.0, 0.0)
    gradient = torch.autograd.grad(loss, prediction)[0]

    torch.testing.assert_close(denominators[0], expected_denominator)
    torch.testing.assert_close(numerators.label, expected_numerator)
    torch.testing.assert_close(loss, expected_numerator / expected_denominator)
    assert torch.count_nonzero(gradient[..., 1 - target_index]) == 0
    assert torch.count_nonzero(gradient[..., target_index]) > 0


@pytest.mark.parametrize("target_index", [None, 0, 1])
def test_opt_in_recipe_keeps_canonical_two_head_gru(
    target_index: int | None, tmp_path: Path
) -> None:
    path = _write_config(tmp_path, supervised_target=target_index)
    config = load_config(path)
    model = build_sequence_model(config["model"])
    prediction, _, _ = model(torch.zeros(1, 3, 112))
    assert prediction.shape == (1, 3, 2)
    assert config["profile_reload"] is True


@pytest.mark.parametrize("invalid", [True, 2, -1, 0.0])
def test_specialist_target_rejects_non_binary_integer(invalid: object, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supervised_target"):
        load_config(_write_config(tmp_path, supervised_target=invalid))


def test_sequence_runtime_can_require_l4_and_rejects_other_devices(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    assert validate_runtime_requirements(config, 1, "NVIDIA L4")
    with pytest.raises(ValueError, match="requires a L4 device"):
        validate_runtime_requirements(config, 1, "NVIDIA B300")


@pytest.mark.parametrize(
    "change",
    [
        {"kd_weight": 0.1},
        {
            "hint_weight": 0.1,
            "model": {
                "architecture": "gru",
                "hidden": 256,
                "layers": 2,
                "heads": 4,
                "context_rows": 512,
                "dropout": 0.0,
                "checkpoint_blocks": False,
                "hint_dim": 8,
                "attention_backend": "math",
                "compile_blocks": False,
            },
        },
        {
            "model": {
                "architecture": "gru",
                "hidden": 256,
                "layers": 2,
                "heads": 4,
                "context_rows": 512,
                "dropout": 0.0,
                "checkpoint_blocks": False,
                "hint_dim": 8,
                "attention_backend": "math",
                "compile_blocks": False,
            }
        },
        {"blend_objective_alpha": 0.1},
        {"objective_mode": "labels_wmse_v1"},
        {"correlation_prediction_std_floor": 0.01},
    ],
)
def test_specialist_recipe_rejects_auxiliary_and_mixed_objectives(
    change: dict[str, object], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="single-target supervision"):
        load_config(_write_config(tmp_path, supervised_target=0, **change))


def test_specialist_rejects_fit_reference_caches_but_can_use_read_only_monitor(
    tmp_path: Path,
) -> None:
    config = load_config(_write_config(tmp_path, supervised_target=1))
    validate_fit_reference_inputs(config, None, None, None)
    with pytest.raises(ValueError, match="reject --teacher-cache"):
        validate_fit_reference_inputs(config, Path("teacher"), None, None)
    with pytest.raises(ValueError, match="reject --reference-cache"):
        validate_fit_reference_inputs(config, None, Path("reference"), None)


@pytest.mark.parametrize("target_index", [0, 1])
def test_specialist_objective_contract_records_selected_target(
    target_index: int, tmp_path: Path
) -> None:
    config = load_config(_write_config(tmp_path, supervised_target=target_index))
    contract = objective_contract(config)
    assert contract is not None
    assert contract["kind"] == "single_target_numerator_two_target_denominator"
    assert contract["supervised_target"] == target_index
    assert contract["denominator"] == "sum_abs_clipped_t0_t1_over_mask"


def test_shared_control_preserves_both_target_numerator(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, supervised_target=None))
    target = torch.tensor([[[0.5, 1.5]]])
    prediction = torch.zeros_like(target)
    empty = torch.empty(1, 1, 0)
    mask = torch.ones(1, 1, dtype=torch.bool)
    context = DistributedContext(0, 0, 1, torch.device("cpu"), False)
    numerators = objective_numerators(
        prediction,
        empty,
        target,
        torch.zeros_like(target),
        empty,
        mask,
        objective_mode=STANDALONE_LABELS_MODE,
        context=context,
        supervised_target=config.get("supervised_target"),
    )
    assert numerators.label.item() == pytest.approx(0.5 * 0.5**2 + 1.5 * 1.5**2)


def test_profile_reload_opt_in_is_required_by_fit_admission(tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    identity = run_identity(config, {}, {"sha256": "a" * 64, "files": {}}, {}, 1)
    identity["model_parameters"] = {"total": 2, "trainable": 2}
    receipt: dict[str, object] = {
        "receipt_schema": "wnn.sequence-profile.v1",
        "status": "profile_complete",
        "approved": True,
        "identity": identity,
        "updates": 1,
        "forwarded_rows": 1,
        "real_rows": 1,
        "valid_rows": 1,
        "epochs_traversed": 0,
        "elapsed_seconds": 1.0,
        "scientific_checkpoint_written": False,
        "errors": [],
        "error_count": 0,
        "losses": {"label": 1.0, "distillation": 0.0, "hint": 0.0, "total": 1.0},
        "model_parameters": identity["model_parameters"],
        "memory_by_rank": [
            {
                "rank": 0,
                "peak_allocated_bytes": 1,
                "peak_reserved_bytes": 1,
                "device_total_memory_bytes": 1,
                "host_peak_rss_bytes": 1,
            }
        ],
        "peak_memory_bytes": 1,
        "peak_reserved_bytes": 1,
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="per-rank model/optimizer/RNG reload"):
        validate_profile_receipt(path, identity)

    receipt["engineering_reload"] = {
        "status": "passed",
        "scientific_resume_schema_written": False,
        "by_rank": [
            {
                "status": "passed",
                "model_optimizer_scheduler_rng_equal": True,
                "scientific_resume_schema_written": False,
            }
        ],
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    validate_profile_receipt(path, identity)
