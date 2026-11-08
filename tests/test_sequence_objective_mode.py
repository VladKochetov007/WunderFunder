from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from wnn.training.sequence_config import DEFAULTS, load_config
from wnn.training.sequence_fit_admission import (
    prepare_fit_admission,
    validate_fit_data_contract,
)
from wnn.training.sequence_objectives import (
    objective_numerators,
    scaled_objective,
    window_denominators,
)
from wnn.training.sequence_provenance import canonical_sha256

ROOT = Path(__file__).parents[1]
LEGACY_RECIPE = (
    ROOT
    / "research/experiment_drafts/generalization-fix-20261108/l4/configs/raw112_d0_s101101.json"
)
NEW_RECIPE = (
    ROOT / "research/experiment_drafts/training-repair-20261108/l4-standalone-labels-wmse-v2.json"
)
LEGACY_CONFIG_SHA256 = "4c7217b49191deb7add9b93da43a76c787dc594f3d0eb7b94ac1dbec4d398883"
PRIMARY_SPLIT_SCHEMA = "wnn.chronological-primary.v1"


class _Records:
    def __init__(
        self,
        members: list[tuple[str, int]],
        training_mask: str,
        *,
        cache: object | None = None,
        hint_members: list[tuple[str, int]] | None = None,
        hint_dim: int = 0,
    ) -> None:
        self.members = members
        self.training_mask = training_mask
        self.cache = cache
        self.hint_members = hint_members
        self.hint_dim = hint_dim

    def __len__(self) -> int:
        return len(self.members)


def _data_fixture() -> SimpleNamespace:
    train_groups, valid_groups = (3, 8), (1, 5)
    split_hash = "a" * 64
    split = SimpleNamespace(
        schema=PRIMARY_SPLIT_SCHEMA,
        train_groups=train_groups,
        valid_groups=(),
        evaluation_groups=valid_groups,
        evaluation_source="valid",
        evaluation_role="valid",
        evaluation_mask="is_scored",
        sha256=split_hash,
        fold=0,
    )
    provenance: dict[str, Any] = {
        "split_sha256": split_hash,
        "fit_members": [["train", group] for group in train_groups],
        "evaluation_members": [["valid", group] for group in valid_groups],
        "normalization": {"scope": "all_fit", "sequence_count": len(train_groups)},
        "teacher": None,
        "hint_members": None,
    }
    return SimpleNamespace(
        split=split,
        fit=_Records([("train", group) for group in train_groups], "all_after_warmup"),
        evaluation=_Records([("valid", group) for group in valid_groups], "is_scored"),
        provenance=provenance,
    )


def _chunked_weighted_loss_and_grad(
    config: dict[str, Any], prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted = prediction.clone().requires_grad_(True)
    teacher = torch.zeros_like(target)
    hint_prediction = predicted.new_empty((*predicted.shape[:2], 0))
    hint_target = hint_prediction.detach()
    denominator = window_denominators(target, mask, hint_target)
    chunks = [(0, 2), (2, target.shape[1])]
    objective = sum(
        (
            scaled_objective(
                objective_numerators(
                    predicted[:, start:end],
                    hint_prediction[:, start:end],
                    target[:, start:end],
                    teacher[:, start:end],
                    hint_target[:, start:end],
                    mask[:, start:end],
                ),
                denominator,
                1,
                float(config["kd_weight"]),
                float(config["hint_weight"]),
            )
            for start, end in chunks
        ),
        start=predicted.new_zeros(()),
    )
    gradient = torch.autograd.grad(objective, predicted)[0]
    return objective.detach(), gradient.detach()


def test_full_new_recipe_loads_and_legacy_identity_is_unchanged() -> None:
    legacy = load_config(LEGACY_RECIPE)
    candidate = load_config(NEW_RECIPE)
    expected = copy.deepcopy(legacy)
    expected["objective_mode"] = "standalone_labels_wmse_v2"
    expected["monitoring"]["train_eval_sequences"] = 128
    expected["monitoring"]["train_eval_selection"] = "hash_rank_v1"
    expected["monitoring"]["early_stopping"] = {
        "min_epoch": 2,
        "patience": 2,
        "min_delta": 0.0001,
        "action": "report_only",
    }
    expected["checkpoint_retention"] = "all_epochs"

    assert candidate == expected
    assert candidate["epochs"] == 6
    assert candidate["seed"] == 101101
    assert candidate["schedule"] == "warmup_cosine_v2"
    assert candidate["normalization_scope"] == "all_fit"
    assert candidate["training_mask"] == "all_after_warmup"
    assert candidate["kd_weight"] == candidate["hint_weight"] == 0
    assert "objective_mode" not in DEFAULTS
    assert "objective_mode" not in legacy
    assert canonical_sha256(legacy) == LEGACY_CONFIG_SHA256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kd_weight", 0.01),
        ("kd_weight", False),
        ("hint_weight", 0.01),
        ("model_hint_dim", 1),
        ("hint_mode", "component_permuted"),
        ("training_mask", "is_scored"),
        ("normalization_scope", "prefix"),
        ("objective_mode", "mixture_wmse_corr"),
        ("objective_mode", None),
        ("objective_correlation_weight", 0.1),
        ("blend_alpha", 0.5),
    ],
)
def test_new_mode_rejects_reference_terms_and_nonfit_policies(
    tmp_path: Path, field: str, value: object
) -> None:
    raw = json.loads(NEW_RECIPE.read_text(encoding="utf-8"))
    if field in ("objective_correlation_weight", "blend_alpha"):
        raw[field] = value
    elif field == "model_hint_dim":
        raw["model"]["hint_dim"] = value
    else:
        raw[field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("argument", ["teacher_cache", "initializer", "final_fit_manifest"])
def test_new_mode_rejects_teacher_initializer_and_final_fit_admission(
    tmp_path: Path, argument: str
) -> None:
    options: dict[str, object] = {argument: tmp_path / "forbidden-input"}
    with pytest.raises(ValueError, match="forbids|official primary split"):
        prepare_fit_admission(
            NEW_RECIPE,
            tmp_path / "fit",
            tmp_path / "primary-split.json",
            options.get("teacher_cache"),
            None,
            1,
            None,
            options.get("initializer"),
            options.get("final_fit_manifest"),
            None,
            None,
            tmp_path,
        )


def test_new_mode_data_contract_declares_wmse_and_qualified_ancestry() -> None:
    config = load_config(NEW_RECIPE)
    data = _data_fixture()

    assert validate_fit_data_contract(config, data, 1)
    assert data.provenance["objective_contract"] == {
        "kind": "standalone_label_weighted_mse",
        "mode": "standalone_labels_wmse_v2",
        "target_clip": [-2.0, 2.0],
        "target_weight": "abs(clipped_target)",
        "aggregation": "global_ddp_weighted_wmse_per_actual_optimizer_window",
        "pearson_term": "none",
        "reference_training_ancestry": "none",
    }
    admission = data.provenance["objective_admission"]
    assert admission["fit_membership"] == "official_train_fit_only"
    assert admission["normalization_membership"] == "all_fitting_members"
    assert admission["evaluation_source"] == admission["evaluation_role"] == "valid"
    assert admission["evaluation_mask"] == "is_scored"
    assert admission["teacher_source"] == admission["initializer_source"] == "none"


@pytest.mark.parametrize(
    "invalid",
    [
        "control",
        "teacher",
        "initializer",
        "fit_members",
        "evaluation_members",
        "normalization",
        "hint_dim",
        "missing_teacher_declaration",
        "missing_hint_declaration",
    ],
)
def test_new_mode_rejects_nonprimary_or_referenced_data(invalid: str) -> None:
    config = load_config(NEW_RECIPE)
    data = _data_fixture()
    if invalid == "control":
        data.split.evaluation_source = "train"
        data.split.evaluation_role = "control"
    elif invalid == "teacher":
        data.provenance["teacher"] = {"kind": "teacher"}
    elif invalid == "initializer":
        data.provenance["initializer"] = {"sha256": "b" * 64}
    elif invalid == "fit_members":
        data.fit.members[0] = ("valid", data.fit.members[0][1])
    elif invalid == "evaluation_members":
        data.evaluation.members.pop()
    elif invalid == "normalization":
        data.provenance["normalization"]["scope"] = "prefix"
    elif invalid == "hint_dim":
        data.fit.hint_dim = 1
    elif invalid == "missing_teacher_declaration":
        data.provenance.pop("teacher")
    elif invalid == "missing_hint_declaration":
        data.provenance.pop("hint_members")

    with pytest.raises(ValueError, match="full official-valid primary evaluation"):
        validate_fit_data_contract(config, data, 1)


def test_new_mode_wmse_loss_and_gradient_match_legacy_label_control() -> None:
    legacy = load_config(LEGACY_RECIPE)
    candidate = load_config(NEW_RECIPE)
    prediction = torch.tensor([[[0.1, -0.3], [0.5, 0.7], [1.0, -1.0], [2.5, 0.2]]])
    target = torch.tensor([[[0.4, -0.2], [3.0, 0.0], [-2.5, 0.8], [1.0, -0.7]]])
    mask = torch.tensor([[True, True, False, True]])

    legacy_loss, legacy_gradient = _chunked_weighted_loss_and_grad(legacy, prediction, target, mask)
    new_loss, new_gradient = _chunked_weighted_loss_and_grad(candidate, prediction, target, mask)
    clipped = target.clamp(-2.0, 2.0)
    weights = clipped.abs() * mask[..., None]
    denominator = weights.sum()
    expected_loss = ((prediction - clipped).square() * weights).sum() / denominator
    expected_gradient = 2.0 * weights * (prediction - clipped) / denominator

    torch.testing.assert_close(legacy_loss, new_loss, rtol=0, atol=0)
    torch.testing.assert_close(legacy_gradient, new_gradient, rtol=0, atol=0)
    torch.testing.assert_close(new_loss, expected_loss, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(new_gradient, expected_gradient, rtol=1e-6, atol=1e-7)
