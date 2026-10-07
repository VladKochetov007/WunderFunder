"""Focused, low-cost checks derived from the frozen distillation contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts import package_submission, train_rnn
from wnn.distillation import append_family_output_head, extend_family_output_head
from wnn.splits import CampaignSplit

ROOT = Path(__file__).resolve().parents[1]


def _small_gru_args() -> dict[str, object]:
    return {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 4,
        "layers": 1,
        "dropout": 0.0,
        "readout_activation": "gelu",
    }


def _first_shuffled_epoch(*, family_head: bool) -> torch.Tensor:
    """Follow the C2 or supported C3 initialization with RandomSampler's first epoch."""
    torch.manual_seed(3)
    model = train_rnn.make_model(_small_gru_args(), n_in=3, n_out=23)
    if family_head:
        model = append_family_output_head(model)
    loader = DataLoader(TensorDataset(torch.arange(32)), batch_size=4, shuffle=True, num_workers=0)
    order = torch.cat([batch[0] for batch in loader])
    del model
    return order


def _first_shuffled_epoch_with_direct_family_width() -> torch.Tensor:
    """Reproduce the old construction that let eight extra rows shift global RNG."""
    torch.manual_seed(3)
    model = train_rnn.make_model(_small_gru_args(), n_in=3, n_out=31)
    loader = DataLoader(TensorDataset(torch.arange(32)), batch_size=4, shuffle=True, num_workers=0)
    order = torch.cat([batch[0] for batch in loader])
    del model
    return order


def test_family_head_keeps_same_seed_random_sampler_order() -> None:
    """C2/C3 matched seeds must not change row order when only the head is added."""
    c2_order = _first_shuffled_epoch(family_head=False)
    c3_order = _first_shuffled_epoch(family_head=True)

    torch.testing.assert_close(c2_order, c3_order, rtol=0, atol=0)


def test_direct_31_output_initialization_reproduces_sampler_order_confound() -> None:
    """The former construction is a regression witness for the global RNG shift."""
    c2_order = _first_shuffled_epoch(family_head=False)
    direct_c3_order = _first_shuffled_epoch_with_direct_family_width()

    assert not torch.equal(c2_order, direct_c3_order)


def test_family_head_warm_start_preserves_23_rows_and_package_uses_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The training checkpoint keeps parent rows; the native interface keeps t0/t1."""
    parent = {
        "trunk": torch.arange(12, dtype=torch.float32),
        "head.2.weight": torch.arange(23 * 3, dtype=torch.float32).reshape(23, 3),
        "head.2.bias": torch.arange(23, dtype=torch.float32),
    }
    initialized = {
        "trunk": torch.zeros(12),
        "head.2.weight": torch.full((31, 3), -5.0),
        "head.2.bias": torch.full((31,), -5.0),
    }
    merged = extend_family_output_head(parent, initialized)
    torch.testing.assert_close(merged["trunk"], parent["trunk"], rtol=0, atol=0)
    torch.testing.assert_close(
        merged["head.2.weight"][:23], parent["head.2.weight"], rtol=0, atol=0
    )
    torch.testing.assert_close(merged["head.2.bias"][:23], parent["head.2.bias"], rtol=0, atol=0)

    checkpoint = {
        "args": {"readout_activation": "gelu"},
        "model": merged,
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    package_path = tmp_path / "weights.npz"
    package_submission.export("critic_student", package_path)

    with np.load(package_path, allow_pickle=False) as packaged:
        np.testing.assert_array_equal(
            packaged["head.2.weight"], parent["head.2.weight"][:2].numpy()
        )
        np.testing.assert_array_equal(packaged["head.2.bias"], parent["head.2.bias"][:2].numpy())


def test_family_disabled_model_has_legacy_23_output_shape() -> None:
    """The opt-in head is represented only by a larger requested output width."""
    torch.manual_seed(11)
    model = train_rnn.make_model(_small_gru_args(), n_in=3, n_out=23)
    with torch.inference_mode():
        output, _state = model(torch.zeros(2, 5, 3))
    assert output.shape == (2, 5, 23)


def test_family_head_extension_rejects_unsupported_model() -> None:
    """Only the supported Sequential(..., Linear) output head may be extended."""
    with pytest.raises(ValueError, match="sequential linear output head"):
        append_family_output_head(torch.nn.Linear(3, 23))


def test_teacher_membership_rejects_corrupt_train_and_evaluation_groups() -> None:
    """Teacher fit provenance may use only the clean training and validation pools."""
    from scripts.build_distill_cache import _verify_run_membership
    from wnn.data import CORRUPT_TRAIN_ROW_GROUPS

    split = CampaignSplit(
        Path("fold0.json"),
        "fold0-sha",
        0,
        4,
        (10,),
        (20,),
        (21,),
        (("train", 10, 0), ("valid", 20, 1), ("valid", 21, 2)),
    )
    assert 528 in CORRUPT_TRAIN_ROW_GROUPS
    provenance = {
        "split_manifest_sha256": split.sha256,
        "fit_row_groups": {"train": [528], "valid": [20]},
    }
    with pytest.raises(ValueError, match="outside the clean train pool"):
        _verify_run_membership(provenance, split, "corrupt_teacher")
    provenance["fit_row_groups"] = {"train": [10], "valid": [21]}
    with pytest.raises(ValueError, match="outside the clean valid pool"):
        _verify_run_membership(provenance, split, "eval_teacher")


def _parse_registered_command(command: list[str]) -> dict[str, str | bool]:
    assert command[:2] == [".venv/bin/python", "scripts/train_rnn.py"]
    options: dict[str, str | bool] = {}
    index = 2
    while index < len(command):
        option = command[index]
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            options[option] = command[index + 1]
            index += 2
        else:
            options[option] = True
            index += 1
    return options


def test_minimal_c1_has_no_full_family_cache_dependency() -> None:
    path = (
        ROOT / "research/campaign_2026-10-06/bandit_continuation/local/E511_C1_MINIMAL_PAIRED.json"
    )
    recipe = json.loads(path.read_text())
    assert len(recipe["commands"]) == 2
    assert len(recipe["profile_only_commands"]) == 2
    for command in recipe["commands"]:
        options = _parse_registered_command(command)
        assert "--distill-cache-root" not in options
        assert "--distill-cache-variant" not in options
        assert options["--distill"] == ("c20261003_combo_s3_f0*0.8125,c20261003_combo_s4_f0*0.8125")
        assert "--save-training-state" in options
    for command in recipe["profile_only_commands"]:
        options = _parse_registered_command(command)
        assert options["--profile-steps"] == "2"
        assert "--save-training-state" not in options


def test_r1_family_target_recipes_are_matched_except_for_the_target_mode() -> None:
    path = (
        ROOT
        / "research/campaign_2026-10-06/bandit_continuation/local/R1_FAMILY_TARGET_MODE_PAIRED.json"
    )
    recipe = json.loads(path.read_text())
    assert len(recipe["commands"]) == 4
    commands = [_parse_registered_command(command) for command in recipe["commands"]]
    for seed_index in range(2):
        clipped = commands[seed_index]
        raw = commands[seed_index + 2]
        assert clipped["--distill-family-target-mode"] == "clip2"
        assert raw["--distill-family-target-mode"] == "raw"
        assert clipped["--distill-cache-variant"] == "source"
        assert clipped["--distill-family-weight"] == "0.1"
        assert {
            key: value
            for key, value in clipped.items()
            if key not in {"--name", "--distill-family-target-mode"}
        } == {
            key: value
            for key, value in raw.items()
            if key not in {"--name", "--distill-family-target-mode"}
        }
