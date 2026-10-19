from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest
import torch

from scripts.predict_run import restore_input_representation
from wnn.training.environment import InputConfiguration, TrainingEnvironment
from wnn.training.input_representation import (
    CODE_WIDTH,
    EMA_ALPHA,
    EXTRA_WIDTH,
    RAW_WIDTH,
    SLOT_WIDTH,
    FittedInputRepresentation,
    build_representation_features,
    causal_signed_ema,
    deserialize_input_representation,
    extend_input_projection_state,
    serialize_input_representation,
)
from wnn.training.input_representation_dictionary import raw_normalizer
from wnn.training.input_representation_fit import (
    REPRESENTATION_RECIPE,
    representation_recipe_sha256,
    select_fit_members,
    validate_fitted_representation,
)
from wnn.training.inputs import build_inputs


def _naive_ema(values: torch.Tensor) -> torch.Tensor:
    """Small serial FP32 recurrence used as a numerical oracle."""
    result = torch.zeros_like(values, dtype=torch.float32)
    alpha = torch.tensor(EMA_ALPHA, dtype=torch.float32)
    beta = torch.tensor(1.0 - EMA_ALPHA, dtype=torch.float32)
    source = values.to(torch.float32)
    for row in range(1, source.shape[1]):
        result[:, row] = alpha * (source[:, row] - source[:, row - 1]) + beta * result[:, row - 1]
    return result


def _fitted() -> FittedInputRepresentation:
    """Build a deterministic synthetic dictionary with the expected dimensions."""
    generator = torch.Generator().manual_seed(19082)
    return FittedInputRepresentation(
        raw_mean=torch.zeros(RAW_WIDTH, dtype=torch.float32),
        raw_scale=torch.ones(RAW_WIDTH, dtype=torch.float32),
        encoder_weight=torch.randn(CODE_WIDTH, RAW_WIDTH, generator=generator),
        decoder_weight=torch.randn(RAW_WIDTH, CODE_WIDTH, generator=generator),
        fit_metadata={"fit_members_sha256": "synthetic"},
    )


def test_causal_ema_matches_fp32_recurrence_and_resets_per_sequence() -> None:
    values = torch.randn((2, 700, 5), generator=torch.Generator().manual_seed(12))
    actual = causal_signed_ema(values)
    expected = _naive_ema(values)
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, rtol=2e-5, atol=2e-6)
    assert torch.equal(actual[:, 0], torch.zeros_like(actual[:, 0]))
    assert torch.equal(actual[1:], causal_signed_ema(values[1:]))
    destination = torch.empty_like(actual)
    written = causal_signed_ema(values, output=destination)
    assert written.data_ptr() == destination.data_ptr()
    assert torch.equal(written, actual)
    changed = values.clone()
    changed[:, 500:] += 10
    assert torch.equal(actual[:, :500], causal_signed_ema(changed)[:, :500])


def test_all_modes_have_fixed_suffix_slots_and_fp32_ema() -> None:
    raw = torch.randn((2, 300, RAW_WIDTH), generator=torch.Generator().manual_seed(41))
    fitted = _fitted()
    features = {
        mode: build_representation_features(raw, mode, fitted)
        for mode in ("base", "ema", "sparse", "ema_sparse")
    }
    for value in features.values():
        assert value.shape == (*raw.shape[:2], EXTRA_WIDTH)
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()
    assert torch.count_nonzero(features["base"]) == 0
    assert torch.count_nonzero(features["ema"][..., :SLOT_WIDTH]) == 0
    assert torch.count_nonzero(features["sparse"][..., SLOT_WIDTH:]) == 0
    assert torch.count_nonzero(features["ema_sparse"][..., :SLOT_WIDTH]) > 0
    expected_raw_ema = causal_signed_ema(raw)
    assert torch.allclose(
        features["ema"][..., SLOT_WIDTH : SLOT_WIDTH + RAW_WIDTH], expected_raw_ema
    )
    assert torch.count_nonzero(features["ema"][..., -CODE_WIDTH:]) == 0
    expected_dictionary_ema = causal_signed_ema(features["sparse"][..., :SLOT_WIDTH])
    assert torch.equal(features["ema_sparse"][..., SLOT_WIDTH:], expected_dictionary_ema)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_features = build_representation_features(raw, "ema_sparse", fitted)
    assert autocast_features.dtype == torch.float32
    assert torch.equal(autocast_features, features["ema_sparse"])


def test_fit_statistics_are_applied_with_matching_fp32_normalization() -> None:
    raw_rows = np.random.default_rng(71).normal(size=(19, RAW_WIDTH)).astype(np.float32)
    mean, scale = raw_normalizer(raw_rows)
    mean_fp32 = mean.astype(np.float32)
    scale_fp32 = scale.astype(np.float32)
    expected = (raw_rows - mean_fp32) / scale_fp32
    fitted = FittedInputRepresentation(
        raw_mean=torch.from_numpy(mean_fp32),
        raw_scale=torch.from_numpy(scale_fp32),
        encoder_weight=torch.zeros((CODE_WIDTH, RAW_WIDTH)),
        decoder_weight=torch.zeros((RAW_WIDTH, CODE_WIDTH)),
        fit_metadata={},
    )
    features = build_representation_features(torch.from_numpy(raw_rows[None]), "sparse", fitted)
    torch.testing.assert_close(
        features[0, :, CODE_WIDTH:SLOT_WIDTH],
        torch.from_numpy(expected),
        rtol=0.0,
        atol=0.0,
    )


def test_input_assembly_preserves_legacy_prefix_and_adds_256_slots() -> None:
    raw = torch.randn((2, 9, RAW_WIDTH), generator=torch.Generator().manual_seed(73))
    legacy = build_inputs(raw, environment=TrainingEnvironment())
    environment = TrainingEnvironment(
        inputs=InputConfiguration(
            representation_mode="base",
            input_representation=_fitted(),
        )
    )
    experimental = build_inputs(raw, environment=environment)
    assert legacy.shape[-1] == 211
    assert experimental.shape[-1] == 467
    assert torch.equal(experimental[..., :211], legacy)
    assert torch.count_nonzero(experimental[..., 211:]) == 0


def test_warm_start_projection_copies_parent_and_zeros_added_columns() -> None:
    generator = torch.Generator().manual_seed(101)
    parent = {
        "inp.weight": torch.randn((7, 211), generator=generator),
        "mean": torch.randn(211, generator=generator),
        "std": torch.rand(211, generator=generator) + 0.1,
        "inp.bias": torch.randn(7, generator=generator),
    }
    target = {
        "inp.weight": torch.empty((7, 467)),
        "mean": torch.empty(467),
        "std": torch.empty(467),
    }
    extended = extend_input_projection_state(parent, target)
    assert torch.equal(extended["inp.weight"][:, :211], parent["inp.weight"])
    assert torch.count_nonzero(extended["inp.weight"][:, 211:]) == 0
    assert torch.equal(extended["mean"][:211], parent["mean"])
    assert torch.count_nonzero(extended["mean"][211:]) == 0
    assert torch.equal(extended["std"][:211], parent["std"])
    assert torch.equal(extended["std"][211:], torch.ones(256))
    assert extended["inp.bias"] is parent["inp.bias"]
    raw = torch.randn((4, 211), generator=generator)
    extra = torch.randn((4, 256), generator=generator)
    old_normalized = (raw - parent["mean"]) / parent["std"]
    new_normalized = torch.cat((old_normalized, extra / extended["std"][211:]), dim=-1)
    assert torch.equal(
        torch.nn.functional.linear(old_normalized, parent["inp.weight"], parent["inp.bias"]),
        torch.nn.functional.linear(new_normalized, extended["inp.weight"], parent["inp.bias"]),
    )


def test_representation_checkpoint_round_trip_is_weights_only_safe() -> None:
    original = _fitted()
    buffer = io.BytesIO()
    torch.save(serialize_input_representation(original), buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location="cpu", weights_only=True)
    restored = deserialize_input_representation(payload)
    for name in ("raw_mean", "raw_scale", "encoder_weight", "decoder_weight"):
        assert torch.equal(getattr(restored, name), getattr(original, name))
        assert getattr(restored, name).dtype == torch.float32
    payload["state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        deserialize_input_representation(payload)


def test_embedded_dictionary_requires_the_exact_fold_recipe() -> None:
    members = [{"source": "train", "group": 3, "component": 8}]
    member_text = json.dumps(members, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    metadata = {
        "split_manifest_sha256": "split-sha",
        "fit_members": members,
        "fit_members_sha256": hashlib.sha256(member_text.encode("ascii")).hexdigest(),
        "recipe": dict(REPRESENTATION_RECIPE),
        "recipe_sha256": representation_recipe_sha256(),
    }
    base = _fitted()
    fitted = FittedInputRepresentation(
        base.raw_mean, base.raw_scale, base.encoder_weight, base.decoder_weight, metadata
    )
    validate_fitted_representation(fitted, "split-sha", members)
    incompatible = dict(metadata)
    incompatible["recipe"] = {**REPRESENTATION_RECIPE, "top_k": 3}
    invalid = FittedInputRepresentation(
        base.raw_mean, base.raw_scale, base.encoder_weight, base.decoder_weight, incompatible
    )
    with pytest.raises(ValueError, match="fixed fold recipe"):
        validate_fitted_representation(invalid, "split-sha", members)


def test_prediction_restores_representation_and_rejects_mismatched_layout() -> None:
    fitted = _fitted()
    payload = serialize_input_representation(fitted)
    mode, restored = restore_input_representation(
        {
            "args": {"representation_mode": "ema_sparse"},
            "n_in": 467,
            "input_representation": payload,
        },
        torch.device("cpu"),
    )
    assert mode == "ema_sparse"
    assert restored is not None
    assert torch.equal(restored.encoder_weight, fitted.encoder_weight)
    raw = torch.randn((1, 270, RAW_WIDTH), generator=torch.Generator().manual_seed(77))
    expected = build_inputs(
        raw,
        environment=TrainingEnvironment(
            inputs=InputConfiguration(
                representation_mode=mode,
                input_representation=fitted,
            )
        ),
    )
    actual = build_inputs(
        raw,
        environment=TrainingEnvironment(
            inputs=InputConfiguration(
                representation_mode=mode,
                input_representation=restored,
            )
        ),
    )
    assert torch.equal(actual, expected)
    assert restore_input_representation({"args": {}, "n_in": 211}, torch.device("cpu")) == (
        None,
        None,
    )
    with pytest.raises(ValueError, match="467-wide"):
        restore_input_representation(
            {"args": {"representation_mode": "base"}, "n_in": 211, "input_representation": payload},
            torch.device("cpu"),
        )


def test_fit_member_selection_is_train_only_and_deterministic() -> None:
    class SplitFixture:
        """Minimal campaign split fixture for the selector boundary."""

        def __init__(self) -> None:
            self.components = [
                (source, group, component)
                for component in range(70)
                for source, group in (
                    ("train", component * 10 + 1),
                    ("train", component * 10 + 3),
                    ("valid", component * 10 + 5),
                )
            ]

        def assert_fit_groups(self, groups: np.ndarray, extra: tuple[int, ...]) -> None:
            assert extra == ()
            assert set(map(int, groups)) == {
                group for source, group, _ in self.components if source == "train"
            }

    split = SplitFixture()
    train_groups = np.asarray(
        [group for source, group, _ in split.components if source == "train"], dtype=np.int64
    )
    first = select_fit_members(train_groups, split)
    second = select_fit_members(train_groups, split)
    assert first == second
    assert len(first) == 64
    assert all(member["source"] == "train" for member in first)
    assert all(int(member["group"]) % 10 == 1 for member in first)
    assert first == sorted(first, key=lambda member: int(member["component"]))
