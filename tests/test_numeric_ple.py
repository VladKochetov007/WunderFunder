from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.export_onnx as export_onnx
import scripts.predict_run as predict_run
import scripts.train_rnn as train_rnn
from scripts.train_rnn import load_component_fit_subset, load_e540_fit_subset
from wnn.features import event_features_torch
from wnn.nn import RNNModel
from wnn.numeric_ple import NumericFeatureMap, fit_training_quantile_knots
from wnn.sentinel import SENTINEL
from wnn.splits import CampaignSplit
from wnn.streaming import StreamingGraph, initial_state


def _knots(n_intervals: int) -> tuple[np.ndarray, np.ndarray]:
    knots = np.tile(np.linspace(-2.0, 2.0, n_intervals + 1, dtype=np.float32), (112, 1))
    counts = np.full(112, n_intervals + 1, dtype=np.int64)
    return knots, counts


def _small_model(mode: str) -> RNNModel:
    return RNNModel(
        211,
        hidden=8,
        layers=1,
        dropout=0.0,
        n_out=2,
        numeric_map=mode,
        ple_intervals=2,
    )


def _stream_predictions(model: RNNModel, raw: torch.Tensor) -> torch.Tensor:
    graph = StreamingGraph(model).eval()
    state = initial_state(model)
    predictions = []
    with torch.no_grad():
        for row in raw.unbind(dim=1):
            output = graph(row, *state)
            predictions.append(output[0])
            state = list(output[1:])
    return torch.stack(predictions, dim=1)


def test_quantile_fit_is_bounded_membership_bound_and_handles_ties() -> None:
    train_a = np.zeros((4, 112), dtype=np.float32)
    train_a[:, 0] = np.arange(4, dtype=np.float32)
    train_a[:, 1] = 7.0
    train_a[:, 2] = SENTINEL
    train_a[:, 3] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    train_b = np.zeros((4, 112), dtype=np.float32)
    train_b[:, 0] = np.arange(4, 8, dtype=np.float32)
    train_b[:, 1] = 7.0
    train_b[:, 2] = SENTINEL
    train_b[:, 3] = np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32)
    withheld = np.full((4, 112), 1_000.0, dtype=np.float32)

    knots, counts, metadata = fit_training_quantile_knots(
        [train_a, train_b],
        [("train", 3), ("train", 8)],
        np.zeros(112, dtype=np.float32),
        np.ones(112, dtype=np.float32),
        n_intervals=4,
        samples_per_sequence=4,
        max_samples=8,
        seed=17,
    )

    assert withheld.max() > knots[0, -1]
    assert counts[0] == 5
    assert knots[0, 0] == 0.0
    assert knots[0, -1] == 7.0
    assert counts[1] == 1
    assert np.all(knots[1] == 7.0)
    assert counts[2] == 1
    assert np.all(knots[2] == SENTINEL)
    assert counts[3] == 3
    assert np.array_equal(knots[3, :3], np.array([0.0, 0.5, 1.0], dtype=np.float32))
    assert np.all(knots[3, 3:] == 1.0)
    assert metadata["candidate_sample_rows"] == 8
    assert metadata["retained_sample_rows"] == 8
    expected_members = json.dumps([["train", 3], ["train", 8]], separators=(",", ":")).encode(
        "ascii"
    )
    assert metadata["fit_members_sha256"] == hashlib.sha256(expected_members).hexdigest()


def test_quantile_fit_rejects_nonfinite_or_mismatched_feature_sequences() -> None:
    mean = np.zeros(112, dtype=np.float32)
    std = np.ones(112, dtype=np.float32)
    nonfinite = np.zeros((2, 112), dtype=np.float32)
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_training_quantile_knots([nonfinite], [("train", 1)], mean, std)
    with pytest.raises(ValueError):
        fit_training_quantile_knots([], [("train", 1)], mean, std)
    with pytest.raises(ValueError, match="train/valid"):
        fit_training_quantile_knots(
            [np.zeros((1, 112), dtype=np.float32)], [("evaluation", 1)], mean, std
        )


def test_quantile_sequence_reader_uses_only_explicit_fit_membership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[tuple[str, int, bool, bool]] = []

    class FeatureOnlyReader:
        def __init__(self, path: Path) -> None:
            self.split = path.stem

        def read(self, group: int, *, targets: bool, mask: bool) -> SimpleNamespace:
            reads.append((self.split, group, targets, mask))
            return SimpleNamespace(features=np.full((1, 112), group, dtype=np.float32))

    monkeypatch.setattr(train_rnn, "ROOT", tmp_path)
    monkeypatch.setattr(train_rnn, "SequenceFile", FeatureOnlyReader)
    features = list(train_rnn.iter_e540_fit_features([3, 8], [12]))

    assert [float(array[0, 0]) for array in features] == [3.0, 8.0, 12.0]
    assert reads == [
        ("train", 3, False, False),
        ("train", 8, False, False),
        ("valid", 12, False, False),
    ]


def test_ple_slots_are_continuous_and_keep_unbounded_raw_bypass() -> None:
    ple = NumericFeatureMap(113, "ple", n_intervals=2)
    with pytest.raises(RuntimeError, match="training membership"):
        ple(torch.zeros(1, 1, 113))
    knots, counts = _knots(2)
    knots[0] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    ple.set_knots(knots, counts)
    inputs = torch.zeros(1, 5, 113)
    inputs[0, :, 0] = torch.tensor([-1.0, 0.5, 1.0, 1.5, 7.0])
    inputs[0, :, 112] = torch.tensor([3.0, 4.0, 5.0, 6.0, 7.0])

    encoded = ple(inputs)
    feature_zero = encoded[0, :, :3]
    assert torch.allclose(
        feature_zero,
        torch.tensor(
            [[0.0, 0.0, -1.0], [0.5, 0.0, 0.5], [1.0, 0.0, 1.0], [1.0, 0.5, 1.5], [1.0, 1.0, 7.0]]
        ),
    )
    assert torch.equal(encoded[..., 112 * 3], inputs[..., 112])
    near_knot = inputs[:, 1:4].clone()
    near_knot[0, :, 0] = torch.tensor([0.9999, 1.0, 1.0001])
    near_encoded = ple(near_knot)[0, :, :2]
    assert torch.max(torch.abs(near_encoded[1:] - near_encoded[:-1])) < 2e-4

    linear = NumericFeatureMap(113, "linear", n_intervals=2)
    linear_slots = linear(inputs)[0, :, :3]
    assert torch.equal(linear_slots[:, :2], torch.zeros_like(linear_slots[:, :2]))
    assert torch.equal(linear_slots[:, 2], inputs[0, :, 0])


def test_matched_models_share_parameters_and_ple_checkpoint_state() -> None:
    torch.manual_seed(412)
    linear = _small_model("linear")
    torch.manual_seed(412)
    ple = _small_model("ple")
    assert {name: value.shape for name, value in linear.named_parameters()} == {
        name: value.shape for name, value in ple.named_parameters()
    }
    assert sum(parameter.numel() for parameter in linear.parameters()) == sum(
        parameter.numel() for parameter in ple.parameters()
    )
    ple.load_state_dict(linear.state_dict())
    knots, counts = _knots(2)
    ple.numeric_feature_map.set_knots(knots, counts)
    inputs = torch.randn(1, 11, 211)
    raw = inputs[..., :112]

    basis_columns = torch.arange(112).unsqueeze(1) * 3 + torch.arange(2).unsqueeze(0)
    with torch.no_grad():
        ple.inp.weight[:, basis_columns.flatten()] = 0.0
    linear.eval()
    ple.eval()
    linear_out, _ = linear(inputs)
    ple_out, _ = ple(inputs)
    assert torch.allclose(linear_out, ple_out, atol=1e-6, rtol=1e-6)

    with torch.no_grad():
        ple.inp.weight[:, basis_columns.flatten()] = 0.03
    full, _ = ple(inputs)
    first, state = ple(inputs[:, :4])
    rest, _ = ple(inputs[:, 4:], state)
    assert torch.allclose(full, torch.cat((first, rest), dim=1), atol=1e-6, rtol=1e-6)

    restored = _small_model("ple")
    restored.load_state_dict(ple.state_dict())
    restored.eval()
    reloaded, _ = restored(inputs)
    assert torch.equal(restored.numeric_feature_map.knots, ple.numeric_feature_map.knots)
    assert torch.allclose(full, reloaded, atol=1e-6, rtol=1e-6)

    restored.train()
    gradient_out, _ = restored(inputs)
    gradient_out.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in restored.parameters()
    )
    assert raw.shape[-1] == 112


def test_screen_models_allocate_equal_274050_parameters() -> None:
    linear = RNNModel(211, hidden=128, layers=1, dropout=0.0, n_out=2, numeric_map="linear")
    ple = RNNModel(211, hidden=128, layers=1, dropout=0.0, n_out=2, numeric_map="ple")
    linear_count = sum(parameter.numel() for parameter in linear.parameters())
    ple_count = sum(parameter.numel() for parameter in ple.parameters())

    assert linear.numeric_feature_map.output_width == 1107
    assert ple.numeric_feature_map.output_width == 1107
    assert linear_count == ple_count == 274_050
    assert (112 * 8 * 128) == 114_688


def test_ple_matches_every_row_stream_and_resets_between_sequences() -> None:
    torch.manual_seed(83)
    model = _small_model("ple").eval()
    knots, counts = _knots(2)
    model.numeric_feature_map.set_knots(knots, counts)
    first_raw = torch.randn(1, 8, 112)
    second_raw = torch.randn(1, 9, 112)

    for raw in (first_raw, second_raw):
        full_inputs = torch.cat((raw, event_features_torch(raw)), dim=-1)
        with torch.no_grad():
            full, _ = model(full_inputs)
        streamed = _stream_predictions(model, raw)
        assert torch.allclose(full[..., :2], streamed, atol=2e-6, rtol=2e-6)

    graph = StreamingGraph(model).eval()
    state = initial_state(model)
    with torch.no_grad():
        for row in first_raw.unbind(dim=1):
            output = graph(row, *state)
            state = list(output[1:])
    state = initial_state(model)
    second_stream = []
    with torch.no_grad():
        for row in second_raw.unbind(dim=1):
            output = graph(row, *state)
            second_stream.append(output[0])
            state = list(output[1:])
        second_streamed = torch.stack(second_stream, dim=1)
    second_inputs = torch.cat((second_raw, event_features_torch(second_raw)), dim=-1)
    with torch.no_grad():
        second_full, _ = model(second_inputs)
    assert torch.allclose(second_full[..., :2], second_streamed, atol=2e-6, rtol=2e-6)


def _subset_manifest(
    train: list[int],
    valid: list[int],
    components: list[int],
    evaluation: list[int],
    experiment_id: str = "E540",
) -> dict[str, object]:
    members = sorted([("train", group) for group in train] + [("valid", group) for group in valid])
    return {
        "version": 1,
        "experiment_id": experiment_id,
        "split_manifest_sha256": "split-sha",
        "selected_groups": {"train": train, "valid": valid},
        "selected_components": components,
        "selected_component_sha256": hashlib.sha256(
            np.asarray(components, dtype="<i8").tobytes()
        ).hexdigest(),
        "evaluation_groups": evaluation,
        "evaluation_group_count": len(evaluation),
        "evaluation_membership_sha256": hashlib.sha256(
            np.asarray(evaluation, dtype="<i8").tobytes()
        ).hexdigest(),
        "selected_members_sha256": hashlib.sha256(
            json.dumps(members, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
    }


def test_fit_subset_requires_complete_non_evaluation_components(tmp_path: Path) -> None:
    split = CampaignSplit(
        path=tmp_path / "split.json",
        sha256="split-sha",
        fold=0,
        n_folds=2,
        train_groups=(1, 2, 3, 4),
        valid_groups=(10, 11, 12, 13),
        evaluation_groups=(14,),
        components=(
            ("train", 1, 100),
            ("train", 2, 100),
            ("train", 3, 200),
            ("train", 4, 300),
            ("valid", 10, 100),
            ("valid", 11, 200),
            ("valid", 12, 300),
            ("valid", 13, 400),
            ("valid", 14, 500),
        ),
    )
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(_subset_manifest([1], [10], [100], [14])))
    with pytest.raises(ValueError, match="splits an eligible fitting component"):
        load_e540_fit_subset(path, split)

    path.write_text(json.dumps(_subset_manifest([1, 2], [10], [100], [14])))
    subset = load_e540_fit_subset(path, split)
    assert subset["train_groups"] == [1, 2]
    assert subset["valid_groups"] == [10]

    invalid_evaluation = _subset_manifest([1, 2], [10], [100], [14])
    invalid_evaluation["evaluation_groups"] = [13]
    invalid_evaluation["evaluation_group_count"] = 1
    invalid_evaluation["evaluation_membership_sha256"] = hashlib.sha256(
        np.asarray([13], dtype="<i8").tobytes()
    ).hexdigest()
    path.write_text(json.dumps(invalid_evaluation))
    with pytest.raises(ValueError, match="exact evaluation groups"):
        load_e540_fit_subset(path, split)

    path.write_text(json.dumps(_subset_manifest([1, 2], [14], [100], [14])))
    with pytest.raises(ValueError, match="outside the fit split"):
        load_e540_fit_subset(path, split)


def test_component_fit_subset_binds_non_e540_experiment(tmp_path: Path) -> None:
    split = CampaignSplit(
        path=tmp_path / "split.json",
        sha256="split-sha",
        fold=0,
        n_folds=2,
        train_groups=(1, 2),
        valid_groups=(10,),
        evaluation_groups=(14,),
        components=(
            ("train", 1, 100),
            ("train", 2, 100),
            ("valid", 10, 100),
            ("valid", 14, 500),
        ),
    )
    experiment_id = "E-QAT-DEPLOYMENT-20261013-R1"
    path = tmp_path / "qat_subset.json"
    path.write_text(json.dumps(_subset_manifest([1, 2], [10], [100], [14], experiment_id)))

    subset = load_component_fit_subset(path, split, expected_experiment_id=experiment_id)
    assert subset["train_groups"] == [1, 2]
    assert subset["valid_groups"] == [10]
    assert subset["manifest"]["experiment_id"] == experiment_id
    with pytest.raises(ValueError, match="different experiment"):
        load_component_fit_subset(path, split, expected_experiment_id="E540")
    with pytest.raises(ValueError, match="different experiment"):
        load_e540_fit_subset(path, split)


def test_e540_prediction_uses_frozen_evaluation_membership(tmp_path: Path) -> None:
    groups = [14, 18]
    manifest = {
        "version": 1,
        "experiment_id": "E540",
        "split_manifest_sha256": "split-sha",
        "evaluation_groups": groups,
        "evaluation_group_count": len(groups),
        "evaluation_membership_sha256": hashlib.sha256(
            np.asarray(groups, dtype="<i8").tobytes()
        ).hexdigest(),
    }
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(manifest))
    checkpoint = {
        "args": {"fit_subset_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "provenance": {"split_manifest_sha256": "split-sha"},
    }

    assert predict_run.load_e540_evaluation_groups(path, checkpoint) == groups

    manifest["evaluation_groups"] = [14, 19]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="training manifest"):
        predict_run.load_e540_evaluation_groups(path, checkpoint)


def test_e540_prediction_reader_does_not_request_targets_or_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[tuple[int, bool, bool]] = []

    class FeatureOnlyReader:
        def __init__(self, path: Path) -> None:
            assert path == tmp_path / "datasets" / "valid.parquet"

        def read(self, group: int, *, targets: bool, mask: bool) -> SimpleNamespace:
            reads.append((group, targets, mask))
            return SimpleNamespace(features=np.zeros((1, 112), dtype=np.float32))

    monkeypatch.setattr(predict_run, "ROOT", tmp_path)
    monkeypatch.setattr(predict_run, "SequenceFile", FeatureOnlyReader)
    dataset = predict_run.FeatureOnlySequences("valid", [14, 18])

    assert dataset[1].shape == (1, 112)
    assert reads == [(18, False, False)]


def test_numeric_map_onnx_parity_must_be_declared_fit_train_group() -> None:
    checkpoint = {
        "args": {"numeric_map": "ple"},
        "provenance": {"fit_row_groups": {"train": [14, 18], "valid": [20]}},
    }

    export_onnx.validate_parity_membership(checkpoint, "train", 18)

    with pytest.raises(ValueError, match="fitting train group"):
        export_onnx.validate_parity_membership(checkpoint, "valid", 20)
    with pytest.raises(ValueError, match="outside checkpoint fitting membership"):
        export_onnx.validate_parity_membership(checkpoint, "train", 20)
