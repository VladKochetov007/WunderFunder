from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from wnn.features import StreamingEventFeatures
from wnn.training.sequence_config import load_config
from wnn.training.sequence_epoch_index import publish_epoch_index
from wnn.training.sequence_features import sequence_inputs_from_raw
from wnn.training.sequence_fixed_readout import ReadoutScores, fixed_mix
from wnn.training.sequence_models import build_sequence_model
from wnn.training.sequence_normalization import compute_population_moments
from wnn.training.sequence_predict import load_endpoint
from wnn.training.sequence_provenance import canonical_sha256, file_sha256
from wnn.training.sequence_swa import SWA_SCHEMA, average_states, build_swa

PACKET = Path(__file__).parents[1] / "research/experiment_drafts/generalization-fix-20261108/l4"


def config(arm: str = "raw112_d0", seed: int = 101101) -> dict[str, object]:
    return load_config(PACKET / "configs" / f"{arm}_s{seed}.json")


def test_six_recipes_are_matched_except_declared_treatments() -> None:
    baseline = config()
    for arm, mode, dropout in (
        ("raw112_d0", "raw112", 0),
        ("raw112_d02", "raw112", 0.2),
        ("event211_d02", "event211", 0.2),
    ):
        for seed in (101101, 101102):
            expected = baseline | {
                "seed": seed,
                "model": baseline["model"] | {"input_feature_mode": mode, "dropout": dropout},
            }
            assert config(arm, seed) == expected
    assert baseline["training_mask"] == "all_after_warmup"
    assert baseline["normalization_scope"] == "all_fit" and baseline["epochs"] == 6
    assert baseline["batch_size"] == 32 and baseline["chunk_rows"] == 1000
    assert baseline["checkpoint_retention"] == "all_epochs"


def test_event211_full_sequence_streaming_chunk_boundaries_and_future_causality() -> None:
    raw = np.random.default_rng(111).normal(size=(1103, 112)).astype(np.float32)
    full = sequence_inputs_from_raw(raw, "event211")
    streaming = StreamingEventFeatures()
    pieces = []
    for left, right in ((0, 997), (997, 1000), (1000, 1001), (1001, 1103)):
        pieces.extend(
            np.concatenate((row, streaming.update(row).copy())) for row in raw[left:right]
        )
    np.testing.assert_allclose(np.stack(pieces), full, atol=1e-6, rtol=1e-6)
    changed = raw.copy()
    changed[1001:] += 15
    np.testing.assert_array_equal(sequence_inputs_from_raw(changed, "event211")[:1001], full[:1001])
    torch_values = sequence_inputs_from_raw(torch.from_numpy(raw), "event211")
    np.testing.assert_allclose(torch_values.numpy(), full, atol=1e-6, rtol=1e-6)
    streaming.reset()
    np.testing.assert_allclose(
        np.concatenate((raw[0], streaming.update(raw[0]))), full[0], atol=1e-6
    )


def test_event_gru_eval_causality_chunk_state_reset_and_dropout_policy() -> None:
    settings = config("event211_d02")["model"]
    model = build_sequence_model(settings).eval()
    assert model.gru.dropout == 0.2 and model.mean.numel() == 211
    raw = torch.randn(1, 1007, 112)
    values = sequence_inputs_from_raw(raw, "event211")
    with torch.no_grad():
        full, _, _ = model(values)
        first, _, state = model(values[:, :1000])
        second, _, _ = model(values[:, 1000:], state)
        perturbed = raw.clone()
        perturbed[:, 1001:] += 5
        future, _, _ = model(sequence_inputs_from_raw(perturbed, "event211"))
        reset, _, _ = model(values)
    torch.testing.assert_close(torch.cat((first, second), 1), full, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(future[:, :1001], full[:, :1001], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(reset, full, atol=0, rtol=0)


def test_raw_dropout_arms_couple_initial_parameters() -> None:
    torch.manual_seed(101101)
    control = build_sequence_model(config()["model"])
    torch.manual_seed(101101)
    treatment = build_sequence_model(config("raw112_d02")["model"])
    assert control.state_dict().keys() == treatment.state_dict().keys()
    assert all(
        torch.equal(control.state_dict()[key], value)
        for key, value in treatment.state_dict().items()
    )


def test_event_moments_read_only_fitting_members() -> None:
    members = [("train", 0), ("train", 2)]
    rng = np.random.default_rng(12)
    features = {group: rng.normal(size=(11, 112)).astype(np.float32) for _, group in members}
    read = []

    def reader(source: str, group: int) -> np.ndarray:
        assert source == "train" and group in features
        read.append((source, group))
        return sequence_inputs_from_raw(features[group], "event211")

    mean, std = compute_population_moments(members, reader, feature_count=211)
    assert read == members and mean.shape == std.shape == (211,)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()


def test_swa_averages_parameters_and_refuses_buffer_drift() -> None:
    states = [
        {"p": torch.tensor([float(i)]), "mean": torch.tensor([4.0]), "counter": torch.tensor(3)}
        for i in (1, 2, 3, 4)
    ]
    result = average_states(states, {"p"})
    assert result["p"].item() == 2.5 and result["mean"].item() == 4
    assert result["counter"].item() == 3
    states[-1]["mean"][0] += 1
    with pytest.raises(ValueError, match="buffer must be identical"):
        average_states(states, {"p"})


def epoch_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, object]]:
    if os.name == "nt":
        monkeypatch.setattr("wnn.training.sequence_epoch_index_io.fsync_directory", lambda _: None)
    cfg = config()
    data = {"fit_members": [["train", 0]], "teacher": None}
    files = {"fixture.py": "a" * 64}
    identity = {
        "config_sha256": canonical_sha256(cfg),
        "data": data,
        "data_sha256": canonical_sha256(data),
        "source_files": files,
        "source_sha256": canonical_sha256(files),
    }
    model = build_sequence_model(cfg["model"])
    folder = root / "epoch_checkpoints"
    folder.mkdir()
    previous = None
    for epoch in range(1, 7):
        state = model.state_dict()
        for name in dict(model.named_parameters()):
            state[name] = torch.full_like(state[name], epoch / 10)
        payload = {
            "checkpoint_schema": "wnn.sequence-epoch.v1",
            "config": cfg,
            "data": data,
            "identity": identity,
            "model": state,
            "epoch": epoch - 1,
            "completed_epochs": epoch,
            "global_step": 10 * epoch,
            "provenance": {"identity": identity, "normalization": {"fixed": True}},
        }
        temporary = folder / "temporary.pt"
        torch.save(payload, temporary)
        sha = file_sha256(temporary)
        path = folder / f"epoch-{epoch:06d}-{sha}.pt"
        temporary.rename(path)
        previous = publish_epoch_index(
            root,
            identity,
            epoch - 1,
            epoch,
            10 * epoch,
            {
                "epoch": epoch - 1,
                "completed_epochs": epoch,
                "global_step": 10 * epoch,
                "checkpoint_path": f"epoch_checkpoints/{path.name}",
                "checkpoint_sha256": sha,
            },
            previous,
        )
    return root / "epoch_index.json", previous


def test_swa_parent_hashes_exact_fixed_epochs_and_canonical_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path, index = epoch_fixture(tmp_path, monkeypatch)
    target = tmp_path / "swa.pt"
    receipt = build_swa(index_path, index["index_sha256"], target)
    assert [parent["completed_epochs"] for parent in receipt["parents"]] == [3, 4, 5, 6]
    model, payload = load_endpoint(target, torch.device("cpu"))
    assert payload["checkpoint_schema"] == SWA_SCHEMA
    assert not {"optimizer", "scheduler", "rank_rng_states"} & payload.keys()
    for _name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter, torch.full_like(parameter, 0.45), rtol=1e-6, atol=1e-6
        )
    assert torch.equal(model.mean, torch.zeros_like(model.mean))
    assert torch.equal(model.std, torch.ones_like(model.std))


def test_swa_missing_parent_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_path, index = epoch_fixture(tmp_path, monkeypatch)
    (tmp_path / index["entries"][2]["checkpoint_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        build_swa(index_path, index["index_sha256"], tmp_path / "swa.pt")


def test_fixed_mixture_composes_before_official_clipping() -> None:
    candidate = np.array([[-8, 4], [2, 1], [8, -3]], dtype=np.float32)
    reference = np.array([[4, 1], [1, 2], [-4, -1]], dtype=np.float32)
    expected = np.float32(0.9) * reference + np.float32(0.1) * candidate
    np.testing.assert_array_equal(fixed_mix(reference, candidate), expected)
    assert expected[0, 0] > 2
    scores = ReadoutScores()
    scores.add(
        np.array([[1, 1], [2, -1], [-2, 2]], dtype=np.float32),
        candidate,
        reference,
        np.ones(3, bool),
    )
    result = scores.result()
    assert result["candidate"]["selected_rows"] == result["mixture"]["selected_rows"] == 3
    assert np.isfinite(result["mixture"]["wmse"])


def test_readout_excluded_nonfinite_rows_do_not_poison_wmse() -> None:
    targets = np.array([[np.nan, np.nan], [1, -1], [-1, 2], [2, 1]], dtype=np.float32)
    candidate = np.array([[np.inf, np.nan], [0.3, -0.5], [-0.1, 1.1], [1.3, 0.5]], dtype=np.float32)
    reference = np.array([[np.nan, np.inf], [0.8, -0.2], [-0.7, 1.6], [1.5, 0.7]], dtype=np.float32)
    masked = ReadoutScores()
    masked.add(targets, candidate, reference, np.array([False, True, True, True]))
    selected = ReadoutScores()
    selected.add(targets[1:], candidate[1:], reference[1:], np.ones(3, bool))
    assert masked.result() == selected.result()
