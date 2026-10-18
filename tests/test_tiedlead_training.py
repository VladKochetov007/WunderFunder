from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts import train_rnn as legacy_train_rnn
from wnn.features import event_features_torch
from wnn.training.checkpoints import inference_checkpoint_fields
from wnn.training.cli import build_parser
from wnn.training.environment import InputConfiguration, TrainingEnvironment
from wnn.training.evaluation import predict
from wnn.training.inputs import build_inputs
from wnn.training.models import make_model
from wnn.training.normalization import fit_normalization
from wnn.training.provenance import record_provenance
from wnn.training.source_identity import training_source_identity
from wnn.training.state import TrainingRun as StateTrainingRun
from wnn.training.tiedlead import (
    TIEDLEAD_INPUT_WIDTH,
    TiedLeadRNNModel,
    append_causal_i1_lead,
    tiedlead_configuration_error,
)


def _raw_sequences() -> torch.Tensor:
    raw = torch.zeros((2, 7, 112), dtype=torch.float32)
    raw[0, :, 52:104] = torch.arange(7, dtype=torch.float32)[None, :, None]
    raw[1, :, 52:104] = 100 + torch.arange(7, dtype=torch.float32)[None, :, None]
    raw[0, 2:, 0] = raw[0, 2:, 11] = 1
    raw[0, 5:, 0] = raw[0, 5:, 11] = 2
    raw[1, 3:, 0] = raw[1, 3:, 11] = 1
    return raw


def _arguments() -> dict[str, object]:
    parsed = build_parser().parse_args(
        ["--name", "tiny-tiedlead", "--device", "cpu", "--arch", "tiedlead", "--precision", "fp32"]
    )
    return vars(parsed)


def _model() -> TiedLeadRNNModel:
    return make_model(_arguments(), TIEDLEAD_INPUT_WIDTH, 2)  # type: ignore[return-value]


def test_causal_lead_is_ordered_sequence_local_and_future_independent() -> None:
    raw = _raw_sequences()
    event = event_features_torch(raw)
    lead = append_causal_i1_lead(raw, event)

    assert torch.equal(lead[0, 0], raw[0, 0, 52:104])
    assert torch.equal(lead[0, 1], raw[0, 0, 52:104])
    assert torch.equal(lead[0, 2], raw[0, 2, 52:104])
    assert torch.equal(lead[0, 4], raw[0, 2, 52:104])
    assert torch.equal(lead[0, 5], raw[0, 5, 52:104])
    assert torch.equal(lead[1, 2], raw[1, 0, 52:104])
    assert torch.equal(lead[1, 3], raw[1, 3, 52:104])

    changed_future = raw.clone()
    changed_future[0, 6, 0] = changed_future[0, 6, 11] = 3
    changed_future[0, 6, 52:104] += 500
    changed = append_causal_i1_lead(changed_future, event_features_torch(changed_future))
    assert torch.equal(lead[:, :6], changed[:, :6])

    assembled = build_inputs(
        raw,
        environment=TrainingEnvironment(inputs=InputConfiguration(tiedlead=True)),
    )
    assert assembled.shape == (2, 7, 263)
    assert torch.equal(assembled[..., :211], torch.cat((raw, event), dim=-1))
    assert torch.equal(assembled[..., 211:], lead)


def test_shared_raw_scale_encoder_zero_residuals_and_chunk_reset() -> None:
    torch.manual_seed(17)
    model = _model().eval()
    assert model.inp.in_features == 263
    assert model.inp.out_features == 256
    assert model.gru.input_size == 256
    assert model.gru.hidden_size == 256
    assert model.gru.num_layers == 2
    assert model.tied[0].in_features == 104
    assert model.tied[0].out_features == 128
    assert model.tied[2].in_features == 128
    assert model.tied[2].out_features == 64
    assert torch.count_nonzero(model.tied_out.weight) == 0
    assert torch.count_nonzero(model.tied_out.bias) == 0
    assert torch.count_nonzero(model.lead_out.weight) == 0
    assert torch.count_nonzero(model.lead_out.bias) == 0

    model.mean.copy_(torch.linspace(-0.5, 0.5, 263))
    model.std.copy_(torch.linspace(0.5, 2.0, 263))
    normalized = torch.randn((1, 7, 263))
    reconstructed = normalized * model.std + model.mean
    main_inputs: list[torch.Tensor] = []
    shared_inputs: list[torch.Tensor] = []
    model.inp.register_forward_pre_hook(
        lambda _module, args: main_inputs.append(args[0].detach().clone())
    )
    model.tied.register_forward_pre_hook(
        lambda _module, args: shared_inputs.append(args[0].detach().clone())
    )
    model.encode(normalized)

    assert torch.equal(main_inputs[0], normalized)
    assert len(shared_inputs) == 3
    expected_delta = torch.cat(
        (
            torch.zeros_like(reconstructed[:, :1, :104]),
            reconstructed[:, 1:, :104] - reconstructed[:, :-1, :104],
        ),
        dim=1,
    )
    assert torch.equal(
        shared_inputs[0], torch.cat((reconstructed[..., :52], expected_delta[..., :52]), -1)
    )
    assert torch.equal(
        shared_inputs[1], torch.cat((reconstructed[..., 52:104], expected_delta[..., 52:104]), -1)
    )
    assert torch.equal(
        shared_inputs[2],
        torch.cat(
            (
                reconstructed[..., 52:104],
                reconstructed[..., 52:104] - reconstructed[..., 211:263],
            ),
            -1,
        ),
    )

    with torch.no_grad():
        for parameter in (model.tied_out.weight, model.lead_out.weight):
            torch.nn.init.normal_(parameter, std=0.01)
    branch_outputs: list[torch.Tensor] = []

    def retain_branch(
        _module: torch.nn.Module,
        _args: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        output.retain_grad()
        branch_outputs.append(output)

    model.tied.register_forward_hook(retain_branch)
    with torch.no_grad():
        model.lead_out.weight.normal_(std=0.01)
        model.tied_out.weight.normal_(std=0.01)
    model.encode(normalized).square().sum().backward()
    assert len(branch_outputs) == 3
    assert all(
        output.grad is not None and torch.count_nonzero(output.grad) for output in branch_outputs
    )

    inputs = build_inputs(
        _raw_sequences()[:1],
        environment=TrainingEnvironment(inputs=InputConfiguration(tiedlead=True)),
    )
    model = _model().eval()
    shared_calls: list[torch.Tensor] = []
    recurrent_states: list[torch.Tensor | None] = []
    model.tied.register_forward_pre_hook(
        lambda _module, args: shared_calls.append(args[0].detach().clone())
    )
    model.gru.register_forward_pre_hook(lambda _module, args: recurrent_states.append(args[1]))
    _first, state = model(inputs[:, :4])
    _second, final_state = model(inputs[:, 4:], state)
    assert len(shared_calls) == 6
    assert torch.equal(shared_calls[3][:, 0, 52:], torch.zeros_like(shared_calls[3][:, 0, 52:]))
    assert torch.equal(shared_calls[4][:, 0, 52:], torch.zeros_like(shared_calls[4][:, 0, 52:]))
    assert recurrent_states[1] is state
    assert final_state.shape == (2, 1, 256)
    # The lead was computed for the whole sequence before the chunk split.
    assert torch.equal(inputs[:, 4, 211:], inputs[:, 2, 211:])


def test_fit_normalization_checkpoint_prediction_export_and_provenance(tmp_path) -> None:
    raw = _raw_sequences()
    environment = TrainingEnvironment()
    environment.inputs.tiedlead = True
    model = _model()
    fit_normalization(
        model,
        DataLoader(TensorDataset(raw), batch_size=2),
        torch.device("cpu"),
        batches=1,
        environment=environment,
    )
    expected_inputs = build_inputs(raw, environment=environment).reshape(-1, 263)[::7]
    assert torch.allclose(model.mean, expected_inputs.mean(0))
    assert torch.allclose(model.std, expected_inputs.std(0).clamp_min(0.001))

    exported = inference_checkpoint_fields(model, "raw")
    checkpoint_path = tmp_path / "model.pt"
    torch.save({"args": _arguments(), "n_in": 263, **exported}, checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    restored = make_model(checkpoint["args"], checkpoint["n_in"], 2)
    restored.load_state_dict(checkpoint["model"])
    env = TrainingEnvironment()
    env.inputs.tiedlead = checkpoint["args"]["arch"] == "tiedlead"
    predictions = predict(restored, raw, 4, torch.device("cpu"), environment=env)
    assert predictions.shape == (2, 7, 2)
    assert np.isfinite(predictions).all()
    old_input_flag = legacy_train_rnn.INPUT_TIEDLEAD
    try:
        legacy_train_rnn.INPUT_TIEDLEAD = checkpoint["args"]["arch"] == "tiedlead"
        legacy_predictions = legacy_train_rnn.predict(restored, raw, 4, torch.device("cpu"))
    finally:
        legacy_train_rnn.INPUT_TIEDLEAD = old_input_flag
    assert np.array_equal(legacy_predictions, predictions)

    identity = training_source_identity(environment)
    files = identity["files"]
    expected_sources = {
        "scripts/train_rnn.py",
        "scripts/predict_run.py",
        "scripts/eval_run.py",
        "src/wnn/training/tiedlead.py",
        "src/wnn/training/inputs.py",
        "src/wnn/training/models.py",
        "src/wnn/training/environment.py",
        "src/wnn/training/compatibility.py",
        "src/wnn/data.py",
        "src/wnn/features.py",
        "src/wnn/nn.py",
    }
    assert expected_sources <= set(files)
    assert (
        hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        == identity["sha256"]
    )

    parser = build_parser()
    args = parser.parse_args(
        ["--name", "tiny-tiedlead", "--device", "cpu", "--arch", "tiedlead", "--precision", "fp32"]
    )
    run = StateTrainingRun(environment)
    run.options.args = args
    run.data.train_idx = np.asarray([0, 1], dtype=np.int64)
    run.data.valid_idx = np.asarray([2], dtype=np.int64)
    run.data.train_split = "train"
    record_provenance(run)
    profile = run.data.provenance["tiedlead_profile"]
    assert profile["input_width"] == 263
    assert profile["chunk_semantics"].startswith("delta resets at each encode call")
    assert set(profile["source_sha256"]) == expected_sources


def test_legacy_defaults_and_profile_rejections() -> None:
    parser = build_parser()
    args = parser.parse_args(["--name", "legacy", "--device", "cpu"])
    assert args.arch == "gru"
    assert isinstance(
        make_model({"arch": "gru", "hidden": 8, "layers": 1, "dropout": 0.0}, 211, 2),
        torch.nn.Module,
    )
    legacy_inputs = build_inputs(torch.randn((1, 3, 112)))
    assert legacy_inputs.shape == (1, 3, 211)

    assert tiedlead_configuration_error({"arch": "tiedlead", "precision": "bf16"}) is not None
    with pytest.raises(ValueError, match="exactly 263"):
        make_model(_arguments(), 211, 2)
    with pytest.raises(ValueError, match="unmodified canonical 211-channel"):
        build_inputs(
            torch.randn((1, 3, 113)),
            environment=TrainingEnvironment(inputs=InputConfiguration(tiedlead=True)),
        )
