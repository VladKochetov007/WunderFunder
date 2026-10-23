from __future__ import annotations

import numpy as np
import pytest
import torch

from wnn.research_features import (
    StreamingResearchFeatures,
    research_features_numpy,
    research_features_torch,
)
from wnn.tiedlead import TIEDLEAD_INPUT_WIDTH, tied_book_deltas
from wnn.training.environment import TrainingEnvironment
from wnn.training.forward import detach_state
from wnn.training.inputs import build_inputs
from wnn.training.models import make_model


def _synthetic_raw() -> np.ndarray:
    """Create two sequences with sparse i0 price events and changing i1 books."""
    raw = np.zeros((2, 8, 112), dtype=np.float32)
    for row in range(raw.shape[1]):
        raw[:, row, 52:104] = np.arange(52, dtype=np.float32) + row * 100
    raw[1, :, 52:104] += 1000
    for row, value in ((2, 1.0), (4, 2.0)):
        raw[0, row, 0] = value
        raw[0, row, 11] = value
    return raw


def _teacher_args(precision: str = "fp32") -> dict[str, object]:
    """Return the fixed clean TL3 teacher settings used by the model factory."""
    return {
        "arch": "gru",
        "cell": "gru",
        "enc": "tiedlead",
        "hidden": 256,
        "layers": 2,
        "dropout": 0.0,
        "precision": precision,
        "numeric_map": "none",
        "readout_activation": "gelu",
        "research_features": "tl3_lead",
        "research_feature_control": "none",
        "privileged_mode": "none",
        "prop_input": "",
        "init": "",
        "rnn_init": "default",
        "qat": "none",
        "activation_backend": "none",
        "raw_only": False,
        "heuristics": False,
        "online_norm": False,
        "init_qmc": False,
        "recurrent_rank": 0,
        "sentinel_base_width": 0,
        "distill_family_weight": 0.0,
        "aux_struct": 0.0,
        "aux_next": 0.0,
        "aux_tier": 0.0,
        "aux_mask": 0.0,
        "aux_sign": 0.0,
        "aux_quote": 0.0,
        "aux": "",
        "sentinel_indicators": "none",
        "sentinel_values": "unchanged",
        "encoder_width": None,
        "recurrent_width": None,
        "readout_width": None,
        "head_layers": 1,
        "distill": "",
        "distill_cache_root": "",
        "ema_teacher": 0.0,
        "td_alpha": 0.0,
    }


def test_tl3_lead_is_causal_sequence_local_and_streaming_equivalent() -> None:
    """The lead snapshot sees only the latest prefix event and resets per sequence."""
    raw = _synthetic_raw()
    tensor = torch.from_numpy(raw)
    feature = research_features_torch(tensor, "tl3_lead")
    expected = research_features_numpy(raw, "tl3_lead")
    assert feature.shape == (2, 8, 52)
    np.testing.assert_array_equal(feature.numpy(), expected)
    np.testing.assert_array_equal(feature[0, 0].numpy(), raw[0, 0, 52:104])
    np.testing.assert_array_equal(feature[0, 1].numpy(), raw[0, 0, 52:104])
    np.testing.assert_array_equal(feature[0, 2].numpy(), raw[0, 2, 52:104])
    np.testing.assert_array_equal(feature[0, 3].numpy(), raw[0, 2, 52:104])
    np.testing.assert_array_equal(feature[1, 0].numpy(), raw[1, 0, 52:104])

    changed_future = tensor.clone()
    changed_future[0, 4:, 52:104] += 500
    torch.testing.assert_close(
        research_features_torch(changed_future, "tl3_lead")[:, :4], feature[:, :4], rtol=0, atol=0
    )
    differentiable_raw = tensor.detach().requires_grad_(True)
    research_features_torch(differentiable_raw, "tl3_lead").square().mean().backward()
    assert differentiable_raw.grad is not None
    assert torch.isfinite(differentiable_raw.grad).all()

    stream = StreamingResearchFeatures("tl3_lead")
    streamed = np.stack([stream.update(row) for row in raw[0]])
    np.testing.assert_array_equal(streamed, expected[0])
    stream.reset()
    np.testing.assert_array_equal(stream.update(raw[1, 0]), expected[1, 0])


def test_tiedlead_chunk_state_preserves_book_deltas_and_backpropagates() -> None:
    """Chunk calls match full-sequence state and retain the canonical parameter shapes."""
    raw = torch.from_numpy(_synthetic_raw())
    environment = TrainingEnvironment()
    environment.inputs.research_features = "tl3_lead"
    inputs = build_inputs(raw, environment=environment)
    model = make_model(_teacher_args(), TIEDLEAD_INPUT_WIDTH, 2)
    model.eval()

    assert inputs.shape == (2, 8, TIEDLEAD_INPUT_WIDTH)
    assert model.tied[0].weight.shape == (128, 104)
    assert model.tied[2].weight.shape == (64, 128)
    assert model.tied_out.weight.shape == (256, 128)
    assert model.lead_out.weight.shape == (256, 64)
    assert model.gru.weight_ih_l0.shape == (3 * 256, 256)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())

    books = inputs[..., :104]
    full_delta, _ = tied_book_deltas(books, None)
    first_delta, carried_book = tied_book_deltas(books[:, :4], None)
    second_delta, final_book = tied_book_deltas(books[:, 4:], carried_book)
    torch.testing.assert_close(torch.cat((first_delta, second_delta), dim=1), full_delta)
    torch.testing.assert_close(final_book, books[:, -1])
    reset_delta, _ = tied_book_deltas(books[:, 4:], None)
    torch.testing.assert_close(reset_delta[:, 0], torch.zeros_like(reset_delta[:, 0]))
    assert not torch.equal(reset_delta[:, 0], full_delta[:, 4])

    with torch.no_grad():
        full, _ = model(inputs)
        first, state = model(inputs[:, :4])
        assert isinstance(state, tuple) and len(state) == 2
        assert state[0].shape == (2, 2, 256)
        assert state[1].shape == (2, 104)
        state = detach_state(state)
        second, state = model(inputs[:, 4:], state)
        torch.testing.assert_close(torch.cat((first, second), dim=1), full, rtol=1e-5, atol=1e-6)
        assert state[0].shape == (2, 2, 256)
        assert state[1].shape == (2, 104)

    differentiable_raw = raw.detach().requires_grad_(True)
    differentiable_inputs = build_inputs(differentiable_raw, environment=environment)
    prediction, _ = model(differentiable_inputs)
    prediction.square().mean().backward()
    assert differentiable_raw.grad is not None
    assert torch.isfinite(differentiable_raw.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_tiedlead_factory_rejects_non_fp32_profile() -> None:
    """Only the fixed strict-FP32 teacher profile can select this encoder."""
    with pytest.raises(ValueError, match="precision"):
        make_model(_teacher_args("fp16"), TIEDLEAD_INPUT_WIDTH, 2)


def test_tiedlead_canonical_cli_builds_fixed_teacher_profile() -> None:
    """Exercise actual CLI defaults and the factory's independent-width admission together."""
    from wnn.training.cli import build_parser

    args = build_parser().parse_args(
        [
            "--name",
            "tl3_cli_admission",
            "--arch",
            "gru",
            "--cell",
            "gru",
            "--enc",
            "tiedlead",
            "--research-features",
            "tl3_lead",
            "--hidden",
            "256",
            "--layers",
            "2",
            "--dropout",
            "0",
            "--precision",
            "fp32",
            "--readout-activation",
            "gelu",
            "--head-layers",
            "1",
        ]
    )
    model = make_model(vars(args), TIEDLEAD_INPUT_WIDTH, 2)
    assert model.inp.in_features == 263
    assert model.gru.hidden_size == 256
    assert model.gru.num_layers == 2
    args.encoder_width = 384
    with pytest.raises(ValueError):
        make_model(vars(args), TIEDLEAD_INPUT_WIDTH, 2)
