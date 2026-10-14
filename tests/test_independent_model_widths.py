from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from scripts.export_native_weights import write_wbin
from scripts.train_rnn import independent_width_configuration_error, make_model
from wnn.nn import RNNModel
from wnn.optim import muon_param_groups


class _LegacyRNN(nn.Module):
    """Original equal-width construction used to pin default initialization and names."""

    def __init__(self, n_in: int, hidden: int, layers: int, n_out: int = 2) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, n_out)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        z = torch.nn.functional.gelu(self.inp((values - self.mean) / self.std))
        output, _ = self.gru(z)
        return self.head(torch.cat((output, z), dim=-1))


def _state(model: nn.Module) -> dict[str, np.ndarray]:
    return {
        name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        for name, value in model.state_dict().items()
    }


def _weight_parameter_count(state: dict[str, torch.Tensor]) -> int:
    return sum(
        value.numel()
        for name, value in state.items()
        if name == "inp.weight"
        or name.startswith("gru.weight_")
        or name.startswith("head.")
        and name.endswith(".weight")
    )


def _expected_weight_count(
    encoder: int, recurrent: int, readout: int, layers: int, head_layers: int, inputs: int = 211
) -> int:
    count = (
        inputs * encoder
        + 3 * recurrent * (encoder + recurrent)
        + 6 * (layers - 1) * recurrent**2
        + (recurrent + encoder) * readout
        + 2 * readout
    )
    return count + (readout**2 if head_layers == 2 else 0)


def test_equal_width_default_is_bitwise_legacy_compatible() -> None:
    torch.manual_seed(1401)
    legacy = _LegacyRNN(211, 9, 2)
    torch.manual_seed(1401)
    default = RNNModel(211, hidden=9, layers=2)
    torch.manual_seed(1401)
    explicit = RNNModel(
        211, hidden=9, layers=2, encoder_width=9, recurrent_width=9, readout_width=9, head_layers=1
    )

    assert tuple(default.state_dict()) == tuple(legacy.state_dict())
    assert tuple(explicit.state_dict()) == tuple(legacy.state_dict())
    for name, expected in legacy.state_dict().items():
        assert torch.equal(default.state_dict()[name], expected), name
        assert torch.equal(explicit.state_dict()[name], expected), name
    values = torch.arange(3 * 11 * 211, dtype=torch.float32).reshape(3, 11, 211) / 1000
    assert torch.equal(default(values)[0], legacy(values))


def test_trainer_factory_trains_independent_widths_and_matches_parameter_formula() -> None:
    encoder, recurrent, readout, layers, head_layers = 12, 8, 6, 2, 2
    model = make_model(
        {
            "arch": "gru",
            "hidden": 8,
            "layers": layers,
            "dropout": 0.0,
            "encoder_width": encoder,
            "recurrent_width": recurrent,
            "readout_width": readout,
            "head_layers": head_layers,
        },
        211,
        2,
    )
    assert model.inp.weight.shape == (encoder, 211)
    assert model.gru.weight_ih_l0.shape == (3 * recurrent, encoder)
    assert model.gru.weight_ih_l1.shape == (3 * recurrent, recurrent)
    assert model.head[0].weight.shape == (readout, encoder + recurrent)
    assert model.head.extra.weight.shape == (readout, readout)
    assert model.head[2].weight.shape == (2, readout)
    assert _weight_parameter_count(model.state_dict()) == _expected_weight_count(
        encoder, recurrent, readout, layers, head_layers
    )

    torch.manual_seed(1402)
    inputs = torch.randn(2, 17, 211)
    targets = torch.randn(2, 17, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    before = model.head.extra.weight.detach().clone()
    prediction, _ = model(inputs)
    loss = torch.nn.functional.mse_loss(prediction, targets)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, model.head.extra.weight)
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())


@pytest.mark.parametrize(
    "incompatible",
    [
        {"cell": "lstm"},
        {"enc": "mlp"},
        {"numeric_map": "linear"},
        {"readout_activation": "fixed_mix"},
        {"loss": "ce_joint"},
        {"research_features": "volume28"},
        {"sentinel_values": "neutral"},
        {"recurrent_rank": 4},
        {"student_frontier_contract": "frozen-equal-width-contract.json"},
    ],
)
def test_independent_widths_reject_non_native_training_combinations(incompatible) -> None:
    options = {
        "arch": "gru",
        "hidden": 8,
        "layers": 2,
        "dropout": 0.0,
        "encoder_width": 12,
        "recurrent_width": 8,
        "readout_width": 6,
        "head_layers": 2,
        **incompatible,
    }
    error = independent_width_configuration_error(options, n_in=211)
    assert error and "native-compatible" in error
    with pytest.raises(ValueError, match="native-compatible"):
        make_model(options, 211, 2)


def test_independent_widths_reject_non_211_input_export_shape() -> None:
    options = {
        "arch": "gru",
        "hidden": 8,
        "layers": 2,
        "dropout": 0.0,
        "encoder_width": 12,
        "recurrent_width": 8,
        "readout_width": 6,
        "head_layers": 2,
    }
    assert independent_width_configuration_error(options, n_in=212)


def test_muon_hidden_scope_covers_both_readout_hidden_layers() -> None:
    model = RNNModel(
        211,
        hidden=8,
        layers=2,
        encoder_width=12,
        recurrent_width=8,
        readout_width=6,
        head_layers=2,
    )
    groups = muon_param_groups(model, "hidden")
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    assert {names[id(parameter)] for parameter in groups[-1]["params"]} == {
        "head.0.weight",
        "head.extra.weight",
    }


@pytest.mark.parametrize(
    ("widths", "head_layers", "magic", "header_values"),
    [((8, 8, 8), 1, b"WNNWTS01", 5), ((12, 8, 6), 2, b"WNNWTS02", 7)],
)
def test_native_binary_export_versions_shapes_and_keeps_final_output(
    tmp_path, widths, head_layers, magic, header_values
) -> None:
    encoder, recurrent, readout = widths
    model = RNNModel(
        211,
        hidden=8,
        layers=2,
        encoder_width=encoder,
        recurrent_width=recurrent,
        readout_width=readout,
        head_layers=head_layers,
    )
    state = _state(model)
    path = tmp_path / "weights.wbin"
    write_wbin(state, path)
    data = path.read_bytes()
    assert data[:8] == magic
    dimensions = np.frombuffer(data[8 : 8 + 4 * header_values], dtype="<u4")
    if magic == b"WNNWTS01":
        np.testing.assert_array_equal(dimensions, [211, recurrent, 2, readout, 2])
    else:
        np.testing.assert_array_equal(
            dimensions, [211, encoder, recurrent, 2, readout, head_layers, 2]
        )
    matrix_elements = sum(value.size for value in state.values())
    assert len(data) == 8 + 4 * header_values + 4 * matrix_elements
