"""Trainer integration for the raw-scratch block-memory teacher.

These checks construct tiny models and exercise checkpoint reload, TBPTT detach
and chunk parity. They do not read competition data or claim a measured score.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_rnn as training  # noqa: E402

from wnn.block_memory_attention import BlockMemoryAttention, BlockState  # noqa: E402


def _args(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "arch": "block-memory",
        "hidden": 8,
        "layers": 2,
        "dropout": 0.0,
        "cell": "gru",
        "enc": "linear",
        "block_heads": 2,
        "block_rows": 4,
        "freeze": "",
    }
    values.update(overrides)
    return values


def _model(**overrides: object) -> BlockMemoryAttention:
    torch.manual_seed(41)
    torch.set_num_threads(1)
    model = training.make_model(_args(**overrides), 5, 3)
    assert isinstance(model, BlockMemoryAttention)
    with torch.no_grad():
        model.mean.copy_(torch.linspace(-0.4, 0.4, 5))
        model.std.copy_(torch.linspace(0.6, 1.4, 5))
    return model


def _recipe(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "arch": "block-memory",
        "numeric_map": "none",
        "qat": "none",
        "recurrent_rank": 0,
        "privileged_mode": "none",
        "init": "",
        "rnn_init": "default",
        "cell": "gru",
        "enc": "linear",
        "dropout": 0.0,
        "optim": "adamw",
        "optim_switch": "",
        "credit_mode": "deterministic",
        "readout_activation": "gelu",
        "sentinel_indicators": "none",
        "sentinel_values": "unchanged",
        "sentinel_base_width": 0,
        "init_qmc": False,
        "block_heads": 8,
        "block_rows": 256,
        "hidden": 256,
        "layers": 4,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_make_model_reloads_fixed_checkpoint_head_and_normalization() -> None:
    """Output width lives on head.2, and mean/std survive a state-dict reload."""
    model = _model()
    assert isinstance(model.head[0], torch.nn.Linear)
    assert isinstance(model.head[1], torch.nn.GELU)
    assert model.head[2].weight.shape == (3, 8)
    assert model.head[2].bias.shape == (3,)
    state = model.state_dict()
    assert "head.2.weight" in state and "mean" in state and "std" in state
    restored = training.make_model(_args(), 5, 3)
    restored.load_state_dict(state)
    data = torch.randn(2, 9, 5)
    with torch.no_grad():
        expected, _ = model(data)
        actual, _ = restored(data)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(restored.mean, model.mean)
    torch.testing.assert_close(restored.std, model.std)


def test_detach_keeps_partial_blocks_and_cuts_the_backward_graph() -> None:
    """Trainer detach carries BlockState values and stops credit at the boundary."""
    model = _model()
    data = torch.randn(2, 9, 5)
    first, state = training.forward_chunk(model, data, slice(0, 3), None)
    first.retain_grad()
    detached = training.detach_state(state)
    assert isinstance(detached, tuple)
    assert all(isinstance(part, BlockState) for part in detached)
    for part in detached:
        assert part.inputs.shape == (2, 3, 8)
        assert part.memory.shape == (2, 8)
        assert part.output_sum.shape == (2, 8)
        assert not part.memory.requires_grad
        assert not part.inputs.requires_grad
        assert not part.output_sum.requires_grad
    second, completed = training.forward_chunk(model, data, slice(3, 4), detached)
    assert all(part.inputs.shape[1] == 0 for part in completed)
    second.sum().backward()
    assert first.grad is None


def test_detached_chunks_match_the_full_sequence() -> None:
    """Arbitrary chunk edges, including a detach, preserve causal predictions."""
    model = _model().eval()
    data = torch.randn(2, 13, 5)
    with torch.no_grad():
        expected, _ = model(data)
        for chunks in ([3, 6, 4], [1] * 13, [4, 4, 5]):
            state = None
            start = 0
            outputs = []
            for rows in chunks:
                output, state = training.forward_chunk(
                    model, data, slice(start, start + rows), state
                )
                state = training.detach_state(state)
                outputs.append(output)
                start += rows
            torch.testing.assert_close(
                torch.cat(outputs, 1), expected, rtol=1e-5, atol=1e-6
            )


def test_make_model_rejects_incompatible_geometry_and_freeze() -> None:
    with pytest.raises(ValueError, match="divisible"):
        training.make_model(_args(hidden=8, block_heads=3), 5, 2)
    with pytest.raises(ValueError, match="dropout 0"):
        training.make_model(_args(dropout=0.1), 5, 2)
    with pytest.raises(ValueError, match="project,layers"):
        training.make_model(_args(freeze="inp,gru"), 5, 2)


def test_block_memory_defaults_and_clean_recipe() -> None:
    args = _recipe(layers=2)
    training.apply_block_memory_defaults(args, ["--arch", "block-memory"])
    assert args.layers == 4
    explicit = _recipe(layers=3)
    training.apply_block_memory_defaults(explicit, ["--layers", "3"])
    assert explicit.layers == 3
    unchanged = _recipe(arch="gru", layers=2)
    training.apply_block_memory_defaults(unchanged, [])
    assert unchanged.layers == 2
    assert training.block_memory_rejection(_recipe()) is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"numeric_map": "ple"}, "--numeric-map"),
        ({"qat": "recurrent"}, "--qat"),
        ({"recurrent_rank": 4}, "--recurrent-rank"),
        ({"privileged_mode": "future"}, "--privileged-mode"),
        ({"init": "parent"}, "--init"),
        ({"rnn_init": "orth"}, "--rnn-init"),
        ({"cell": "lstm"}, "--cell"),
        ({"enc": "mlp"}, "--enc"),
        ({"dropout": 0.1}, "--dropout"),
        ({"optim": "muon"}, "Muon"),
        ({"optim_switch": "0.5:adamw"}, "--optim-switch"),
        ({"credit_mode": "artbp"}, "--credit-mode"),
        ({"readout_activation": "fixed_mix"}, "--readout-activation"),
        ({"sentinel_indicators": "missing"}, "sentinel preprocessing"),
        ({"init_qmc": True}, "--init-qmc"),
        ({"block_rows": 0}, "non-positive"),
        ({"hidden": 256, "block_heads": 6}, "not divisible"),
    ],
)
def test_block_memory_rejection_names_the_modifier(
    overrides: dict[str, object], message: str
) -> None:
    reason = training.block_memory_rejection(_recipe(**overrides))
    assert reason is not None
    assert message in reason


def test_main_rejects_parent_init_before_any_fit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The CLI names the rejection before the trainer can open a dataset."""
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_rnn.py",
            "--name",
            "h218",
            "--arch",
            "block-memory",
            "--init",
            "parent",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        training.main()
    assert caught.value.code == 2
    assert "--init" in capsys.readouterr().err
    assert not (tmp_path / "datasets").exists()
