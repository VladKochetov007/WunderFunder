"""CPU tests of the v8 architectures (clockbank, dssm, dualtower) on small synthetic tensors (no data, no GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
V7_ROOT = SOURCE_ROOT.parent / "h230_source_v7_discord"
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
sys.path.insert(0, str(SOURCE_ROOT / "src"))  # this snapshot's wnn, not the editable root install

import predict_run  # noqa: E402
import train_rnn  # noqa: E402

from wnn.arch_v8 import CLOCK_LAYOUTS, affine_scan  # noqa: E402
from wnn.data import N_FEATURES  # noqa: E402
from wnn.distillation import append_family_output_head  # noqa: E402
from wnn.nn import weighted_mse_loss  # noqa: E402
from wnn.optim import MuonAdamW, muon_param_groups  # noqa: E402

CPU = torch.device("cpu")
ARCHS = ("clockbank", "dssm", "dualtower")
T = 150
SMALL = {"clockbank": {"hidden": 32, "layers": 2}, "dssm": {"hidden": 24, "layers": 2},
         "dualtower": {"hidden": 16, "layers": 2, "event_hidden": 12, "tower_skip": 8}}
FULL = {"clockbank": {"hidden": 384, "layers": 2}, "dssm": {"hidden": 384, "layers": 2},
        "dualtower": {"hidden": 224, "layers": 2, "event_hidden": 160}}


def held_rows(batch: int = 2, rows: int = T, seed: int = 0) -> torch.Tensor:
    """Raw rows where each column group changes only on some rows, so clocks and events fire irregularly."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, rows, N_FEATURES, generator=g)
    groups = [list(range(a, b)) for a, b in [(0, 22), (22, 44), (44, 52), (52, 74), (74, 96), (96, 104), (104, 112)]]
    for cols in groups:
        keep = torch.rand(batch, rows, generator=g) < 0.6  # hold the previous value on 60% of rows
        keep[:, 0] = False
        for t in range(1, rows):
            x[:, t, cols] = torch.where(keep[:, t, None], x[:, t - 1, cols], x[:, t, cols])
    return x


def make(arch: str, n_out: int = 2, **extra):
    x = held_rows()
    n_in = train_rnn.build_inputs(x).shape[-1]
    torch.manual_seed(3)
    model = train_rnn.make_model({"arch": arch, "dropout": 0.0, **SMALL[arch], **extra}, n_in, n_out)
    train_rnn.fit_normalization(model, [(x,)], CPU, batches=1)
    return model.eval()


def run(model, x, state=None):
    with torch.no_grad():
        return model(train_rnn.build_inputs(x), state)


def test_affine_scan_matches_sequential_loop():
    g = torch.Generator().manual_seed(1)
    a, b, h0 = torch.rand(3, 77, 5, generator=g), torch.randn(3, 77, 5, generator=g), torch.randn(3, 5, generator=g)
    h, ref = h0, []
    for t in range(77):
        h = a[:, t] * h + b[:, t]
        ref.append(h)
    for block in (8, 32):
        assert torch.allclose(affine_scan(a, b, h0, block), torch.stack(ref, 1), atol=1e-5)


@pytest.mark.parametrize("arch", ARCHS)
def test_param_count_and_output_shape(arch):
    full = train_rnn.make_model({"arch": arch, "dropout": 0.0, **FULL[arch]}, 211, 2)
    count = sum(p.numel() for p in full.parameters())
    assert 0.95e6 <= count <= 2.2e6, count
    assert full.state_dict()["head.2.weight"].shape[0] == 2  # predict_run reads head.2.weight
    out, state = run(make(arch, n_out=7), held_rows())
    assert out.shape == (2, T, 7) and torch.isfinite(out).all()
    assert isinstance(state, tuple) and all(torch.is_tensor(s) for s in state)


@pytest.mark.parametrize("arch", ARCHS)
def test_causal_and_state_reset_per_sequence(arch):
    model = make(arch)
    x = held_rows()
    base, _ = run(model, x)
    changed = x.clone()
    changed[:, 90:] = held_rows(2, T - 90, seed=9)  # rewrite the future
    out, _ = run(model, changed)
    assert torch.equal(base[:, :90], out[:, :90])  # nothing before row 90 saw the change
    assert not torch.equal(base[:, 90:], out[:, 90:])
    # reset: a sequence's output does not depend on what ran before it or beside it in the batch
    other = held_rows(1, T, seed=5)
    alone, _ = run(model, x[1:2])
    pair, _ = run(model, torch.cat([other, x[1:2]]))
    assert torch.allclose(alone[0], pair[1], atol=1e-5)
    again, _ = run(model, x[1:2])  # no hidden module state survives a call
    assert torch.equal(alone, again)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("sizes", [(50, 100), (1, 37, 62, 50), (149, 1)])
def test_chunk_state_carry_equals_full_sequence(arch, sizes):
    model = make(arch)
    x = held_rows()
    full, full_state = run(model, x)
    parts, state, start = [], None, 0
    inputs = train_rnn.build_inputs(x)
    for n in sizes:
        with torch.no_grad():
            o, state = train_rnn.forward_chunk(model, inputs, slice(start, start + n), state)
        state = train_rnn.detach_state(state)  # the trainer detaches between chunks
        parts.append(o)
        start += n
    assert start == T
    assert torch.allclose(torch.cat(parts, 1), full, atol=2e-5)
    for a, b in zip(state, full_state):
        assert torch.allclose(a.float(), b.float(), atol=2e-5)


def test_clockbank_banks_freeze_between_their_own_clock_ticks():
    model = make("clockbank")
    x = held_rows(1)
    with torch.no_grad():
        tick, _ = model.tick_flags(x[..., :N_FEATURES], None)
        xn = (train_rnn.build_inputs(x) - model.mean) / model.std
        z = torch.nn.functional.gelu(model.inp(xn))
        layer = model.layers[0]
        h, _ = layer(z, tick, z.new_zeros(1, model.hidden), xn[..., :N_FEATURES])
    assert tick[..., -1].all() and not tick[..., :-1].all()  # base bank always ticks, others only sometimes
    d = np.cumsum([0] + layer.bank_dims)
    moved = h[:, 1:] != h[:, :-1]
    for k in range(tick.shape[-1] - 1):
        idle = ~tick[0, 1:, k]
        assert idle.any()
        assert not moved[0, idle][:, d[k]:d[k + 1]].any()  # exact copy while the bank's clock is shut
    assert moved[0][:, d[-2]:d[-1]].any()


def test_clockbank_layouts_cover_valid_raw_columns():
    for groups in CLOCK_LAYOUTS.values():
        cols = [c for _, cs in groups for c in cs]
        assert len(cols) == len(set(cols)) and min(cols) >= 0 and max(cols) < N_FEATURES
    out, _ = make("clockbank", cb_layout="hint")(train_rnn.build_inputs(held_rows()))
    assert out.shape == (2, T, 2)


def test_dualtower_state_layout_and_sparser_trade_events():
    model = make("dualtower")
    x = held_rows(1)
    changed = torch.cat([torch.zeros(1, 1, N_FEATURES, dtype=torch.bool), x[:, 1:] != x[:, :-1]], 1)
    assert changed[..., model.trade_cols].any(-1).any() and changed[..., model.book_cols].any(-1).any()
    _, state = run(model, x)
    assert len(state) == 2 * 2 + 3  # two layer states per tower, two gaps, last raw row


def test_dualtower_output_ignores_rows_between_events_except_through_skip():
    """With the skip and gap inputs frozen, tower outputs depend only on event rows: perturb a non-event row."""
    model = make("dualtower")
    x = held_rows(1)
    changed = torch.cat([torch.zeros(1, 1, N_FEATURES, dtype=torch.bool), x[:, 1:] != x[:, :-1]], 1)
    quiet = ~(changed[..., model.book_cols].any(-1) | changed[..., model.trade_cols].any(-1))[0]
    assert quiet.any()
    row = int(torch.nonzero(quiet[1:])[0]) + 1  # row 0 never fires; need a non-empty prefix
    # make the quiet row's continuation identical so no event fires there or after; compare held tower states
    inputs = train_rnn.build_inputs(x)
    with torch.no_grad():
        ev_state = model(inputs[:, : row + 1], None)[1]
        ev_state_prev = model(inputs[:, :row], None)[1]
    n = model.n_layers
    for a, b in zip(ev_state[: 2 * n], ev_state_prev[: 2 * n]):
        assert torch.allclose(a, b, atol=1e-6)  # a row with no event in either tower leaves every tower state untouched


@pytest.mark.parametrize("arch", ARCHS)
def test_gradients_flow_and_muon_groups_cover_matrices(arch):
    model = make(arch, n_out=4).train()
    x, y = held_rows(), torch.randn(2, T, 2)
    out, _ = model(train_rnn.build_inputs(x), None)
    loss = weighted_mse_loss(out[..., :2], y, torch.ones(2, T, dtype=torch.bool))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    groups = muon_param_groups(model, "rnn")
    assert any(g.get("muon") for g in groups)
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
    before = [p.detach().clone() for p in model.parameters()]
    MuonAdamW(groups, lr=1e-2).step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


@pytest.mark.parametrize("arch", ARCHS)
def test_family_head_and_aux_columns_keep_working(arch):
    model = append_family_output_head(make(arch, n_out=5))
    out, _ = run(model, held_rows())
    assert out.shape == (2, T, 13)


@pytest.mark.parametrize("arch", ARCHS)
def test_checkpoint_round_trip_through_predict_run(tmp_path, arch):
    model = make(arch)
    x = held_rows()
    expected = train_rnn.predict(model, x, 40, CPU)
    n_in = train_rnn.build_inputs(x).shape[-1]
    args = {"arch": arch, "dropout": 0.0, **SMALL[arch]}
    torch.save({"model": model.state_dict(), "args": args, "n_in": n_in, "epoch": 0, "readout": None},
               tmp_path / "model.pt")
    ck = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    loaded = predict_run.load_model(ck, CPU)
    predict_run.configure_inputs(ck, CPU)
    assert np.array_equal(expected, train_rnn.predict(loaded, x, 40, CPU))


@pytest.mark.parametrize("bad", [{"dropout": 0.1}, {"enc": "mlp"}, {"head": "highway"}, {"rnn_init": "orth"}])
def test_unsupported_combinations_are_rejected(bad):
    with pytest.raises(ValueError):
        train_rnn.make_model({"arch": "dssm", "dropout": 0.0, **SMALL["dssm"], **bad}, 211, 2)


def test_freeze_prefix_is_validated():
    with pytest.raises(ValueError):
        train_rnn.make_model({"arch": "clockbank", "dropout": 0.0, **SMALL["clockbank"], "freeze": "nope"}, 211, 2)
    train_rnn.make_model({"arch": "clockbank", "dropout": 0.0, **SMALL["clockbank"], "freeze": "inp,layers"}, 211, 2)


# --- default arch parity vs v7 --------------------------------------------------------------------------------

def test_default_arch_is_bit_identical_to_v7(tmp_path):
    results = []
    for name, root in (("v7", V7_ROOT), ("v8", SOURCE_ROOT)):
        out = tmp_path / f"{name}.json"
        subprocess.run([sys.executable, str(SOURCE_ROOT / "tests" / "parity_fit.py"), str(root), str(out)],
                       check=True, env={"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin"},
                       timeout=300)
        results.append(json.loads(out.read_text()))
    assert results[0]["n_in"] == results[1]["n_in"]
    assert results[0]["losses"] == results[1]["losses"]  # hex floats: exact
    assert results[0]["out"] == results[1]["out"]
    assert results[0]["state"] == results[1]["state"]
