"""CPU tests of the v7 Discord-trick flags on small synthetic tensors (no real data, no GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
V6_ROOT = SOURCE_ROOT.parent / "h230_source_v6_lead"
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
sys.path.insert(0, str(SOURCE_ROOT / "src"))  # this snapshot's wnn, not the editable root install

import predict_run  # noqa: E402
import train_rnn  # noqa: E402

from wnn.augment import PRICE_VOLUME_CHANNELS, static_rows, warp_index, warp_rows  # noqa: E402
from wnn.data import N_FEATURES, WARMUP  # noqa: E402
from wnn.extractor import (BASE_WIDTH, PROJECTED, NextStepExtractor, extractor_features,  # noqa: E402
                           fit_extractor)
from wnn.input_act import INPUT_ACTS, fit_quantile_table, input_act_features  # noqa: E402
from wnn.pretrain import pretrain_next_rows  # noqa: E402
from wnn.training_sampling import AntiCurriculumSampler  # noqa: E402

CPU = torch.device("cpu")
T = 120


@pytest.fixture(autouse=True)
def reset_globals():
    yield
    train_rnn.INPUT_ACT, train_rnn.INPUT_ACT_TABLE, train_rnn.FROZEN_EXTRACTOR = "none", None, None


def raw_rows(batch: int = 2, rows: int = T, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, rows, N_FEATURES, generator=g)


def tiny_model(head: str = "mlp", n_in: int | None = None):
    n_in = n_in or train_rnn.build_inputs(raw_rows()).shape[-1]
    torch.manual_seed(1)
    model = train_rnn.make_model({"hidden": 16, "layers": 1, "dropout": 0.0, "head": head}, n_in, 2)
    train_rnn.fit_normalization(model, [(raw_rows(),)], CPU, batches=1)
    return model.eval()


def run(model, x):
    with torch.no_grad():
        return model(train_rnn.build_inputs(x), None)[0]


def tiny_table() -> torch.Tensor:
    return torch.from_numpy(fit_quantile_table(raw_rows(8, 100, seed=5).reshape(-1, N_FEATURES).numpy(), 33))


# --- 1 highway head -------------------------------------------------------------------------------------------

def test_highway_head_shape_gate_bias_and_exporter_keys():
    model = tiny_model("highway")
    state = model.state_dict()
    assert state["head.2.weight"].shape[0] == 2  # predict_run reads head.2.weight
    assert state["head.0.weight"].shape == (16, 32)
    gates = [v for k, v in state.items() if k.endswith("gate.bias")]
    assert len(gates) == 2 and all(torch.all(g == -2.0) for g in gates)
    assert run(model, raw_rows()).shape == (2, T, 2)


@pytest.mark.parametrize("head", ["mlp", "highway"])
def test_head_is_causal_and_resets_per_sequence(head):
    model = tiny_model(head)
    x = raw_rows()
    base = run(model, x)
    changed = x.clone()
    changed[:, 70:] += torch.randn_like(changed[:, 70:])
    assert torch.equal(base[:, :70], run(model, changed)[:, :70])  # future rows never move earlier outputs
    alone = run(model, x[1:])  # state reset: a sequence alone equals the same sequence in a batch
    assert torch.allclose(base[1:], alone, atol=1e-6)
    inputs = train_rnn.build_inputs(x)
    first, state = model(inputs[:, :50], None)
    second, _ = model(inputs[:, 50:], state)
    assert torch.allclose(torch.cat([first, second], 1), base, atol=1e-5)  # chunk-carried state == full pass


# --- flag-off parity vs v6 ------------------------------------------------------------------------------------

def test_flags_off_is_bit_identical_to_v6(tmp_path):
    results = []
    for name, root in (("v6", V6_ROOT), ("v7", SOURCE_ROOT)):
        out = tmp_path / f"{name}.json"
        subprocess.run([sys.executable, str(SOURCE_ROOT / "tests" / "parity_fit.py"), str(root), str(out)],
                       check=True, env={"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin"},
                       timeout=300)
        results.append(json.loads(out.read_text()))
    assert results[0]["n_in"] == results[1]["n_in"]
    assert results[0]["losses"] == results[1]["losses"]  # hex floats: exact
    assert results[0]["out"] == results[1]["out"]
    assert results[0]["state"] == results[1]["state"]


# --- 6 input activation ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("act", [a for a in INPUT_ACTS if a != "none"])
def test_input_act_is_causal_rowwise_and_bounded(act):
    table = tiny_table()
    x = raw_rows()
    base = input_act_features(x, table, act)
    assert base.shape == x.shape and torch.isfinite(base).all()
    changed = x.clone()
    changed[:, 60:] = 9.0
    assert torch.equal(base[:, :60], input_act_features(changed, table, act)[:, :60])
    assert torch.equal(base[:, 3:4], input_act_features(x[:, 3:4], table, act))  # pure function of the row


def test_input_act_appends_columns_to_build_inputs_without_touching_others():
    x = raw_rows()
    plain = train_rnn.build_inputs(x)
    train_rnn.INPUT_ACT, train_rnn.INPUT_ACT_TABLE = "sigmoid", tiny_table()
    with_act = train_rnn.build_inputs(x)
    assert with_act.shape[-1] == plain.shape[-1] + N_FEATURES
    assert torch.equal(with_act[..., :plain.shape[-1]], plain)


# --- 3 frozen extractor ---------------------------------------------------------------------------------------

def tiny_extractor() -> NextStepExtractor:
    torch.manual_seed(3)
    return NextStepExtractor().eval().requires_grad_(False)


def test_extractor_features_are_causal_reset_and_chunk_exact():
    ext = tiny_extractor()
    base = train_rnn.build_inputs(raw_rows())[..., :BASE_WIDTH]
    assert BASE_WIDTH == 211
    feats = extractor_features(ext, base)
    assert feats.shape == (2, T, PROJECTED)
    changed = base.clone()
    changed[:, 80:] += 1.0
    assert torch.equal(feats[:, :80], extractor_features(ext, changed)[:, :80])
    assert torch.allclose(feats[1:], extractor_features(ext, base[1:]), atol=1e-6)  # batch-independent reset
    first, _, state = ext(base[:, :40], None)
    second, _, _ = ext(base[:, 40:], state)
    assert torch.allclose(torch.cat([first, second], 1), feats, atol=1e-5)


def test_frozen_extractor_feeds_243_inputs_and_main_model_stays_causal():
    ext = tiny_extractor()
    x = raw_rows()
    plain = train_rnn.build_inputs(x)
    train_rnn.FROZEN_EXTRACTOR = ext
    with_ext = train_rnn.build_inputs(x)
    assert with_ext.shape[-1] == plain.shape[-1] + PROJECTED == 243
    assert torch.equal(with_ext[..., :211], plain[..., :211])
    assert not any(p.requires_grad for p in ext.parameters())
    model = tiny_model(n_in=243)
    base = run(model, x)
    changed = x.clone()
    changed[:, 90:] += torch.randn_like(changed[:, 90:])
    assert torch.equal(base[:, :90], run(model, changed)[:, :90])
    assert torch.allclose(base[1:], run(model, x[1:]), atol=1e-6)


def test_fit_extractor_lowers_next_row_loss_and_freezes():
    base = train_rnn.build_inputs(raw_rows(4, 200, seed=2))[..., :BASE_WIDTH]
    ext = NextStepExtractor()
    ext.mean.copy_(base.reshape(-1, BASE_WIDTH).mean(0))
    ext.std.copy_(base.reshape(-1, BASE_WIDTH).std(0).clamp_min(1e-3))
    losses = fit_extractor(ext, lambda: [base], CPU, epochs=6, chunk=100, lr=3e-3)
    assert losses[-1] < losses[0]
    assert not any(p.requires_grad for p in ext.parameters())


# --- 2 next-row pretraining -----------------------------------------------------------------------------------

def test_pretrain_keeps_encoder_and_gru_resets_head_and_never_sees_labels():
    model = tiny_model()
    model.train()
    head_before = {k: v.clone() for k, v in model.head.state_dict().items()}
    inp_before, gru_before = model.inp.weight.clone(), model.gru.weight_hh_l0.clone()
    seen = []

    def batches():  # only features are supplied; there is no label channel to read
        seen.append(1)
        return iter([(raw_rows(2, 150, seed=9), None)])

    history = pretrain_next_rows(model, batches, train_rnn.build_inputs, CPU, epochs=3, chunk=50, lr=3e-3)
    assert len(history) == 3 and history[-1] < history[0]
    assert not torch.equal(model.inp.weight, inp_before) and not torch.equal(model.gru.weight_hh_l0, gru_before)
    for key, value in model.head.state_dict().items():
        assert torch.equal(value, head_before[key])


def test_pretrain_kept_length_masks_padding():
    model = tiny_model()
    x = raw_rows(2, 150, seed=4)
    x[:, 100:] = x[:, 99:100]  # padded tail that would be trivially predictable
    kept = torch.tensor([100, 100])
    history = pretrain_next_rows(model, lambda: iter([(x, kept)]), train_rnn.build_inputs, CPU, 1, 50)
    assert np.isfinite(history[0])


# --- 4 time warp ----------------------------------------------------------------------------------------------

def static_sequence(n: int = 400):
    g = torch.Generator().manual_seed(11)
    x = torch.zeros(n, N_FEATURES)
    changing = torch.rand(n, generator=g) < 0.3
    level = torch.randn(n, PRICE_VOLUME_CHANNELS, generator=g)
    for t in range(1, n):
        x[t, :PRICE_VOLUME_CHANNELS] = level[t] if changing[t] else x[t - 1, :PRICE_VOLUME_CHANNELS]
    x[:, PRICE_VOLUME_CHANNELS:] = torch.arange(n)[:, None].float()  # row identity in the extra columns
    y = torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 2)
    scored = torch.ones(n, dtype=torch.bool)
    return x, y, scored


@pytest.mark.parametrize("factor", [0.5, 1.0, 1.7, 2.0])
def test_warp_only_repeats_or_drops_static_rows_and_aligns_labels(factor):
    x, y, scored = static_sequence()
    torch.manual_seed(0)
    index, kept = warp_index(x, factor)
    quiet = static_rows(x)
    assert torch.equal(index[:WARMUP], torch.arange(WARMUP))  # warm-up untouched
    live = index[:kept]
    assert torch.all(live[1:] >= live[:-1])  # time order preserved: causal repeat/drop only
    for row in torch.nonzero(~quiet).flatten().tolist():  # every non-static row survives exactly once, in order
        if row < live.max().item():
            assert int((live == row).sum()) == 1
    assert int(len(index)) == len(x)
    warped = warp_rows((x, y, scored), (1.0, factor, factor))
    assert torch.equal(warped[1][:, 0], warped[0][:, PRICE_VOLUME_CHANNELS])  # labels travel with their rows
    assert not warped[2][int(warped[3]):].any() and int(warped[3]) <= len(x)


def test_warp_probability_zero_leaves_sequence_unchanged():
    x, y, scored = static_sequence()
    out = warp_rows((x, y, scored), (1e-9, 0.5, 2.0))
    assert torch.equal(out[0], x) and int(out[3]) == len(x)


def test_warp_does_not_change_earlier_causal_outputs_of_the_warped_stream():
    # a warped view is an ordinary stream: model outputs on it stay causal
    x, y, scored = static_sequence(300)
    warped, *_ = warp_rows((x, y, scored), (1.0, 1.5, 1.5))
    model = tiny_model()
    base = run(model, warped[None])
    changed = warped.clone()
    changed[200:] += 1.0
    assert torch.equal(base[:, :200], run(model, changed[None])[:, :200])


# --- 5 anti curriculum ----------------------------------------------------------------------------------------

def test_anti_curriculum_upweights_hard_half_then_anneals_to_uniform():
    size, epochs = 40, 5
    difficulty = np.arange(size, dtype=np.float64)
    sampler = AntiCurriculumSampler(size, seed=3, epochs=epochs, batch_size=4, difficulty=difficulty)
    sampler.set_epoch(-1)
    assert sorted(sampler) == list(range(size))  # normalization epoch: plain permutation
    sampler.set_epoch(0)
    assert np.isclose(sampler.weights()[size // 2:].mean() / sampler.weights()[:size // 2].mean(), 2.0)
    hard_share = np.mean(np.array(list(sampler)) >= size // 2)
    assert hard_share > 0.6 and list(sampler) == list(sampler)  # deterministic per epoch
    sampler.set_epoch(epochs - 1)
    assert sorted(sampler) == list(range(size)) and np.all(sampler.weights() == 1.0)  # uniform at the end


def test_running_difficulty_is_learned_online():
    sampler = AntiCurriculumSampler(8, seed=0, epochs=4, batch_size=2)
    sampler.set_epoch(0)
    assert sorted(sampler) == list(range(8))  # nothing known yet: uniform epoch
    sampler.update(torch.tensor([0, 1, 2, 3]), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    sampler.set_epoch(1)
    assert sampler.weights()[3] > 1.0 and sampler.weights()[0] == 1.0 and sampler.weights()[7] == 1.0


# --- checkpoint round trip through predict_run ----------------------------------------------------------------

@pytest.mark.parametrize("head,act,extractor", [("highway", "none", False), ("mlp", "tanh", True)])
def test_checkpoint_round_trip_through_predict_run(tmp_path, head, act, extractor):
    table_path, extractor_path = tmp_path / "table.npz", tmp_path / "extractor.pt"
    options = {"head": head, "input_act": act, "input_act_table": str(table_path) if act != "none" else "",
               "frozen_extractor": str(extractor_path) if extractor else ""}
    np.savez(table_path, table=tiny_table().numpy())
    ext = tiny_extractor()
    torch.save({"model": ext.state_dict(), "dims": ext.dims}, extractor_path)
    _, width = train_rnn.configure_new_inputs(options, CPU)
    n_in = 211 + width
    model = tiny_model(head, n_in)
    x = raw_rows()
    expected = train_rnn.predict(model, x, 40, CPU)
    args = {"hidden": 16, "layers": 1, "dropout": 0.0, **options}
    torch.save({"model": model.state_dict(), "args": args, "n_in": n_in, "epoch": 0, "readout": None},
               tmp_path / "model.pt")
    train_rnn.INPUT_ACT, train_rnn.INPUT_ACT_TABLE, train_rnn.FROZEN_EXTRACTOR = "none", None, None  # fresh process
    ck = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=False)
    loaded = predict_run.load_model(ck, CPU)
    predict_run.configure_inputs(ck, CPU)
    assert train_rnn.build_inputs(x).shape[-1] == n_in
    got = train_rnn.predict(loaded, x, 40, CPU)
    assert np.array_equal(expected, got)
