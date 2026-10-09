from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from wnn.nn import RNNModel  # noqa: E402
from wnn.optim import GradStats, MuonAdamW, muon_param_groups, orthogonalize, rnn_init_, wsd_lambda  # noqa: E402


def _model(cell="gru", hidden=16, n_in=7, n_out=4, seed=0):
    torch.manual_seed(seed)
    return RNNModel(n_in, hidden, 2, n_out=n_out, cell=cell)


def _loss(model, seed=1, b=4, t=30, n_in=7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, t, n_in, generator=g)
    y = torch.cumsum(x[..., :4], dim=1) / math.sqrt(t)  # needs memory
    out, _ = model(x)
    return ((out - y) ** 2).mean()


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (3, 16, 16), (4, 24, 8)])
def test_orthogonalize_matches_polar_factor(shape):
    g = torch.randn(*shape, generator=torch.Generator().manual_seed(2))
    o = orthogonalize(g)
    u, s, vh = torch.linalg.svd(g, full_matrices=False)
    polar = u @ vh
    sv = torch.linalg.svdvals(o)
    assert o.shape == g.shape
    assert sv.min() > 0.6 and sv.max() < 1.3  # quintic NS: approximate, not exact, unit singular values
    cos = (o * polar).sum((-2, -1)) / (o.norm(dim=(-2, -1)) * polar.norm(dim=(-2, -1)))
    assert cos.min() > 0.95


def test_adam_groups_reproduce_torch_adamw_under_onecycle():
    a, b = _model(), _model()
    opt_a = torch.optim.AdamW(a.parameters(), lr=3e-3, weight_decay=0.01, foreach=False)
    opt_b = MuonAdamW([{"params": list(b.parameters())}], lr=3e-3, weight_decay=0.01)
    sa = torch.optim.lr_scheduler.OneCycleLR(opt_a, max_lr=3e-3, total_steps=60, pct_start=0.05, final_div_factor=100)
    sb = torch.optim.lr_scheduler.OneCycleLR(opt_b, max_lr=3e-3, total_steps=60, pct_start=0.05, final_div_factor=100)
    for step in range(12):
        for m, opt, sch in ((a, opt_a, sa), (b, opt_b, sb)):
            opt.zero_grad(set_to_none=True)
            _loss(m, seed=step).backward()
            opt.step()
            sch.step()
    assert opt_b.param_groups[0]["betas"][0] == opt_a.param_groups[0]["betas"][0]  # momentum cycling reaches us
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cell,gates", [("gru", 3), ("lstm", 4)])
def test_param_groups_split(cell, gates):
    m = _model(cell)
    groups = muon_param_groups(m, "rnn")
    names = {id(p): n for n, p in m.named_parameters()}
    muon = [names[id(p)] for p in groups[1]["params"]]
    assert groups[1]["muon"] and groups[1]["blocks"] == gates
    assert sorted(muon) == sorted(n for n, _ in m.named_parameters() if n.startswith("gru.weight_"))
    adam = {names[id(p)] for p in groups[0]["params"]}
    assert {"inp.weight", "head.0.weight", "head.2.weight", "gru.bias_hh_l0"} <= adam
    hidden = muon_param_groups(m, "hidden")
    assert [names[id(p)] for p in hidden[2]["params"]] == ["head.0.weight"] and hidden[2]["blocks"] == 1
    assert sum(len(g["params"]) for g in hidden) == len(list(m.parameters()))
    for p in m.gru.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError):
        muon_param_groups(m)


def test_muon_step_is_scaled_blockwise_polar_factor():
    m = _model(hidden=32)
    opt = MuonAdamW(muon_param_groups(m), lr=1e-2, weight_decay=0.0)
    before = m.gru.weight_hh_l0.detach().clone()
    _loss(m).backward()
    grad = m.gru.weight_hh_l0.grad.detach().clone()
    opt.step()
    delta = (m.gru.weight_hh_l0.detach() - before).reshape(3, 32, 32)
    # first step: Nesterov momentum is 0.19 g, so the step is the scaled per-gate orthogonalised gradient (rows of
    # the stacked r, z, n gates are never mixed); weights are fp32, hence the absolute tolerance
    expected = -1e-2 * 0.2 * math.sqrt(32) * orthogonalize((0.19 * grad).reshape(3, 32, 32))
    torch.testing.assert_close(delta, expected, rtol=1e-3, atol=2e-7)
    other_gate_mix = -1e-2 * 0.2 * math.sqrt(32) * orthogonalize(0.19 * grad).reshape(3, 32, 32)
    assert (delta - other_gate_mix).abs().max() > 100 * (delta - expected).abs().max()
    top = torch.linalg.svdvals(delta / (1e-2 * 0.2 * math.sqrt(32)))[:, 0]
    assert ((top > 0.7) & (top < 1.3)).all()


def test_muon_trains_a_small_gru_with_scaler_and_onecycle():
    m = _model(hidden=24)
    opt = MuonAdamW(muon_param_groups(m, "rnn"), lr=1e-2, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-2, total_steps=150, pct_start=0.05)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    first = _loss(m, seed=99).item()
    for step in range(150):
        opt.zero_grad(set_to_none=True)
        loss = _loss(m, seed=step % 8)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
    assert _loss(m, seed=99).item() < 0.5 * first  # held-out batch; AdamW at the same lr reaches ~0.3x
    assert torch.isfinite(torch.cat([p.flatten() for p in m.parameters()])).all()


def test_wsd_shape():
    f = wsd_lambda(1000, warmup_frac=0.05, decay_frac=0.2, final_frac=0.01)
    assert f(0) == pytest.approx(1 / 50) and f(49) == 1.0 and f(500) == 1.0 and f(800) == 1.0
    tail = [f(s) for s in range(800, 1001)]
    assert all(x >= y for x, y in zip(tail, tail[1:]))
    assert f(1000) == pytest.approx(0.01) and f(900) == pytest.approx(0.01 + 0.99 * (1 - math.sqrt(0.5)))


@pytest.mark.parametrize("cell", ["gru", "lstm"])
def test_chrono_orth_init(cell):
    m = _model(cell, hidden=32)
    ref = {n: p.detach().clone() for n, p in m.named_parameters()}
    rng = torch.get_rng_state()
    rnn_init_(m, "default", 2000, seed=3)
    assert all(torch.equal(ref[n], p) for n, p in m.named_parameters())
    rnn_init_(m, "chrono_orth", 2000, seed=3)
    assert torch.equal(torch.get_rng_state(), rng)  # data order of the paired default arm is untouched
    h = 32
    for k in range(2):
        b_ih, b_hh = getattr(m.gru, f"bias_ih_l{k}"), getattr(m.gru, f"bias_hh_l{k}")
        gate = b_ih[h:2 * h]
        timescale = 1 / (1 - torch.sigmoid(gate.double()))
        assert timescale.min() >= 2 - 1e-6 and timescale.max() <= 2000 + 1e-3 and timescale.median() > 20
        assert (b_hh[h:2 * h] == 0).all()
        if cell == "lstm":
            torch.testing.assert_close(b_ih[:h], -gate)
        else:
            torch.testing.assert_close(b_ih[:h], ref[f"gru.bias_ih_l{k}"][:h])  # reset gate untouched
        w = getattr(m.gru, f"weight_hh_l{k}")
        for j in range(4 if cell == "lstm" else 3):
            block = w[j * h:(j + 1) * h]
            torch.testing.assert_close(block @ block.T, torch.eye(h), atol=1e-5, rtol=0)
    again = _model(cell, hidden=32)
    rnn_init_(again, "chrono_orth", 2000, seed=3)
    assert all(torch.equal(p, q) for p, q in zip(m.parameters(), again.parameters()))
    with pytest.raises(ValueError):
        rnn_init_(m, "xavier", 2000, seed=3)


def test_grad_stats_total_matches_clip_norm():
    m = _model()
    stats = GradStats(m, clip=0.05)
    for seed in range(3):
        m.zero_grad(set_to_none=True)
        _loss(m, seed=seed).backward()
        stats.observe()
        total = torch.nn.utils.clip_grad_norm_(m.parameters(), 0.05)
        torch.testing.assert_close(stats.rows[-1].pow(2).sum().sqrt(), total.float(), rtol=1e-5, atol=1e-7)
    out = stats.finish()
    assert out["steps"] == 3 and out["nonfinite_steps"] == 0 and 0 <= out["clip_rate"] <= 1
    assert set(out["family_median"]) == {"rnn_hh", "rnn_ih", "inp", "head", "bias"}
    assert sum(out["family_sq_share_mean"].values()) == pytest.approx(1.0, abs=1e-3)
    assert stats.rows == [] and stats.finish() == {}
