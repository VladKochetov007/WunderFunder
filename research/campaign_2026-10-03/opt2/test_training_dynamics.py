from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from probe import fit, options, parity
from sampling import EpochOrderSampler, init_qmc_, input_strata, stratified_order
from torch import nn
from training_dynamics import Dynamics, build_optimizer, build_scheduler, parse_switch, sgdr_lambda

from wnn.nn import RNNModel


@pytest.mark.parametrize("optimizer", ["adamw", "muon"])
def test_default_off_bit_parity(optimizer: str) -> None:
    assert parity(optimizer)["bit_identical"]


@pytest.mark.parametrize("initial,target", [("adamw", "muon"), ("muon", "adamw")])
def test_switch_memory_fit(initial: str, target: str) -> None:
    result = fit(initial, target)
    assert result["ratio"] < 0.7


@pytest.mark.parametrize("initial,target", [("adamw", "muon"), ("muon", "adamw")])
def test_switch_state_and_schedule(initial: str, target: str) -> None:
    model = RNNModel(3, 4, 1)
    args = options(optim=initial, optim_switch=f"0.5:{target}", head_lr_mult=0.5)
    opt = build_optimizer(model, args)
    sched = build_scheduler(opt, args, 60)
    dynamics = Dynamics(model, args, 60)
    for _ in range(3):
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 0.02)
        opt.step()
        sched.step()
    state = {
        p: {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in opt.state[p].items()
        }
        for p in model.parameters()
    }
    lr = sched.get_last_lr()
    age = sched.last_epoch
    beta = [g["betas"] for g in opt.param_groups]
    assert dynamics.maybe_switch(29, opt, sched) is opt
    new = dynamics.maybe_switch(30, opt, sched)
    assert dynamics.maybe_switch(31, new, sched) is new
    assert sched.optimizer is new and sched.last_epoch == age
    assert sched.get_last_lr() == lr == [g["lr"] for g in new.param_groups]
    assert [g["betas"] for g in new.param_groups] == beta
    recurrent = set(model.gru.parameters())
    for p in model.parameters():
        if p in recurrent and p.ndim == 2:
            assert not new.state[p]
        else:
            assert int(new.state[p]["step"]) == 3
            assert torch.equal(new.state[p]["exp_avg"], state[p]["exp_avg"])
            assert torch.equal(new.state[p]["exp_avg_sq"], state[p]["exp_avg_sq"])
    new.step()
    sched.step()
    assert sched.last_epoch == age + 1
    scales = [g["lr_scale"] for g in new.param_groups]
    trunk = next(g["lr"] for g in new.param_groups if g["lr_scale"] == 1)
    assert all(
        g["lr"] == pytest.approx(trunk * scale)
        for g, scale in zip(new.param_groups, scales, strict=True)
    )


@pytest.mark.parametrize("spec", ["bad", "0:adamw", "1:muon", "nan:muon", "0.5:sgd", "0.99:adamw"])
def test_bad_switch_rejected(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_switch(spec, 10)


def test_sgdr_endpoints_and_restarts() -> None:
    schedule = sgdr_lambda(22, 2 / 22, 1, 0.01)
    assert schedule(0) == 0.5 and schedule(1) == 1
    assert schedule(2) == 1 and schedule(11) == pytest.approx(0.01)
    assert schedule(12) == 1 and schedule(21) == 0.01
    assert all(schedule(i) >= schedule(i + 1) for i in range(2, 11))
    assert schedule(22) == 0.01
    with pytest.raises(ValueError):
        sgdr_lambda(4, 0.5, 2, 0.01)


@pytest.mark.parametrize("kind", ["onecycle", "wsd", "sgdr"])
def test_head_scale_across_schedule(kind: str) -> None:
    model = RNNModel(3, 4, 1)
    args = options(sched=kind, warmup_frac=0.1, head_lr_mult=0.25)
    opt = build_optimizer(model, args)
    sched = build_scheduler(opt, args, 60)
    for _ in range(60):
        trunk = opt.param_groups[0]["lr"]
        assert opt.param_groups[1]["lr"] == pytest.approx(0.25 * trunk)
        opt.step()
        sched.step()


def test_per_module_clip_preserves_small_trunk_gradient() -> None:
    model = RNNModel(3, 4, 1)
    for name, p in model.named_parameters():
        p.grad = torch.full_like(p, 20 if name.startswith("head.") else 1e-4)
    before = model.inp.weight.grad.clone()
    dynamics = Dynamics(model, options(clip_per_module="head:0.2,gru:0.3"), 60)
    dynamics.perturb_and_clip(0)
    assert torch.equal(before, model.inp.weight.grad)
    norm = torch.cat([p.grad.flatten() for p in model.head.parameters()]).norm()
    assert norm <= 0.200001
    with pytest.raises(ValueError):
        Dynamics(model, options(clip_per_module="missing:1"), 60)


def test_gradient_noise_variance_decay_and_private_rng() -> None:
    model = nn.Linear(256, 32, bias=False)
    args = options(grad_noise=0.02, clip=1e6)
    first = Dynamics(model, args, 200)
    late = Dynamics(model, args, 200)
    rng = torch.get_rng_state().clone()
    model.weight.grad = torch.zeros_like(model.weight)
    first.perturb_and_clip(0)
    early = model.weight.grad.clone()
    model.weight.grad.zero_()
    late.perturb_and_clip(99)
    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(model.weight.grad, early / 100**0.275)
    assert float(early.var()) == pytest.approx(0.02, rel=0.05)


def test_zoneout_fails_explicitly() -> None:
    with pytest.raises(NotImplementedError, match="timestep"):
        build_optimizer(RNNModel(3, 4, 1), options(zoneout=0.1))


def test_antithetic_epoch_order_and_rng() -> None:
    control = EpochOrderSampler(25, 3)
    opposite = EpochOrderSampler(25, 3, True)
    rng = torch.get_rng_state().clone()
    for epoch in (-1, 0, 1):
        control.set_epoch(epoch)
        opposite.set_epoch(epoch)
        expected = list(opposite) if epoch < 0 else list(opposite)[::-1]
        assert list(control) == expected
        assert sorted(control) == list(range(25))
    control.set_epoch(0)
    old = list(control)
    control.set_epoch(1)
    assert old != list(control)
    assert torch.equal(rng, torch.get_rng_state())


def test_input_strata_ties_and_no_target_arguments() -> None:
    strata = input_strata(np.repeat([0, 1], 6), np.tile(np.arange(6), 2))
    assert np.array_equal(np.bincount(strata), np.full(6, 2))
    tied = input_strata(np.zeros(6), np.ones(6))
    assert len(np.unique(tied)) == 1
    with pytest.raises(ValueError):
        input_strata(np.zeros(2), np.array([1.0, np.nan]))


@pytest.mark.parametrize("counts,batch", [([16, 16, 16, 16], 8), ([1, 3, 13, 8], 7), ([5], 3)])
def test_qmc_order_coverage_quotas(counts: list[int], batch: int) -> None:
    strata = np.repeat(np.arange(len(counts)), counts)
    order = stratified_order(strata, batch, 4)
    assert sorted(order) == list(range(sum(counts)))
    assert order == stratified_order(strata, batch, 4)
    if len(set(counts)) == 1 and batch % len(counts) == 0:
        for start in range(0, len(order) - batch + 1, batch):
            assert np.array_equal(
                np.bincount(strata[order[start : start + batch]], minlength=len(counts)),
                np.full(len(counts), batch // len(counts)),
            )


def test_stratification_falsifier() -> None:
    strata = np.repeat(np.arange(4), 16)
    gradient = strata.astype(float) * 10
    balanced = stratified_order(strata, 8, 2)
    shuffled = np.random.default_rng(2).permutation(len(strata))
    qmc_variance = gradient[balanced].reshape(-1, 8).mean(1).var()
    random_variance = gradient[shuffled].reshape(-1, 8).mean(1).var()
    assert qmc_variance == 0 and random_variance > 0


def test_qmc_init_marginal_bounds_reproducible_rng() -> None:
    model = RNNModel(3, 8, 1)
    other = RNNModel(3, 8, 1)
    rng = torch.get_rng_state().clone()
    init_qmc_(model, 4)
    init_qmc_(other, 4)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(
        torch.equal(p, q) for p, q in zip(model.parameters(), other.parameters(), strict=True)
    )
    for name, p in model.named_parameters():
        fan = 3 if name.startswith("inp.") else 16 if name.startswith("head.0") else 8
        bound = 1 / math.sqrt(fan)
        assert float(p.detach().abs().max()) <= bound
    large = nn.Linear(32, 128, bias=False)
    init_qmc_(large, 3)
    assert abs(float(large.weight.detach().mean())) < 1e-4
    assert float(large.weight.detach().var()) == pytest.approx(1 / (3 * 32), rel=0.01)


def test_qmc_init_validates_before_mutation() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(ValueError, match="unsupported"):
        init_qmc_(model, 3)
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before, strict=True))


@pytest.mark.parametrize("schedule", ["wsd", "sgdr"])
def test_combined_controls_fit(schedule: str) -> None:
    result = fit(
        "muon",
        "adamw",
        sched=schedule,
        warmup_frac=0.1,
        head_lr_mult=0.5,
        clip_per_module="head:0.5,gru:0.5",
        grad_noise=1e-7,
    )
    assert result["ratio"] < 0.7


def test_cpu_amp_skip_then_switch_uses_attempt_clock() -> None:
    model = RNNModel(3, 4, 1)
    args = options(optim_switch="0.5:muon")
    opt = build_optimizer(model, args)
    sched = build_scheduler(opt, args, 60)
    dynamics = Dynamics(model, args, 60)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    before = [p.detach().clone() for p in model.parameters()]
    opt.zero_grad(set_to_none=True)
    scaler.scale(sum(p.sum() for p in model.parameters())).backward()
    model.inp.weight.grad.fill_(float("inf"))
    scaler.unscale_(opt)
    dynamics.perturb_and_clip(29)
    scaler.step(opt)
    scaler.update()
    with pytest.warns(UserWarning, match="before.*optimizer.step"):
        sched.step()  # legacy trainer advances even on skipped updates
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before, strict=True))
    opt = dynamics.maybe_switch(30, opt, sched)
    opt.zero_grad(set_to_none=True)
    scaler.scale(sum(p.square().sum() for p in model.parameters())).backward()
    scaler.unscale_(opt)
    dynamics.perturb_and_clip(30)
    scaler.step(opt)
    scaler.update()
    sched.step()
    assert sched.last_epoch == 2
    assert dynamics.events[0]["completed_attempts"] == 30
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_qmc_initialization_trains_tiny_memory_model() -> None:
    result = fit("adamw", init_qmc=True)
    assert result["ratio"] < 0.7
