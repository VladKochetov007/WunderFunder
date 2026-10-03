from __future__ import annotations

import ast
import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research/campaign_2026-10-03/opt2"))


def test_antithetic_normalization_confound() -> None:
    """Run the live normalization body on identity-feature fixture inputs."""
    from sampling import EpochOrderSampler

    from wnn.nn import RNNModel

    tree = ast.parse((ROOT / "scripts/train_rnn.py").read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fit_normalization"
    )
    namespace = {"torch": torch, "build_inputs": lambda x: x}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "live_normalization", "exec"), namespace)
    data = TensorDataset(torch.arange(64.0).reshape(64, 1, 1).expand(-1, 7, 1))
    models = [RNNModel(1, 4, 1), RNNModel(1, 4, 1)]
    for model, reverse in zip(models, (False, True), strict=True):
        sampler = EpochOrderSampler(64, 3, reverse)
        sampler.set_epoch(-1)
        loader = DataLoader(data, batch_size=8, sampler=sampler, drop_last=True)
        namespace["fit_normalization"](model, loader, torch.device("cpu"))
    print(
        "normalization control/antithetic:",
        [(float(m.mean.item()), float(m.std.item())) for m in models],
    )
    assert not torch.equal(models[0].mean, models[1].mean)
    assert not torch.equal(models[0].std, models[1].std)


def test_switch_boundary_skip_frozen_groups_and_noop() -> None:
    """Exercise all switch directions at a skipped boundary, with cold states."""
    from probe import options
    from training_dynamics import Dynamics, build_optimizer, build_scheduler

    from wnn.nn import RNNModel

    for schedule in ("onecycle", "wsd", "sgdr"):
        for initial, target in (
            ("adamw", "adamw"),
            ("muon", "muon"),
            ("adamw", "muon"),
            ("muon", "adamw"),
        ):
            torch.manual_seed(91)
            model = RNNModel(3, 4, 1)
            for name, p in model.named_parameters():
                if name.startswith("inp.") or name == "gru.bias_hh_l0":
                    p.requires_grad_(False)
            frozen = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
            args = options(
                optim=initial,
                optim_switch=f"0.5:{target}",
                muon_scope="hidden",
                head_lr_mult=0.25,
                sched=schedule,
                warmup_frac=0.1,
            )
            opt = build_optimizer(model, args)
            sched = build_scheduler(opt, args, 20)
            dynamics = Dynamics(model, args, 20)
            control_model = copy.deepcopy(model)
            control_opt = build_optimizer(control_model, args)
            control_sched = build_scheduler(control_opt, args, 20)
            scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
            for attempt in range(20):
                if attempt == 10:
                    group_ids = [[id(p) for p in g["params"]] for g in opt.param_groups]
                    opt = dynamics.maybe_switch(attempt, opt, sched)
                    assert group_ids == [[id(p) for p in g["params"]] for g in opt.param_groups]
                before = [p.clone() for p in model.parameters()]
                opt.zero_grad(set_to_none=True)
                scaler.scale(
                    sum(p.square().sum() for p in model.parameters() if p.requires_grad)
                ).backward()
                if attempt == 10:
                    model.head[0].weight.grad.fill_(float("inf"))
                scaler.unscale_(opt)
                dynamics.perturb_and_clip(attempt)
                scaler.step(opt)
                scaler.update()
                sched.step()
                if attempt == 10:
                    assert all(
                        torch.equal(p, b) for p, b in zip(model.parameters(), before, strict=True)
                    )
                if initial == target:
                    control_opt.zero_grad(set_to_none=True)
                    if attempt != 10:
                        sum(
                            p.square().sum() for p in control_model.parameters() if p.requires_grad
                        ).backward()
                        torch.nn.utils.clip_grad_norm_(control_model.parameters(), args.clip)
                        control_opt.step()
                    control_sched.step()
                    assert all(
                        torch.equal(p, q)
                        for p, q in zip(model.parameters(), control_model.parameters(), strict=True)
                    )
                for name, p in model.named_parameters():
                    if name in frozen:
                        assert torch.equal(p, frozen[name]) and p not in opt.state
                assert all(torch.isfinite(p).all() for p in model.parameters())
            assert sched.last_epoch == 20
            assert dynamics.events[0]["completed_attempts"] == 10
            print(schedule, initial, target, dynamics.events)
