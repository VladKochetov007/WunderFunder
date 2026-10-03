from __future__ import annotations

import argparse
import json

import torch
from sampling import init_qmc_
from training_dynamics import Dynamics, add_arguments, build_optimizer, build_scheduler

from wnn.nn import RNNModel
from wnn.optim import MuonAdamW, muon_param_groups


def options(**overrides: object) -> argparse.Namespace:
    """Construct original trainer defaults plus opt2 defaults for tiny probes."""
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args([])
    vars(args).update(
        optim="adamw",
        muon_scope="rnn",
        lr=0.015,
        beta2=0.999,
        wd=0.01,
        fused_adam=False,
        sched="onecycle",
        wsd_decay=0.2,
        wsd_final=0.01,
        clip=1.0,
        seed=19,
    )
    vars(args).update(overrides)
    return args


def parity(optim: str = "adamw") -> dict[str, object]:
    """Compare loss, updates, schedule and global RNG to original code bitwise."""
    args = options(optim=optim)
    torch.manual_seed(args.seed)
    original = RNNModel(3, 6, 1, n_out=2)
    candidate = RNNModel(3, 6, 1, n_out=2)
    candidate.load_state_dict(original.state_dict())
    if optim == "muon":
        control = MuonAdamW(
            muon_param_groups(original, "rnn"),
            lr=args.lr,
            betas=(0.9, args.beta2),
            weight_decay=args.wd,
        )
    else:
        control = torch.optim.AdamW(
            list(original.parameters()),
            lr=args.lr,
            betas=(0.9, args.beta2),
            weight_decay=args.wd,
            fused=False,
        )
    legacy_sched = torch.optim.lr_scheduler.OneCycleLR(
        control, max_lr=args.lr, total_steps=60, pct_start=0.05, final_div_factor=100
    )
    opt = build_optimizer(candidate, args)
    sched = build_scheduler(opt, args, 60)
    dynamics = Dynamics(candidate, args, 60)
    generator = torch.Generator().manual_seed(43)
    x = torch.randn(3, 8, 3, generator=generator)
    y = x[..., :2].cumsum(1) / 3
    for step in range(12):
        losses = []
        rng_before = torch.get_rng_state().clone()
        for model, optimizer, schedule in (
            (original, control, legacy_sched),
            (candidate, opt, sched),
        ):
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(x)
            loss = (prediction - y).square().mean()
            losses.append(loss.detach())
            loss.backward()
            if model is original:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            else:
                assert dynamics.maybe_switch(step, optimizer, schedule) is optimizer
                dynamics.perturb_and_clip(step)
            optimizer.step()
            schedule.step()
        assert torch.equal(losses[0], losses[1])
        assert torch.equal(rng_before, torch.get_rng_state())
        assert legacy_sched.get_last_lr() == sched.get_last_lr()
        assert control.param_groups[0]["betas"] == opt.param_groups[0]["betas"]
        assert all(
            torch.equal(p, q)
            for p, q in zip(original.parameters(), candidate.parameters(), strict=True)
        )
    return {"optimizer": optim, "updates": 12, "bit_identical": True, "global_rng_unchanged": True}


def fit(initial: str, target: str | None = None, **extra: object) -> dict[str, object]:
    """Fit 24 short synthetic memory sequences; evaluate 8 independent sequences."""
    total = 100
    args = options(optim=initial, optim_switch=f"0.5:{target}" if target else "", **extra)
    torch.manual_seed(args.seed)
    model = RNNModel(3, 8, 1, n_out=2)
    if args.init_qmc:
        init_qmc_(model, args.seed)
    generator = torch.Generator().manual_seed(47)
    x = torch.randn(32, 10, 3, generator=generator)
    y = x[..., :2].cumsum(1) / 3
    opt = build_optimizer(model, args)
    sched = build_scheduler(opt, args, total)
    controls = Dynamics(model, args, total)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    def validation() -> float:
        with torch.no_grad():
            prediction, _ = model(x[24:])
            return float((prediction - y[24:]).square().mean())

    before = validation()
    trace = []
    for step in range(total):
        opt = controls.maybe_switch(step, opt, sched)
        opt.zero_grad(set_to_none=True)
        start = (step % 3) * 8
        prediction, _ = model(x[start : start + 8])
        loss = (prediction - y[start : start + 8]).square().mean()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        controls.perturb_and_clip(step)
        scaler.step(opt)
        scaler.update()
        sched.step()
        trace.append(float(loss.detach()))
    after = validation()
    assert after < before * 0.7, (initial, target, before, after)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert len(controls.events) == int(target is not None)
    return {
        "initial": initial,
        "target": target,
        "init_qmc": args.init_qmc,
        "updates": total,
        "validation_initial": before,
        "validation_final": after,
        "ratio": after / before,
        "events": controls.events,
        "training_loss_first_last": [trace[0], trace[-1]],
    }


def main() -> None:
    """Run bounded parity and optimizer-switch falsifiers with JSON output."""
    torch.set_num_threads(1)
    output = {
        "data": "synthetic only; train 24, validation 8; no campaign sequence IDs",
        "parity": [parity("adamw"), parity("muon")],
        "fits": [
            fit("adamw"),
            fit("muon"),
            fit("muon", "adamw"),
            fit("adamw", "muon"),
            fit("adamw", init_qmc=True),
            fit(
                "muon",
                "adamw",
                sched="sgdr",
                warmup_frac=0.1,
                head_lr_mult=0.5,
                clip_per_module="head:0.5,gru:0.5",
                grad_noise=1e-7,
            ),
        ],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
