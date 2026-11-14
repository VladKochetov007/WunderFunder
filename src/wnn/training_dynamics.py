from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Callable

import torch
from torch import nn

from wnn.optim import MuonAdamW, muon_param_groups, wsd_lambda
from wnn.training.tl3_optimizer import TL3MuonAdamW, build_tl3_muon_optimizer
from wnn.training.tl3_scheduler import TL3MuonOneCycle


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add opt-in arms; existing --sched must separately accept 'sgdr'."""
    group = parser.add_argument_group("opt2 dynamics (experimental, default off)")
    group.add_argument("--optim-switch", default="", help="completed step fraction:adamw|muon")
    group.add_argument("--warmup-frac", type=float, default=None)
    group.add_argument("--restarts", type=int, default=1, help="SGDR restart count (K+1 cycles)")
    group.add_argument("--head-lr-mult", type=float, default=1.0)
    group.add_argument("--clip-per-module", default="", help="top-level module:cap,...")
    group.add_argument(
        "--grad-noise", type=float, default=0.0, help="variance eta, not standard deviation"
    )
    group.add_argument(
        "--zoneout", type=float, default=0.0, help="nonzero is unsupported for fused GRU"
    )
    group.add_argument("--init-qmc", action="store_true")
    group.add_argument("--batch-order", choices=["random", "seeded", "qmc"], default="random")
    group.add_argument("--antithetic", action="store_true")
    group.add_argument(
        "--momentum", type=float, default=0.0, help="SGD momentum; Adam/Muon reject nonzero"
    )
    group.add_argument(
        "--no-gradient-clip",
        action="store_true",
        help="do not transform finite gradients; nonfinite gradients abort",
    )


def parse_switch(spec: str, total: int) -> tuple[int, str] | None:
    """Switch before the update after ceil(frac*total) completed attempts."""
    if not spec:
        return None
    try:
        fraction, name = spec.split(":")
        frac = float(fraction)
    except ValueError as exc:
        raise ValueError("--optim-switch must be STEP_FRAC:adamw|muon") from exc
    if total < 2 or not math.isfinite(frac) or not 0 < frac < 1:
        raise ValueError("switch fraction must be finite and strictly inside (0, 1)")
    threshold = math.ceil(frac * total)
    if threshold >= total or name not in {"adamw", "muon"}:
        raise ValueError("switch must leave at least one update and name adamw or muon")
    return threshold, name


def parse_module_caps(spec: str, model: nn.Module) -> dict[str, float]:
    """Validate top-level module clipping caps; unspecified modules use global cap."""
    caps: dict[str, float] = {}
    modules = {name.split(".")[0] for name, _ in model.named_parameters()}
    for item in spec.split(",") if spec else []:
        name, separator, value = item.partition(":")
        if not separator or name not in modules or name in caps:
            raise ValueError(f"invalid or duplicate module clipping entry: {item}")
        cap = float(value)
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("module clipping caps must be positive and finite")
        caps[name] = cap
    return caps


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    """Preserve original default construction; split groups only for an active arm."""
    if not math.isfinite(args.head_lr_mult) or args.head_lr_mult <= 0:
        raise ValueError("head lr multiplier must be positive and finite")
    if args.zoneout != 0:
        raise NotImplementedError(
            "exact zoneout requires a timestep cell path; fused nn.GRU is unsupported"
        )
    momentum = getattr(args, "momentum", 0.0)
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("momentum must be finite and in [0,1)")
    if args.optim == "sgd":
        if args.optim_switch or args.head_lr_mult != 1 or args.fused_adam:
            raise ValueError(
                "bounded SGD requires no switching, no head multiplier and no fused Adam flag"
            )
        return torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            momentum=momentum,
            weight_decay=args.wd,
        )
    if momentum:
        raise ValueError("--momentum requires --optim sgd")
    if args.optim == "tl3-muon":
        if args.optim_switch or args.head_lr_mult != 1:
            raise ValueError(
                "--optim tl3-muon does not support optimizer switching or head LR scaling"
            )
        return build_tl3_muon_optimizer(
            model,
            lr=args.lr,
            muon_lr_mult=args.muon_lr_mult,
            beta2=args.beta2,
            weight_decay=args.wd,
            fused_adam=bool(args.fused_adam),
            strict_fp32=getattr(args, "precision", "fp16") == "fp32",
        )
    common = dict(lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.wd)
    is_grouped = bool(args.optim_switch) or args.head_lr_mult != 1
    if not is_grouped:
        if args.optim == "muon":
            return MuonAdamW(
                muon_param_groups(model, args.muon_scope),
                strict_fp32=getattr(args, "precision", "fp16") == "fp32",
                **common,
            )
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            fused=bool(args.fused_adam),
            **common,
        )
    if args.optim_switch or args.optim == "muon":
        groups = muon_param_groups(model, args.muon_scope)
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    names = {id(p): name for name, p in model.named_parameters()}
    split = []
    for group in groups:
        for is_head in (False, True):
            params = [p for p in group["params"] if names[id(p)].startswith("head.") == is_head]
            if not params:
                continue
            scale = args.head_lr_mult if is_head else 1.0
            split.append(
                {
                    **group,
                    "params": params,
                    "lr": args.lr * scale,
                    "lr_scale": scale,
                    "switch_muon": group.get("muon", False),
                    "muon": group.get("muon", False) and args.optim == "muon",
                }
            )
    if args.optim == "muon":
        return MuonAdamW(split, strict_fp32=getattr(args, "precision", "fp16") == "fp32", **common)
    return torch.optim.AdamW(split, fused=bool(args.fused_adam), **common)


def sgdr_lambda(
    total: int, warmup_frac: float, restarts: int, final: float
) -> Callable[[int], float]:
    """Linear warmup then K+1 equal integer cosine cycles; final step reaches floor."""
    warm = round(total * warmup_frac)
    remaining = total - warm
    cycles = restarts + 1
    if total < 2 or not 0 <= warmup_frac < 1 or restarts < 0:
        raise ValueError("invalid SGDR total, warmup or restart count")
    if remaining < 2 * cycles or not 0 <= final <= 1:
        raise ValueError("SGDR needs at least two updates per cycle and final in [0,1]")
    boundaries = [warm + (i * remaining) // cycles for i in range(cycles + 1)]

    def multiplier(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        if step >= total - 1:
            return final
        cycle = next(i for i in range(cycles) if step < boundaries[i + 1])
        phase = (step - boundaries[cycle]) / (boundaries[cycle + 1] - boundaries[cycle] - 1)
        return final + (1 - final) * (1 + math.cos(math.pi * phase)) / 2

    return multiplier


def build_scheduler(
    opt: torch.optim.Optimizer, args: argparse.Namespace, total: int
) -> torch.optim.lr_scheduler.LRScheduler | TL3MuonOneCycle:
    """Keep the default schedule and beta1 cycle; maintain head/trunk lr ratio."""
    warm = 0.05 if args.warmup_frac is None else args.warmup_frac
    if not math.isfinite(warm) or not 0 <= warm < 1:
        raise ValueError("warmup fraction must be finite and in [0,1)")
    if args.optim == "tl3-muon":
        if args.sched != "onecycle":
            raise ValueError("--optim tl3-muon requires the historical onecycle schedule")
        if not isinstance(opt, TL3MuonAdamW):
            raise TypeError("--optim tl3-muon requires its composite optimizer")
        return TL3MuonOneCycle(
            opt,
            lr=args.lr,
            muon_lr_mult=args.muon_lr_mult,
            total_steps=total,
            warmup_frac=warm,
        )
    if args.sched == "wsd":
        return torch.optim.lr_scheduler.LambdaLR(
            opt,
            wsd_lambda(
                total, warm, args.wsd_decay, args.wsd_final, getattr(args, "wsd_shape", "sqrt")
            ),
        )
    if args.sched == "sgdr":
        return torch.optim.lr_scheduler.LambdaLR(
            opt, sgdr_lambda(total, warm, args.restarts, args.wsd_final)
        )
    max_lr = args.lr
    if args.head_lr_mult != 1:
        max_lr = [args.lr * g.get("lr_scale", 1.0) for g in opt.param_groups]
    return torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=max_lr,
        total_steps=total,
        pct_start=warm,
        final_div_factor=100,
        cycle_momentum=args.optim != "sgd",
    )


def switch_optimizer(
    opt: torch.optim.Optimizer, sched: torch.optim.lr_scheduler.LRScheduler, target: str
) -> tuple[torch.optim.Optimizer, dict[str, int | str]]:
    """Rebind a live schedule without restarting it or changing group ordering.

    Adam-only moments survive (including bias-correction age). Matrices changing
    update rule cold-start: no fabricated Adam variance or incompatible momentum.
    Keep current lr, beta1/beta2, wd and scheduler metadata. Count attempted steps,
    matching the trainer's existing scheduler even when AMP skips an update.
    """
    if target not in {"muon", "adamw"}:
        raise ValueError(target)
    groups = []
    for old in opt.param_groups:
        if "switch_muon" not in old:
            raise ValueError("switch requires build_optimizer with --optim-switch enabled")
        group = dict(old)
        for key in ("fused", "foreach", "capturable", "differentiable", "amsgrad", "maximize"):
            group.pop(key, None)
        group["muon"] = bool(old["switch_muon"]) and target == "muon"
        groups.append(group)
    first = groups[0]
    common = dict(
        lr=first["lr"], betas=first["betas"], eps=first["eps"], weight_decay=first["weight_decay"]
    )
    new = (
        MuonAdamW(groups, **common)
        if target == "muon"
        else torch.optim.AdamW(groups, foreach=False, **common)
    )
    preserved = reset = 0
    for old_group, new_group in zip(opt.param_groups, new.param_groups, strict=True):
        for parameter in old_group["params"]:
            old_state = opt.state.get(parameter, {})
            if not old_state:
                continue
            if bool(old_group.get("muon", False)) != bool(new_group["muon"]):
                reset += 1
                continue
            state = copy.deepcopy(old_state)
            if not new_group["muon"]:
                age = int(state["step"])
                state["step"] = age if target == "muon" else torch.tensor(float(age))
            new.state[parameter] = state
            preserved += 1
    # No scheduler construction here: constructors can overwrite lr and beta1.
    sched.optimizer = new
    new._opt_called = True
    return new, {"target": target, "preserved": preserved, "cold_started": reset}


class Dynamics:
    """Training controls applied after unscale, before clipping and optimizer step."""

    def __init__(self, model: nn.Module, args: argparse.Namespace, total: int) -> None:
        self.model = model
        self.switch = parse_switch(args.optim_switch, total)
        self.caps = parse_module_caps(args.clip_per_module, model)
        self.eta = args.grad_noise
        if not math.isfinite(self.eta) or self.eta < 0:
            raise ValueError("gradient noise eta must be nonnegative and finite")
        self.clip = args.clip
        self.no_clip = getattr(args, "no_gradient_clip", False)
        if self.no_clip and self.caps:
            raise ValueError("--no-gradient-clip cannot be combined with module clipping")
        self.is_switched = False
        self.events: list[dict[str, int | str]] = []
        self.generators: dict[str, torch.Generator] = {}
        self.seed = args.seed + 104729

    def state_dict(self) -> dict[str, object]:
        """Include private noise RNGs and switch status in epoch continuation."""
        return {
            "switch": self.switch,
            "caps": self.caps,
            "eta": self.eta,
            "clip": self.clip,
            "no_clip": self.no_clip,
            "seed": self.seed,
            "is_switched": self.is_switched,
            "events": copy.deepcopy(self.events),
            "generators": {
                name: generator.get_state() for name, generator in self.generators.items()
            },
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore controls after reconstructing the saved optimizer implementation."""
        current = self.state_dict()
        for key in ("switch", "caps", "eta", "clip", "no_clip", "seed"):
            if current[key] != state[key]:
                raise ValueError(f"resume dynamics {key} differs")
        self.is_switched = state["is_switched"]
        self.events = copy.deepcopy(state["events"])
        self.generators = {}
        for name, generator_state in state["generators"].items():
            generator = torch.Generator(device=name)
            generator.set_state(generator_state)
            self.generators[name] = generator

    def maybe_switch(
        self,
        completed: int,
        opt: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler,
    ) -> torch.optim.Optimizer:
        """Switch once, before unscale_ so GradScaler tracks the new optimizer."""
        if self.switch is not None and not self.is_switched and completed >= self.switch[0]:
            opt, event = switch_optimizer(opt, sched, self.switch[1])
            event["completed_attempts"] = completed
            self.events.append(event)
            self.is_switched = True
        return opt

    @torch.no_grad()
    def perturb_and_clip(self, completed: int) -> None:
        """Add private Gaussian noise to unscaled gradients, then bound norms.

        Noise variance is eta/(1+t)^0.55. Module clipping replaces global clipping
        when requested; unlisted top-level modules independently use --clip.
        Zero noise neither draws RNG nor touches gradients before baseline clip.
        """
        if self.eta:
            sigma = math.sqrt(self.eta / (1 + completed) ** 0.55)
            for parameter in self.model.parameters():
                grad = parameter.grad
                if grad is None:
                    continue
                key = str(grad.device)
                if key not in self.generators:
                    self.generators[key] = torch.Generator(device=grad.device).manual_seed(
                        self.seed
                    )
                noise = torch.randn(
                    grad.shape, device=grad.device, dtype=grad.dtype, generator=self.generators[key]
                )
                grad.add_(noise, alpha=sigma)
        if self.no_clip:
            finite = torch.stack(
                [
                    torch.isfinite(p.grad).all()
                    for p in self.model.parameters()
                    if p.grad is not None
                ]
            ).all()
            if not finite:
                raise FloatingPointError(
                    "unclipped update has nonfinite gradients; no selective skip is allowed"
                )
            return
        if not self.caps:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            return
        families: dict[str, list[nn.Parameter]] = {}
        for name, parameter in self.model.named_parameters():
            families.setdefault(name.split(".")[0], []).append(parameter)
        for name, params in families.items():
            torch.nn.utils.clip_grad_norm_(params, self.caps.get(name, self.clip))
