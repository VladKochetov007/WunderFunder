"""Bounded synthetic capture interface; never fills gaps from evaluation exports."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

VERSION = 1
REQUIRED = {
    "version",
    "evidence_kind",
    "phase",
    "student",
    "ema",
    "h",
    "h_ema",
    "optimizer",
    "parameter_names",
    "optimizer_class",
    "batch",
    "loss_config",
    "clip",
    "clock",
    "rng",
    "execution",
    "source_identity",
    "ema_clock",
}
CONFIG = {
    "main": 1.0,
    "ema": 1.0,
    "struct": 0.3,
    "next": 0.3,
    "alpha": 0.5,
    "bootstrap_h": 8,
    "ema_metric_weight": False,
}


class TinyGRU(nn.Module):
    """A shared recurrent trunk with genuinely distinct loss heads."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.GRU(3, 5, batch_first=True)
        self.head = nn.Linear(5, 13)

    def forward(self, x: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        z, hidden = self.trunk(x, hidden)
        return self.head(z), hidden


def detached_copy(value: Any) -> Any:
    """Own every tensor, including optimizer moments, rather than aliasing state."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: detached_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [detached_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(detached_copy(v) for v in value)
    return copy.deepcopy(value)


def validate_capture(capture: dict[str, Any]) -> None:
    """Admit only the explicit bounded backend implemented here, fail closed."""
    missing = REQUIRED - capture.keys()
    if missing:
        raise ValueError(f"missing capture inputs: {sorted(missing)}")
    if capture["version"] != VERSION or capture["evidence_kind"] != "synthetic":
        raise ValueError("this adapter admits synthetic v1 only; real backend unimplemented")
    if capture["phase"] != "pre_forward_pre_step":
        raise ValueError("wrong capture phase")
    if capture["execution"] != {
        "dtype": "float32",
        "device": "cpu",
        "autocast": False,
        "scaler": None,
        "scheduler": None,
    }:
        raise ValueError("unsupported execution state")
    if capture["optimizer_class"] != "torch.optim.AdamW.cpu.unfused":
        raise ValueError("unsupported optimizer backend")
    if capture["loss_config"] != CONFIG or capture["clip"] != {"max_norm": 0.12, "norm_type": 2.0}:
        raise ValueError("unsupported loss/clip configuration")
    with torch.random.fork_rng(devices=[]):
        model = TinyGRU()
    if capture["parameter_names"] != [name for name, _ in model.named_parameters()]:
        raise ValueError("parameter ordering mismatch")
    for key in ("student", "ema"):
        model.load_state_dict(capture[key], strict=True)
        if any(value.dtype != torch.float32 for value in capture[key].values()):
            raise ValueError("only FP32 model state is supported")
    groups = capture["optimizer"]["param_groups"]
    expected_count = len(capture["parameter_names"])
    if len(groups) != 1 or groups[0]["params"] != list(range(expected_count)):
        raise ValueError("optimizer group/name mapping mismatch")
    if groups[0].get("fused") or groups[0].get("foreach"):
        raise ValueError("unsupported optimizer implementation")
    state = capture["optimizer"]["state"]
    if set(state) != set(range(expected_count)):
        raise ValueError("warm optimizer moments required for every parameter")
    for index, parameter in enumerate(model.parameters()):
        entry = state[index]
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("incomplete AdamW moments")
        if (
            entry["exp_avg"].shape != parameter.shape
            or entry["exp_avg_sq"].shape != parameter.shape
        ):
            raise ValueError("optimizer moment shape mismatch")
    batch = capture["batch"]
    if set(batch) != {
        "x",
        "raw_y",
        "v",
        "teachers",
        "teacher_scale",
        "next_pair",
        "rows",
        "struct_labels",
        "next_distance",
        "next_up",
        "next_known",
    }:
        raise ValueError("batch schema mismatch")
    b, t, width = batch["x"].shape
    if not (b == 2 and t == 10 and width == 3):
        raise ValueError("toy shape bound exceeded")
    shapes = {
        "raw_y": (b, t, 2),
        "v": (b, t),
        "teachers": (2, b, 20, 2),
        "teacher_scale": (2,),
        "next_pair": (b, t),
        "rows": (t,),
        "struct_labels": (b, t, 3),
        "next_distance": (b, t),
        "next_up": (b, t),
        "next_known": (b, t),
    }
    if any(batch[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("batch field shape mismatch")
    if capture["h"].shape != (1, b, 5) or capture["h_ema"].shape != (1, b, 5):
        raise ValueError("hidden shape mismatch")
    float_fields = (
        "x",
        "raw_y",
        "v",
        "teachers",
        "teacher_scale",
        "next_distance",
        "next_up",
        "next_known",
    )
    if any(batch[key].dtype != torch.float32 for key in float_fields):
        raise ValueError("only FP32 floating batch fields are supported")
    if any(batch[key].dtype != torch.int64 for key in ("rows", "next_pair", "struct_labels")):
        raise ValueError("index/label fields must be int64")
    if (batch["rows"] < 0).any() or (batch["rows"] >= 20).any():
        raise ValueError("teacher row index outside sequence")
    if (batch["next_distance"] < 0).any():
        raise ValueError("negative next distance")
    if capture["ema_clock"] != {"updates": 4, "decay": 0.999} or capture["clock"] != {
        "attempted": 4,
        "completed": 4,
    }:
        raise ValueError("only declared warmup clock is supported")
    if not isinstance(capture["source_identity"], str) or len(capture["source_identity"]) != 64:
        raise ValueError("source SHA256 identity required")
    total_bytes = 0

    def inspect(value: Any) -> None:
        nonlocal total_bytes
        if isinstance(value, Tensor):
            total_bytes += value.numel() * value.element_size()
            if value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None:
                raise ValueError("state/labels must be detached CPU tensors")
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite capture")
        elif isinstance(value, dict):
            for item in value.values():
                inspect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                inspect(item)

    inspect(capture)
    if total_bytes > 1_000_000:
        raise ValueError("capture exceeds one MB bound")
    if (batch["v"] < 0).any() or batch["v"].sum() <= 0:
        raise ValueError("invalid row weights")
    if not ((batch["next_known"] == 0) | (batch["next_known"] == 1)).all():
        raise ValueError("invalid future eligibility")
    if capture["clock"]["completed"] > capture["clock"]["attempted"]:
        raise ValueError("invalid update clock")


def capture_toy(
    student: TinyGRU,
    ema: TinyGRU,
    opt: torch.optim.AdamW,
    hidden: Tensor,
    ema_hidden: Tensor,
    batch: dict[str, Tensor],
    completed: int,
    source_identity: str,
) -> dict[str, Any]:
    """Capture caller-owned pre-forward state without advancing any training clock."""
    if any(p.grad is not None for p in student.parameters()):
        raise ValueError("clear prior gradients before pre-forward capture")
    capture = detached_copy(
        {
            "version": VERSION,
            "evidence_kind": "synthetic",
            "phase": "pre_forward_pre_step",
            "student": student.state_dict(),
            "ema": ema.state_dict(),
            "h": hidden,
            "h_ema": ema_hidden,
            "optimizer": opt.state_dict(),
            "parameter_names": [n for n, _ in student.named_parameters()],
            "optimizer_class": "torch.optim.AdamW.cpu.unfused",
            "batch": batch,
            "loss_config": CONFIG,
            "clip": {"max_norm": 0.12, "norm_type": 2.0},
            "clock": {"attempted": completed, "completed": completed},
            "ema_clock": {"updates": completed, "decay": 0.999},
            "rng": torch.get_rng_state(),
            "execution": {
                "dtype": "float32",
                "device": "cpu",
                "autocast": False,
                "scaler": None,
                "scheduler": None,
            },
            "source_identity": source_identity,
        }
    )
    validate_capture(capture)
    return capture


def terms(out: Tensor, ema_out: Tensor, batch: dict[str, Tensor]) -> dict[str, Tensor]:
    """Trainer-equivalent active coefficients on supplied detached synthetic labels."""
    y = batch["raw_y"].clamp(-2, 2)
    teachers = (
        (batch["teachers"] * batch["teacher_scale"][:, None, None, None]).mean(0).clamp(-2, 2)
    )
    ahead = batch["rows"][None, :] + 8
    alive = (batch["next_pair"] > ahead) & (ahead < teachers.shape[1])
    future = teachers.gather(
        1,
        ahead.clamp(max=teachers.shape[1] - 1).expand(y.shape[0], -1)[..., None].expand(-1, -1, 2),
    )
    label = torch.where(alive[..., None], future, y)
    current = teachers.gather(1, batch["rows"][None, :, None].expand(y.shape[0], -1, 2))
    target = 0.5 * label + 0.5 * current
    v = batch["v"]
    w = v[..., None] * y.abs()
    main = (w * (out[..., :2] - target).square()).sum() / w.sum().clamp_min(1e-8)
    ema = (v[..., None] * (out[..., :2] - ema_out[..., :2]).square()).sum() / v.sum().clamp_min(
        1e-8
    )
    ce = torch.zeros_like(v)
    for j, (start, end) in enumerate(((2, 4), (4, 7), (7, 11))):
        ce = ce + F.cross_entropy(
            out[..., start:end].reshape(-1, end - start),
            batch["struct_labels"][..., j].reshape(-1),
            reduction="none",
        ).reshape_as(v)
    struct = 0.3 * (v * ce).sum() / v.sum().clamp_min(1e-8)
    next_w = v * batch["next_known"]
    next_y = (torch.log1p(batch["next_distance"]) - 3.0) / 2.0
    next_error = (out[..., 11] - next_y).square() + F.binary_cross_entropy_with_logits(
        out[..., 12], batch["next_up"], reduction="none"
    )
    nxt = 0.3 * (next_w * next_error).sum() / next_w.sum().clamp_min(1e-8)
    return {"main": main, "ema": ema, "struct": struct, "next": nxt}


def replay(capture: dict[str, Any], multiplier: float) -> dict[str, Tensor]:
    """Reload each arm independently; return gradients, clipping and one-step delta."""
    if not 0 < multiplier < 100:
        raise ValueError("invalid main multiplier")
    validate_capture(capture)
    outer_rng = torch.get_rng_state()
    try:
        torch.set_rng_state(capture["rng"])
        student, ema = TinyGRU(), TinyGRU()
        student.load_state_dict(capture["student"])
        ema.load_state_dict(capture["ema"])
        ema.requires_grad_(False)
        opt = torch.optim.AdamW(student.parameters(), lr=0.03, foreach=False, fused=False)
        opt.load_state_dict(detached_copy(capture["optimizer"]))
        params = tuple(student.parameters())
        before = torch.cat([p.detach().flatten() for p in params])
        out, h_after = student(capture["batch"]["x"], capture["h"])
        with torch.no_grad():
            ema_out, ema_h_after = ema(capture["batch"]["x"], capture["h_ema"])
        losses = terms(out, ema_out, capture["batch"])
        components = {
            key: torch.cat(
                [g.flatten() for g in torch.autograd.grad(loss, params, retain_graph=True)]
            )
            for key, loss in losses.items()
        }
        total = multiplier * losses["main"] + losses["ema"] + losses["struct"] + losses["next"]
        total.backward()
        gradient = torch.cat([p.grad.flatten() for p in params])
        norm = nn.utils.clip_grad_norm_(params, **capture["clip"])
        clipped = torch.cat([p.grad.flatten() for p in params])
        opt.step()
        delta = torch.cat([p.detach().flatten() for p in params]) - before
        result = {
            **{f"g_{key}": g.detach() for key, g in components.items()},
            "gradient": gradient.detach(),
            "clipped": clipped.detach(),
            "delta": delta,
            "preclip_norm": norm.detach(),
            "losses": torch.stack(list(losses.values())).detach(),
            "output": out.detach(),
            "ema_output": ema_out,
            "h_after": h_after.detach(),
            "ema_h_after": ema_h_after,
        }
        if not all(torch.isfinite(value).all() for value in result.values()):
            raise ValueError("nonfinite replay output")
        return result
    finally:
        torch.set_rng_state(outer_rng)


def analytic_output_gradient(
    out: Tensor, ema_out: Tensor, batch: dict[str, Tensor], multiplier: float
) -> Tensor:
    """Independent rowwise derivative oracle; no autodiff loss helpers called."""
    derivative = torch.zeros_like(out)
    v = batch["v"]
    mass = (v[..., None] * batch["raw_y"].clamp(-2, 2).abs()).sum().clamp_min(1e-8)
    r = v.sum().clamp_min(1e-8)
    for b in range(out.shape[0]):
        for t in range(out.shape[1]):
            row = int(batch["rows"][t])
            teacher_now = (
                sum(batch["teacher_scale"][k] * batch["teachers"][k, b, row] for k in range(2)) / 2
            )
            label = batch["raw_y"][b, t].clamp(-2, 2)
            ahead = int(batch["rows"][t]) + 8
            if int(batch["next_pair"][b, t]) > ahead and ahead < 20:
                label = (
                    sum(
                        batch["teacher_scale"][k] * batch["teachers"][k, b, ahead] for k in range(2)
                    )
                    / 2
                ).clamp(-2, 2)
            target = (label + teacher_now.clamp(-2, 2)) / 2
            derivative[b, t, :2] = (
                2
                * multiplier
                * v[b, t]
                * batch["raw_y"][b, t].clamp(-2, 2).abs()
                * (out[b, t, :2] - target)
                / mass
                + 2 * v[b, t] * (out[b, t, :2] - ema_out[b, t, :2]) / r
            )
            for j, (start, end) in enumerate(((2, 4), (4, 7), (7, 11))):
                d = out[b, t, start:end].softmax(0)
                d[int(batch["struct_labels"][b, t, j])] -= 1
                derivative[b, t, start:end] = 0.3 * v[b, t] * d / r
            weight = v[b, t] * batch["next_known"][b, t]
            n = (v * batch["next_known"]).sum().clamp_min(1e-8)
            label_next = (batch["next_distance"][b, t].log1p() - 3) / 2
            derivative[b, t, 11] = 0.6 * weight * (out[b, t, 11] - label_next) / n
            derivative[b, t, 12] = (
                0.3 * weight * (out[b, t, 12].sigmoid() - batch["next_up"][b, t]) / n
            )
    return derivative
