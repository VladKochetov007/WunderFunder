"""Torch Muon semantics for the historical TL3 recurrent training recipes."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

_NS_COEFFICIENTS = (3.4445, -4.775, 2.0315)
_FORMAT = "wnn.tl3-muon-adamw.v1"


def _require_fp32_environment() -> None:
    """Reject autocast or reduced FP32 matmul settings for strict NS work."""
    if torch.is_autocast_enabled("cuda") or torch.is_autocast_enabled("cpu"):
        raise RuntimeError("strict-FP32 Muon requires autocast to be disabled")
    if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("strict-FP32 Muon requires CUDA TF32 to be disabled")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("strict-FP32 Muon requires float32 matmul precision 'highest'")


def _orthogonalize(
    gradient: torch.Tensor, *, dtype: torch.dtype, steps: int = 5, eps: float = 1e-7
) -> torch.Tensor:
    """Apply Torch Muon's quintic Newton–Schulz iteration to one full matrix."""
    if gradient.ndim != 2:
        raise ValueError("Muon requires two-dimensional gradient matrices")
    if steps < 0 or steps >= 100 or not math.isfinite(eps) or eps <= 0:
        raise ValueError("invalid Newton–Schulz steps or epsilon")
    a, b, c = _NS_COEFFICIENTS
    matrix = gradient.to(dtype)
    transposed = matrix.size(0) > matrix.size(1)
    if transposed:
        matrix = matrix.T
    matrix = matrix / matrix.norm().clamp(min=eps)
    for _ in range(steps):
        gram = matrix @ matrix.T
        update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        matrix = torch.addmm(matrix, update, matrix, beta=a)
        if matrix.dtype != dtype:
            raise TypeError("Newton–Schulz changed its declared intermediate dtype")
    if transposed:
        matrix = matrix.T
    return matrix


class _CompatMuon(torch.optim.Optimizer):
    """Torch 2.8-compatible single-tensor implementation of Torch Muon."""

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        *,
        lr: float,
        weight_decay: float,
        ns_dtype: torch.dtype,
    ) -> None:
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": 0.95,
            "nesterov": True,
            "ns_coefficients": _NS_COEFFICIENTS,
            "eps": 1e-7,
            "ns_steps": 5,
            "adjust_lr_fn": "match_rms_adamw",
            "ns_dtype": ns_dtype,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            if any(parameter.ndim != 2 for parameter in group["params"]):
                raise ValueError("Muon groups may contain only 2D parameters")

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        """Apply Muon's decoupled decay and orthogonalized Nesterov update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buffer = state["momentum_buffer"]
                momentum = group["momentum"]
                buffer.lerp_(gradient, 1 - momentum)
                update = gradient.lerp(buffer, momentum) if group["nesterov"] else buffer
                ns_dtype = group["ns_dtype"]
                if ns_dtype == torch.float32:
                    _require_fp32_environment()
                    if parameter.dtype != torch.float32 or gradient.dtype != torch.float32:
                        raise TypeError("strict-FP32 Muon requires FP32 parameters and gradients")
                    if buffer.dtype != torch.float32:
                        raise TypeError("strict-FP32 Muon requires FP32 momentum state")
                update = _orthogonalize(
                    update,
                    dtype=ns_dtype,
                    steps=group["ns_steps"],
                    eps=group["eps"],
                )
                rows, columns = parameter.shape
                adjusted_lr = group["lr"] * 0.2 * math.sqrt(max(rows, columns))
                lr = group["lr"]
                parameter.mul_(1 - lr * group["weight_decay"])
                parameter.add_(update, alpha=-adjusted_lr)
        return loss


def _muon_optimizer(
    params: list[nn.Parameter],
    names: list[str],
    *,
    lr: float,
    weight_decay: float,
    ns_dtype: torch.dtype,
) -> torch.optim.Optimizer:
    """Use native Muon where its arithmetic matches the requested NS dtype."""
    group = {"params": params, "param_names": names, "role": "muon_hidden"}
    native = getattr(torch.optim, "Muon", None)
    if native is not None and ns_dtype == torch.bfloat16:
        return native(
            [group],
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.95,
            nesterov=True,
            ns_coefficients=_NS_COEFFICIENTS,
            eps=1e-7,
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
        )
    return _CompatMuon([group], lr=lr, weight_decay=weight_decay, ns_dtype=ns_dtype)


class TL3MuonAdamW(torch.optim.Optimizer):
    """One trainer-facing optimizer over separate native Muon and AdamW engines."""

    def __init__(
        self,
        muon_params: list[nn.Parameter],
        muon_names: list[str],
        adamw_params: list[nn.Parameter],
        adamw_names: list[str],
        *,
        lr: float,
        muon_lr_mult: float,
        beta2: float,
        weight_decay: float,
        fused_adam: bool,
        ns_dtype: torch.dtype,
    ) -> None:
        """Build a single interface while retaining independent optimizer state."""
        if not muon_params or not adamw_params:
            raise ValueError("TL3 Muon needs nonempty Muon and AdamW parameter groups")
        if not math.isfinite(muon_lr_mult) or muon_lr_mult <= 0:
            raise ValueError("Muon learning-rate multiplier must be positive and finite")
        if len({id(parameter) for parameter in [*muon_params, *adamw_params]}) != (
            len(muon_params) + len(adamw_params)
        ):
            raise ValueError("TL3 Muon parameter groups overlap")
        initial_groups = [
            {"params": muon_params, "lr": lr * muon_lr_mult, "role": "muon_hidden"},
            {"params": adamw_params, "lr": lr, "role": "adamw_excluded"},
        ]
        super().__init__(initial_groups, {"lr": lr, "weight_decay": weight_decay})
        self.muon_optimizer = _muon_optimizer(
            muon_params,
            muon_names,
            lr=lr * muon_lr_mult,
            weight_decay=weight_decay,
            ns_dtype=ns_dtype,
        )
        self.adamw_optimizer = torch.optim.AdamW(
            [
                {
                    "params": adamw_params,
                    "param_names": adamw_names,
                    "role": "adamw_excluded",
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            ],
            lr=lr,
            betas=(0.9, beta2),
            weight_decay=weight_decay,
            fused=fused_adam,
        )
        self.param_groups = self.muon_optimizer.param_groups + self.adamw_optimizer.param_groups
        self.state = defaultdict(dict)
        self._sync_state()

    def _sync_state(self) -> None:
        """Expose both suboptimizers' state through the trainer-facing object."""
        self.state.clear()
        self.state.update(self.muon_optimizer.state)
        self.state.update(self.adamw_optimizer.state)

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:
        """Update both disjoint parameter groups once."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon_optimizer.step()
        self.adamw_optimizer.step()
        self._sync_state()
        return loss

    def state_dict(self) -> dict[str, Any]:
        """Serialize both optimizer states with an explicit composite format."""
        return {
            "format": _FORMAT,
            "muon": self.muon_optimizer.state_dict(),
            "adamw": self.adamw_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore both states and refresh the exposed parameter-group aliases."""
        if state_dict.get("format") != _FORMAT or not {"muon", "adamw"} <= state_dict.keys():
            raise ValueError("not a TL3 Muon/AdamW optimizer state")
        self.muon_optimizer.load_state_dict(state_dict["muon"])
        self.adamw_optimizer.load_state_dict(state_dict["adamw"])
        self.param_groups = self.muon_optimizer.param_groups + self.adamw_optimizer.param_groups
        self._sync_state()


def build_tl3_muon_optimizer(
    model: nn.Module,
    *,
    lr: float,
    muon_lr_mult: float,
    beta2: float,
    weight_decay: float,
    fused_adam: bool,
    strict_fp32: bool,
) -> TL3MuonAdamW:
    """Split all hidden 2D matrices from input/output/vector AdamW parameters."""
    muon_names: list[str] = []
    muon_params: list[nn.Parameter] = []
    adamw_names: list[str] = []
    adamw_params: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        input_or_output = (
            name.startswith("inp.")
            or name.startswith("head.2.")
            or name in {"mean", "std"}
            or name.endswith((".mean", ".std"))
        )
        if parameter.ndim == 2 and not input_or_output:
            muon_names.append(name)
            muon_params.append(parameter)
        else:
            adamw_names.append(name)
            adamw_params.append(parameter)
    return TL3MuonAdamW(
        muon_params,
        muon_names,
        adamw_params,
        adamw_names,
        lr=lr,
        muon_lr_mult=muon_lr_mult,
        beta2=beta2,
        weight_decay=weight_decay,
        fused_adam=fused_adam,
        ns_dtype=torch.float32 if strict_fp32 else torch.bfloat16,
    )
