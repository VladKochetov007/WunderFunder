"""Opt-in training telemetry: loss curve, optimizer hyper-parameters, gradient/weight/update norms.

Everything stays on the device until a flush, so a trace costs one host sync per ``flush_every`` rows.
Rows are appended to ``trace.jsonl`` as ``{"kind": "step"|"state", ...}``:

- ``step``: total loss, named loss parts, per-param-group learning rate and first-moment coefficient,
  AMP loss scale, pre-clip total and per-family gradient norms (captured between unscale and clip).
- ``state``: per-family weight norm, update ratio ``||w_t - w_prev|| / ||w_prev||`` since the previous
  state row, and optimizer-state statistics (second-moment RMS, momentum norm) when the optimizer exposes
  the usual ``exp_avg`` / ``exp_avg_sq`` / ``momentum_buffer`` tensors.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn


def parameter_families(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    groups: dict[str, list[nn.Parameter]] = {"rnn_hh": [], "rnn_ih": [], "inp": [], "head": [], "bias": [], "other": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1:
            groups["bias"].append(parameter)
        elif ".weight_hh" in name:
            groups["rnn_hh"].append(parameter)
        elif ".weight_ih" in name:
            groups["rnn_ih"].append(parameter)
        elif name.startswith("inp."):
            groups["inp"].append(parameter)
        elif name.startswith("head."):
            groups["head"].append(parameter)
        else:
            groups["other"].append(parameter)
    return {key: value for key, value in groups.items() if value}


def _norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        return torch.zeros(())
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(tensors)))


def _group_value(group: dict, key: str, index: int | None = None):
    value = group.get(key)
    if value is None:
        return None
    if index is not None and isinstance(value, (tuple, list)):
        value = value[index]
    return float(value)


class TrainTrace:
    def __init__(self, path: Path, model: nn.Module, every: int, state_every: int, flush_every: int = 64) -> None:
        self.path, self.every, self.state_every, self.flush_every = Path(path), max(1, every), state_every, flush_every
        self.families = parameter_families(model)
        self.previous: dict[str, list[torch.Tensor]] | None = None
        self.pending: list[tuple[dict, torch.Tensor, list[str]]] = []
        self.grad_row: tuple[torch.Tensor, list[str]] | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def observe_grad(self, step: int) -> None:
        """Call after ``scaler.unscale_`` and before clipping."""
        if step % self.every:
            self.grad_row = None
            return
        norms, names = [], []
        for family, parameters in self.families.items():
            grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
            norms.append(_norm(grads).to(parameters[0].device) if grads else torch.zeros((), device=parameters[0].device))
            names.append(f"gn_{family}")
        total = torch.linalg.vector_norm(torch.stack(norms)) if norms else torch.zeros(())
        self.grad_row = (torch.cat([total.reshape(1), torch.stack(norms)]), ["gn_total", *names])

    def record(self, step: int, epoch: int, loss: torch.Tensor, parts: dict[str, torch.Tensor],
               opt: torch.optim.Optimizer, scaler: object) -> None:
        if step % self.every:
            return
        names = ["loss", *[f"loss_{key}" for key in sorted(parts)]]
        values = [loss.detach().float().reshape(1), *[parts[key].detach().float().reshape(1) for key in sorted(parts)]]
        if self.grad_row is not None:
            values.append(self.grad_row[0].to(values[0].device))
            names += self.grad_row[1]
        scale = getattr(scaler, "_scale", None)
        if isinstance(scale, torch.Tensor):
            values.append(scale.detach().float().reshape(1).to(values[0].device))
            names.append("amp_scale")
        groups = getattr(opt, "param_groups", [])
        meta = {"kind": "step", "step": step, "epoch": epoch,
                "lr": [_group_value(g, "lr") for g in groups],
                "beta1": [_group_value(g, "betas", 0) if "betas" in g else _group_value(g, "momentum") for g in groups]}
        self.pending.append((meta, torch.cat(values), names))
        if self.state_every and step % self.state_every == 0:
            self.pending.append(self._state_row(step, epoch, opt))
        if len(self.pending) >= self.flush_every:
            self.flush()

    @torch.no_grad()
    def _state_row(self, step: int, epoch: int, opt: torch.optim.Optimizer):
        values, names = [], []
        current = {family: [p.detach().clone() for p in parameters] for family, parameters in self.families.items()}
        for family, parameters in self.families.items():
            weights = [p.detach().float() for p in parameters]
            values.append(_norm(weights).reshape(1))
            names.append(f"w_{family}")
            if self.previous is not None:
                delta = _norm([a.float() - b.float() for a, b in zip(current[family], self.previous[family], strict=True)])
                values.append((delta / _norm([b.float() for b in self.previous[family]]).clamp_min(1e-12)).reshape(1))
                names.append(f"ur_{family}")
            second, first = [], []
            for p in parameters:
                state = opt.state.get(p, {}) if hasattr(opt, "state") else {}
                if "exp_avg_sq" in state:
                    second.append(state["exp_avg_sq"].float())
                if "exp_avg" in state:
                    first.append(state["exp_avg"].float())
                elif "momentum_buffer" in state and state["momentum_buffer"] is not None:
                    first.append(state["momentum_buffer"].float())
            if second:
                values.append(torch.sqrt(torch.cat([t.reshape(-1) for t in second]).mean()).reshape(1))
                names.append(f"v_rms_{family}")
            if first:
                values.append(_norm(first).reshape(1))
                names.append(f"m_norm_{family}")
        self.previous = current
        return {"kind": "state", "step": step, "epoch": epoch}, torch.cat(values), names

    def flush(self) -> None:
        if not self.pending:
            return
        with self.path.open("a") as handle:
            for meta, vector, names in self.pending:
                numbers = vector.cpu().tolist()
                row = dict(meta)
                row.update({name: (None if not math.isfinite(number) else number) for name, number in zip(names, numbers, strict=True)})
                handle.write(json.dumps(row) + "\n")
        self.pending.clear()
