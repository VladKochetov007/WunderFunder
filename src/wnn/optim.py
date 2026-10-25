"""Optimizer, schedule and recurrent-initialisation options for scripts/train_rnn.py (research flags, default off).

- ``MuonAdamW``: Muon (orthogonalised Nesterov momentum, K. Jordan 2024) on the recurrent core's gate matrices, one
  Newton-Schulz orthogonalisation per gate block, scaled to AdamW's update RMS (0.2 sqrt(max(rows, cols)), Liu et al.
  2025, arXiv 2502.16982) so one lr / weight-decay schedule drives both; AdamW for inputs, head output and biases.
- ``wsd_lambda``: warmup-stable-decay multiplier with a (1 - sqrt) cooldown (Hagele et al. 2024, arXiv 2405.18392).
- ``rnn_init_``: chrono gate biases (Tallec & Ollivier 2018, arXiv 1804.11188) and per-gate orthogonal W_hh.
- ``GradStats``: pre-clip gradient norms per parameter family, kept on device, summarised once per epoch.
"""

from __future__ import annotations

import math

import torch
from torch import nn

NS_COEFFS = (3.4445, -4.7750, 2.0315)  # quintic Newton-Schulz, K. Jordan's Muon


def orthogonalize(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate U V^T of every trailing (m, n) matrix of ``g`` (singular values pushed into ~[0.7, 1.2]).

    Always fp32: the strict-FP32 policy forbids bf16 casts on any device.
    """
    a, b, c = NS_COEFFS
    x = g.to(torch.float32)
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * s @ s) @ x
    if tall:
        x = x.mT
    return x.to(g.dtype)


class MuonAdamW(torch.optim.Optimizer):
    """One optimizer for two update rules, so GradScaler, OneCycleLR (which cycles ``betas[0]``) and the epoch
    update counters work unchanged. Groups with ``muon=True`` hold 2-D weights split into ``blocks`` equal row
    blocks (the stacked gates of nn.GRU / nn.LSTM); every other group takes torch.optim.AdamW's (single-tensor) step.
    ``betas[0]`` is both Adam's beta1 and Muon's momentum.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        ns_steps: int = 5,
        rms: float = 0.2,
        update: str = "ns",
    ):
        if update not in ("ns", "rms"):
            raise ValueError(f"unknown Muon update rule: {update}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            rms=rms,
            update=update,
            muon=False,
            blocks=1,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            if group["muon"]:
                for p in group["params"]:
                    if p.ndim != 2 or p.shape[0] % group["blocks"]:
                        raise ValueError(
                            f"Muon needs 2-D weights with rows divisible by {group['blocks']}"
                        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if group["muon"]:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.lerp_(g, 1 - beta1)
                    k = group["blocks"]
                    rows, cols = p.shape[0] // k, p.shape[1]
                    nesterov = g.lerp(buf, beta1).reshape(k, rows, cols)
                    if group["update"] == "rms":
                        # Ablation: same per-block step RMS as the polar factor, singular spread kept.
                        block_rms = nesterov.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt()
                        update = nesterov / (
                            block_rms.clamp_min(1e-12) * math.sqrt(max(rows, cols))
                        )
                    else:
                        update = orthogonalize(nesterov, group["ns_steps"])
                    p.mul_(1 - lr * wd)
                    p.add_(
                        update.reshape(p.shape),
                        alpha=-lr * group["rms"] * math.sqrt(max(rows, cols)),
                    )
                    continue
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                p.mul_(1 - lr * wd)
                exp_avg.lerp_(g, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bias1 = 1 - beta1 ** state["step"]
                bias2 = 1 - beta2 ** state["step"]
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias2)).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-lr / bias1)
        return loss


def _gates(rnn: nn.Module) -> int:
    if isinstance(rnn, nn.LSTM):
        return 4
    if isinstance(rnn, nn.GRU):
        return 3
    raise ValueError(f"no gate layout for {type(rnn).__name__}")


def muon_param_groups(model: nn.Module, scope: str = "rnn") -> list[dict]:
    """Trainable parameters split for MuonAdamW.

    scope 'rnn': ``model.gru`` and optional ``model.experts`` recurrent
    ``weight_ih_l*`` / ``weight_hh_l*`` matrices (one Muon block per gate);
    scope 'hidden': also the hidden readout matrices ``head.0.weight`` and,
    when present, ``head.extra.weight``. The input projection ``inp`` (first
    layer), output layers and biases stay on AdamW, as Muon prescribes for
    input/output layers and vectors.
    """
    if scope not in ("rnn", "hidden"):
        raise ValueError(scope)
    rnn = getattr(model, "gru", None)
    if not isinstance(rnn, (nn.GRU, nn.LSTM)):
        raise ValueError("Muon groups need an RNNModel-style recurrent core at model.gru")
    gates = _gates(rnn)
    recurrent_prefixes = ("gru.weight_",)
    experts = getattr(model, "experts", None)
    if experts is not None:
        if not isinstance(experts, nn.ModuleList):
            raise ValueError("Muon specialist experts must be an nn.ModuleList")
        if any(type(expert) is not type(rnn) or _gates(expert) != gates for expert in experts):
            raise ValueError("Muon requires specialist experts with the same recurrent gate layout")
        recurrent_prefixes += tuple(f"experts.{index}.weight_" for index in range(len(experts)))
    gate_mats, hidden_mats, adam = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith(recurrent_prefixes):
            gate_mats.append(p)
        elif scope == "hidden" and name in {"head.0.weight", "head.extra.weight"}:
            hidden_mats.append(p)
        else:
            adam.append(p)
    if not gate_mats:
        raise ValueError("no trainable recurrent matrices for Muon (frozen core?)")
    groups = [{"params": adam}, {"params": gate_mats, "muon": True, "blocks": gates}]
    if hidden_mats:
        groups.append({"params": hidden_mats, "muon": True, "blocks": 1})
    return groups


def wsd_lambda(
    total: int,
    warmup_frac: float = 0.05,
    decay_frac: float = 0.2,
    final_frac: float = 0.01,
    shape: str = "sqrt",
):
    """LambdaLR multiplier: linear warmup over ``warmup_frac`` of the steps, constant peak, then a cooldown to
    ``final_frac`` of the peak over the last ``decay_frac`` of the steps.  ``shape="sqrt"`` is the (1 - sqrt)
    cooldown; ``shape="linear"`` with ``decay_frac=1`` and ``final_frac=0`` is linear decay-to-zero after warmup
    (Bergsma et al. 2025, arXiv 2502.15938)."""
    if shape not in {"sqrt", "linear"}:
        raise ValueError(f"unknown WSD cooldown shape: {shape}")
    warm = max(1, round(warmup_frac * total))
    start = max(warm, total - round(decay_frac * total))

    def multiplier(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        if step < start:
            return 1.0
        t = min(1.0, (step - start) / max(1, total - start))
        return final_frac + (1.0 - final_frac) * (1.0 - (t if shape == "linear" else math.sqrt(t)))

    return multiplier


@torch.no_grad()
def rnn_init_(model: nn.Module, mode: str, t_max: float, seed: int) -> None:
    """Re-initialise the recurrent core of an RNNModel-style ``model.gru`` from a private generator (the global RNG,
    and hence data order, stays identical to the default arm with the same seed).

    chrono: memory timescales T ~ U[2, t_max] rows. GRU (h' = (1 - z) n + z h): b_z = log(T - 1) so z = 1 - 1/T;
      LSTM: b_f = log(T - 1), b_i = -b_f. The input-side bias carries the value, the hidden-side bias is zeroed.
    orth: each gate's H x H block of weight_hh is an independent orthogonal matrix.
    """
    if mode == "default":
        return
    parts = set(mode.split("_"))
    if not parts <= {"chrono", "orth"}:
        raise ValueError(mode)
    rnn = getattr(model, "gru", None)
    gates = _gates(rnn)
    h = rnn.hidden_size
    gen = torch.Generator().manual_seed(int(seed) + 7919)
    for k in range(rnn.num_layers):
        if "chrono" in parts:
            # the memory gate is the second block in both layouts: z of GRU (r, z, n), f of LSTM (i, f, g, o)
            b_ih, b_hh = getattr(rnn, f"bias_ih_l{k}"), getattr(rnn, f"bias_hh_l{k}")
            bias = torch.empty(h).uniform_(1.0, t_max - 1.0, generator=gen).log().to(b_ih)
            b_ih[h : 2 * h] = bias
            b_hh[h : 2 * h] = 0.0
            if gates == 4:
                b_ih[:h] = -bias
                b_hh[:h] = 0.0
        if "orth" in parts:
            w = getattr(rnn, f"weight_hh_l{k}")
            for j in range(gates):
                block = torch.empty(h, h)
                nn.init.orthogonal_(block, generator=gen)
                w[j * h : (j + 1) * h] = block.to(w)


class GradStats:
    """Pre-clip gradient norms per parameter family (recurrent hh / ih, input projection, head, biases) for every
    optimizer step, kept on device; ``finish`` makes one host copy per epoch. Call after GradScaler.unscale_ and
    before clipping. AMP-skipped (non-finite) steps are counted, not averaged."""

    def __init__(self, model: nn.Module, clip: float):
        self.clip = clip
        fam = {"rnn_hh": [], "rnn_ih": [], "inp": [], "head": [], "bias": [], "other": []}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:
                fam["bias"].append(p)
            elif ".weight_hh" in name:
                fam["rnn_hh"].append(p)
            elif ".weight_ih" in name:
                fam["rnn_ih"].append(p)
            elif name.startswith("inp."):
                fam["inp"].append(p)
            elif name.startswith("head."):
                fam["head"].append(p)
            else:
                fam["other"].append(p)
        self.families = {k: v for k, v in fam.items() if v}
        self.rows: list[torch.Tensor] = []

    @torch.no_grad()
    def observe(self) -> None:
        norms = []
        for params in self.families.values():
            grads = [p.grad for p in params if p.grad is not None]
            norms.append(
                torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)).float())
                if grads
                else torch.zeros((), device=params[0].device)
            )
        self.rows.append(torch.stack(norms))

    def finish(self) -> dict:
        if not self.rows:
            return {}
        per = torch.stack(self.rows).cpu().double()  # (steps, families)
        self.rows = []
        total = per.pow(2).sum(1).sqrt()
        ok = torch.isfinite(total)
        out = {"steps": int(len(total)), "nonfinite_steps": int((~ok).sum()), "clip": self.clip}
        t = total[ok]
        if len(t):
            q = torch.quantile(t, torch.tensor([0.05, 0.5, 0.95], dtype=t.dtype)).tolist()
            out.update(
                total_q05_q50_q95=[round(v, 5) for v in q],
                clip_rate=round(float((t > self.clip).double().mean()), 4),
                mean_clip_factor=round(float(torch.clamp(self.clip / t, max=1.0).mean()), 4),
            )
            out["family_median"] = {
                k: round(float(per[ok, i].median()), 5) for i, k in enumerate(self.families)
            }
            share = per[ok].pow(2) / t.pow(2)[:, None]
            out["family_sq_share_mean"] = {
                k: round(float(share[:, i].mean()), 4) for i, k in enumerate(self.families)
            }
        return out
