"""Critic unit checks (CPU only): PooledCorrMemory gradient/numerics, head-only freeze, aux memory sharing.

    WNN_CPU_PCT=7 WNN_MEM_MAX=4G CUDA_VISIBLE_DEVICES='' scripts/limited .venv/bin/python \
        research/campaign_2026-10-01/critic/check_units.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from wnn.nn import PooledCorrMemory, RNNModel, weighted_pooled_corr_loss  # noqa: E402

torch.manual_seed(0)
results = {}


def explicit_rho(ps, ys, ws):
    """Whole-pool |y|-weighted Pearson over explicit rows (float64, two-pass)."""
    p, y, w = torch.cat(ps), torch.cat(ys), torch.cat(ws)
    n = w.sum(0)
    my, mp = (w * y).sum(0) / n, (w * p).sum(0) / n
    cov = (w * (y - my) * (p - mp)).sum(0) / n
    vy, vp = (w * (y - my) ** 2).sum(0) / n, (w * (p - mp) ** 2).sum(0) / n
    return cov / torch.sqrt(vy * vp)


def chunk(b=4, t=50, offset=0.0, scale=1.0, dtype=torch.float64):
    y = (torch.randn(b, t, 2, dtype=torch.float64) * 0.8).round()  # quantised, many zeros like the targets
    p = (0.3 * y + torch.randn(b, t, 2, dtype=torch.float64) * 0.5) * scale + offset
    v = (torch.rand(b, t, dtype=torch.float64) > 0.2).double() * torch.rand(b, t, dtype=torch.float64)
    return p.to(dtype), y, v


# 1. decay=0 must reproduce the per-chunk pooled loss
p, y, v = chunk()
m0 = PooledCorrMemory(0.0)
m0(p, y, v)
a = m0(p, y, v)
b = weighted_pooled_corr_loss(p, y, v)
results["decay0_equals_pooled_loss"] = abs(float(a) - float(b)) < 1e-6

# 2. gradient = whole-pool gradient restricted to current rows / share
for offset, scale, tag in [(0.0, 1.0, "plain"), (50.0, 1e-3, "offset50_sd1e-3"), (0.0, 1e-4, "tiny_scale")]:
    decay = 0.9
    mem = PooledCorrMemory(decay)
    history = [chunk(offset=offset, scale=scale) for _ in range(3)]
    for hp, hy, hv in history:
        mem(hp, hy, hv)
    cp, cy, cv = chunk(offset=offset, scale=scale)
    cp = cp.clone().requires_grad_(True)
    loss = mem(cp, cy, cv)
    loss.backward()
    g_mem = cp.grad.clone()

    # explicit pool: chunk i (0-based, of 3) carries weight decay^(3-i) at the time of the current call
    ps, ys, ws = [], [], []
    for age, (hp, hy, hv) in zip((3, 2, 1), history):
        yc = hy.clamp(-2, 2)
        ps.append(hp.reshape(-1, 2)); ys.append(yc.reshape(-1, 2))
        ws.append((yc.abs() * hv[..., None]).reshape(-1, 2) * decay ** age)
    cp2 = cp.detach().clone().requires_grad_(True)
    ycur = cy.clamp(-2, 2)
    wcur = (ycur.abs() * cv[..., None]).reshape(-1, 2)
    rho = explicit_rho(ps + [cp2.reshape(-1, 2)], ys + [ycur.reshape(-1, 2)], ws + [wcur])
    n_pool = torch.cat(ws + [wcur]).sum(0)
    share = wcur.sum(0) / n_pool
    ((1 - rho) / share).mean().backward()
    g_ref = cp2.grad
    rel = float((g_mem - g_ref).abs().max() / g_ref.abs().max())
    results[f"grad_matches_pool_over_share[{tag}]"] = {"max_rel_err": rel, "ok": rel < 1e-6}

# 3. fp16 prediction input (autocast head output) -> grad dtype and finiteness under a GradScaler-like scale
mem = PooledCorrMemory(0.98)
for _ in range(60):  # reach steady state share ~ 1/50
    hp, hy, hv = chunk(b=16, t=200)
    mem(hp.half(), hy, hv)
cp, cy, cv = chunk(b=16, t=200)
cp16 = cp.half().requires_grad_(True)
loss = mem(cp16, cy, cv)
(loss * 65536.0).backward()
g = cp16.grad
results["fp16_grad"] = {"dtype": str(g.dtype), "finite": bool(torch.isfinite(g).all()),
                        "nonzero_frac": float((g != 0).float().mean()),
                        "loss_value_steady_state": float(loss), "note": "loss value ~ (1-rho)/share, inflated ~50x"}

# 4. chunk with zero weight: loss explodes by 1/1e-6 but grads are zero
mem = PooledCorrMemory(0.98)
hp, hy, hv = chunk()
mem(hp, hy, hv)
cp, cy, _ = chunk()
cp = cp.clone().requires_grad_(True)
loss = mem(cp, cy, torch.zeros(cy.shape[:2], dtype=torch.float64))
loss.backward()
results["zero_weight_chunk"] = {"loss_value": float(loss), "grad_abs_max": float(cp.grad.abs().max())}

# 5. the same instance is used for the --aux shifted-target heads in train_rnn.objective(): memory is shared
mem = PooledCorrMemory(0.98)
p, y, v = chunk()
mem(p, y, v)
main_only = mem.memory.clone()
pa, ya, va = chunk(offset=3.0)  # an aux head with different target/prediction levels
mem(pa, ya, va)
results["aux_calls_mutate_main_memory"] = bool(not torch.allclose(mem.memory, 0.98 * main_only))

# 6. joint_head_ft freeze rule: only head.* get grads; state carried across chunks
model = RNNModel(12, hidden=16, layers=2, n_out=2 + 21)
for name, prm in model.named_parameters():
    prm.requires_grad_(name.startswith("head."))
x = torch.randn(3, 40, 12)
h = None
total = 0
for s in range(0, 40, 20):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, h = model(x[:, s:s + 20], h)
    h = h.detach()
    total = total + out[..., :2].float().pow(2).mean()
total.backward()
grads = {n: prm.grad is not None for n, prm in model.named_parameters()}
results["joint_freeze_only_head_grads"] = all(v == n.startswith("head.") for n, v in grads.items())
# the ft_gclp head stage freezes by prefix inp,gru: same trainable set for enc=linear RNNModel
ft_trainable = {n for n, _ in model.named_parameters() if not n.startswith(("inp", "gru"))}
results["ft_gclp_prefix_equals_head_rule"] = ft_trainable == {n for n in grads if n.startswith("head.")}

print(json.dumps(results, indent=1))
