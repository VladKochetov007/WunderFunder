"""SAM backward only; caller owns recurrence, clipping, AdamW, scaler and EMA."""
from __future__ import annotations

import torch


def sam_backward(loss, closure, parameters, scaler, rho, *, force_two_pass=False):
    """Leave scaled second-pass gradients and exactly restored parameter values.

    First-pass overflow is handed to the caller's ordinary GradScaler path:
    no perturbation is made, and scaled first-pass gradients trigger its skip.
    Neither optimizer/scaler state nor an incoming recurrent state is modified.
    The caller must zero parameter gradients before this function.
    """
    if rho < 0:
        raise ValueError("negative SAM radius")
    params = tuple(p for p in parameters if p.requires_grad)
    if not params or any(p.grad is not None for p in params):
        raise ValueError("SAM requires trainable parameters with gradients cleared")
    if rho == 0 and not force_two_pass:
        scaler.scale(loss).backward()
        return {"two_pass": False, "first_overflow": False}

    scaled_grads = torch.autograd.grad(scaler.scale(loss), params)
    scale = scaler.get_scale()
    grads = tuple(g.detach().float() / scale for g in scaled_grads)
    finite = torch.stack([torch.isfinite(g).all() for g in grads]).all()
    if not bool(finite):
        if not scaler.is_enabled():
            raise FloatingPointError("nonfinite SAM first gradient with disabled GradScaler")
        for p, g in zip(params, scaled_grads):
            p.grad = g.detach()
        return {"two_pass": False, "first_overflow": True}

    # Float64 reduction keeps finite large gradients from overflowing the norm
    # and accidentally taking an ordinary (non-SAM) update.
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(g, dtype=torch.float64) for g in grads
    ]))
    perturbations = tuple(g * (rho / norm.clamp_min(1e-12)) for g in grads)
    saved = tuple(p.detach().clone() for p in params)
    try:
        with torch.no_grad():
            for p, old, epsilon in zip(params, saved, perturbations):
                p.copy_(old + epsilon.to(p.dtype))
        # First autograd.grad has released the first graph. A closure exception,
        # backward exception or second-pass overflow cannot leave perturbed weights.
        second_loss = closure()
        scaler.scale(second_loss).backward()
    finally:
        with torch.no_grad():
            for p, old in zip(params, saved):
                p.copy_(old)
    return {"two_pass": True, "first_overflow": False}


def skip_evaluation_rng(loader):
    """Match sequential, map-style, nonpersistent DataLoader iteration RNG."""
    if loader.persistent_workers:
        raise ValueError("RNG placeholder requires nonpersistent evaluation workers")
    if not isinstance(loader.sampler, torch.utils.data.SequentialSampler):
        raise ValueError("RNG placeholder requires a sequential evaluation sampler")
    torch.empty((), dtype=torch.int64).random_(generator=loader.generator)
