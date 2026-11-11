"""Independent, resumable Gaussian hidden-state perturbations for card08."""

from __future__ import annotations

import math

import torch
from torch import Tensor

NOISE_STD_FRACTION = 0.01
HIDDEN_RMS_FLOOR = 1e-3
MAX_EXACT_FLOAT32_COUNTER = 2**24


def perturb_hidden_state(hidden: Tensor, counter: Tensor, seed: int) -> Tensor:
    """Add sequencewise RMS-scaled noise using only the persisted FP32 counter.

    A fresh device generator is seeded from the immutable fit seed and the
    registered counter. This keeps the perturbation stream independent from
    dropout/global RNG while saving its exact position in a canonical FP32 buffer.
    """
    if hidden.ndim != 3 or hidden.dtype != torch.float32:
        raise ValueError("card08 hidden state must be FP32 [layers,batch,width]")
    if counter.shape != () or counter.dtype != torch.float32 or counter.device != hidden.device:
        raise ValueError("card08 RNG counter must be a scalar FP32 buffer on the hidden device")
    if not bool(torch.isfinite(hidden).all()):
        raise FloatingPointError("card08 clean hidden state is nonfinite")
    raw_counter = float(counter.detach().item())
    if (
        not math.isfinite(raw_counter)
        or raw_counter < 0
        or raw_counter != int(raw_counter)
        or raw_counter >= MAX_EXACT_FLOAT32_COUNTER
    ):
        raise ValueError("card08 perturbation counter is outside its exact FP32 range")

    stream_seed = (seed * 0x9E3779B97F4A7C15 + (int(raw_counter) + 1) * 0xBF58476D1CE4E5B9) & (
        (1 << 63) - 1
    )
    generator = torch.Generator(device=hidden.device)
    generator.manual_seed(stream_seed)
    noise = torch.randn(
        hidden.shape, dtype=torch.float32, device=hidden.device, generator=generator
    )
    rms = hidden.detach().square().mean(dim=(0, 2), keepdim=True).sqrt().clamp_min(HIDDEN_RMS_FLOOR)
    with torch.no_grad():
        counter.add_(1.0)
    perturbed = hidden + NOISE_STD_FRACTION * rms * noise
    if not bool(torch.isfinite(perturbed).all()):
        raise FloatingPointError("card08 perturbed hidden state is nonfinite")
    return perturbed
