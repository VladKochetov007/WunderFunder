"""Causal scoring-propensity network P(is_scored | rows <= t)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wnn.resources import configure_strict_fp32

LEGACY_PROFILE = "legacy_autocast_v1"
STRICT_FP32_PROFILE = "strict_fp32_v1"
PROPENSITY_PROFILES = (LEGACY_PROFILE, STRICT_FP32_PROFILE)
PROPENSITY_FIT_SOURCE_FILES = (
    "scripts/train_propensity.py",
    "scripts/train_rnn.py",
    "src/wnn/data.py",
    "src/wnn/features.py",
    "src/wnn/portability.py",
    "src/wnn/propensity.py",
    "src/wnn/resources.py",
    "src/wnn/splits.py",
    "src/wnn/training/checkpoints.py",
    "src/wnn/training/compatibility.py",
    "src/wnn/training/datasets.py",
    "src/wnn/training/environment.py",
    "src/wnn/training/inputs.py",
)


class PropensityNet(nn.Module):
    def __init__(self, n_in: int, hidden: int = 64):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("std", torch.ones(n_in))
        self.inp = nn.Linear(n_in, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x, h=None):
        z = F.gelu(self.inp((x - self.mean) / self.std))
        o, h = self.gru(z, h)
        return self.out(torch.cat([o, z], -1)).squeeze(-1), h


@torch.no_grad()
def propensity(model, inputs, chunk: int = 4000, *, profile: str | None = None):
    """Probabilities (B, T) for already-built inputs, streaming over chunks."""
    profile = profile or getattr(model, "_propensity_profile", LEGACY_PROFILE)
    if profile not in PROPENSITY_PROFILES:
        raise ValueError(f"unknown propensity precision profile: {profile}")
    if profile == STRICT_FP32_PROFILE and (inputs.dtype != torch.float32):
        raise ValueError("strict propensity inference requires float32 inputs")
    if profile == STRICT_FP32_PROFILE and not torch.isfinite(inputs).all().item():
        raise FloatingPointError("strict propensity inputs contain non-finite values")
    backend = _strict_fp32_backend() if profile == STRICT_FP32_PROFILE else _unchanged_backend()
    with backend:
        h, outs = None, []
        for s in range(0, inputs.shape[1], chunk):
            with torch.autocast(
                inputs.device.type,
                dtype=torch.float16,
                enabled=inputs.is_cuda and profile == LEGACY_PROFILE,
            ):
                logit, h = model(inputs[:, s : s + chunk], h)
            if profile == STRICT_FP32_PROFILE and not torch.isfinite(logit).all().item():
                raise FloatingPointError("strict propensity logits contain non-finite values")
            outs.append(torch.sigmoid(logit.float()))
        result = torch.cat(outs, 1)
        if profile == STRICT_FP32_PROFILE and (
            not torch.isfinite(result).all().item() or torch.any((result < 0) | (result > 1)).item()
        ):
            raise FloatingPointError("strict propensity output is not finite probability data")
        return result


@contextmanager
def _unchanged_backend():
    yield


@contextmanager
def _strict_fp32_backend():
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    configure_strict_fp32()
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def ordered_groups_sha256(groups: Iterable[int]) -> str:
    """Hash row-group identities as ordered little-endian int64 values."""
    values = np.asarray(list(map(int, groups)), dtype="<i8")
    if values.ndim != 1 or len(set(values.tolist())) != len(values):
        raise ValueError("row groups must be a unique one-dimensional sequence")
    return hashlib.sha256(values.tobytes()).hexdigest()


def fit_normalization(
    model: PropensityNet,
    batches: Iterable[torch.Tensor],
    *,
    std_floor: float = 1e-3,
) -> dict[str, object]:
    """Fit population mean/std by ordered float64 Welford merges over fit batches.

    Statistics use float64 accumulation for stability; model buffers remain float32.
    The caller controls membership and ordering by supplying the batch iterable.
    """
    if std_floor <= 0:
        raise ValueError("std_floor must be positive")
    count = 0
    mean = m2 = None
    for inputs in batches:
        if inputs.ndim < 2 or inputs.shape[-1] != model.mean.numel():
            raise ValueError("normalization batch has the wrong feature width")
        if inputs.dtype != torch.float32:
            raise ValueError("strict normalization requires float32 feature batches")
        if not torch.isfinite(inputs).all().item():
            raise FloatingPointError("strict normalization inputs contain non-finite values")
        flat = inputs.reshape(-1, inputs.shape[-1]).to(dtype=torch.float64)
        batch_count = int(flat.shape[0])
        if batch_count == 0:
            continue
        batch_mean = flat.mean(0)
        batch_m2 = (flat - batch_mean).square().sum(0)
        if mean is None:
            count, mean, m2 = batch_count, batch_mean, batch_m2
            continue
        delta = batch_mean - mean
        total = count + batch_count
        m2 = m2 + batch_m2 + delta.square() * (count * batch_count / total)
        mean = mean + delta * (batch_count / total)
        count = total
    if count == 0 or mean is None or m2 is None:
        raise ValueError("cannot fit propensity normalization from no rows")
    mean32 = mean.to(device=model.mean.device, dtype=torch.float32)
    std32 = (
        (m2 / count).sqrt().clamp_min(std_floor).to(device=model.std.device, dtype=torch.float32)
    )
    if not torch.isfinite(mean32).all().item() or not torch.isfinite(std32).all().item():
        raise FloatingPointError("strict normalization statistics are not finite float32 values")
    model.mean.copy_(mean32)
    model.std.copy_(std32)
    return {
        "method": "ordered_fit_population_welford_float64_v1",
        "accumulator_dtype": "float64",
        "buffer_dtype": "float32",
        "count": count,
        "std_floor": std_floor,
        "mean_sha256": _tensor_sha256(mean32),
        "std_sha256": _tensor_sha256(std32),
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().numpy().astype("<f4", copy=False).tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def load_propensity(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = PropensityNet(ck["n_in"], ck["hidden"])
    model.load_state_dict(ck["model"])
    provenance = ck.get("provenance", {})
    profile = provenance.get("precision_profile", LEGACY_PROFILE)
    if profile not in PROPENSITY_PROFILES:
        raise ValueError(f"unknown propensity checkpoint precision profile: {profile}")
    model._propensity_profile = profile
    model._propensity_provenance = provenance
    return model.to(device).eval()
