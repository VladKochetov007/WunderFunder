"""Fixed R7 MLP, transforms, paired initialization and prediction helpers."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import Tensor, nn

INPUT_WIDTH = 224
HIDDEN_WIDTH = 64
SEED = 19031
BATCH_SIZE = 512
EPOCHS = 12
LEARNING_RATE = 0.001
BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.0001
LOGVAR_MIN = -4.0
LOGVAR_MAX = 4.0


class MeanVarianceMLP(nn.Module):
    """One shared 224→64 GELU trunk with separate two-target output heads."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(INPUT_WIDTH, HIDDEN_WIDTH)
        self.mean_head = nn.Linear(HIDDEN_WIDTH, 2)
        self.logvar_head = nn.Linear(HIDDEN_WIDTH, 2)

    def _hidden(self, features: Tensor) -> Tensor:
        """Compute the shared fixed-width hidden representation."""
        return torch.nn.functional.gelu(self.trunk(features.float()))

    def forward_mean(self, features: Tensor) -> Tensor:
        """Compute only the mean head, leaving the control variance head unused."""
        return self.mean_head(self._hidden(features))

    def forward_all(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Compute raw mean and logvariance outputs for candidate/evaluation."""
        hidden = self._hidden(features)
        return self.mean_head(hidden), self.logvar_head(hidden)


def fit_normalizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit the shared feature normalizer using only supplied fitting rows."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != INPUT_WIDTH or not np.isfinite(values).all():
        raise ValueError("expected finite fit-only features with shape (N,224)")
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def apply_normalizer(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply already fit-only statistics and return contiguous FP32 values."""
    values = np.asarray(features, dtype=np.float32)
    center = np.asarray(mean, dtype=np.float32)
    spread = np.asarray(scale, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != INPUT_WIDTH or center.shape != (INPUT_WIDTH,):
        raise ValueError("normalizer dimensions do not match 224 input features")
    if spread.shape != center.shape or not np.isfinite(values).all():
        raise ValueError("invalid normalizer scale or nonfinite features")
    if not np.isfinite(center).all() or not np.isfinite(spread).all() or (spread <= 0).any():
        raise ValueError("normalizer statistics must be finite with positive scales")
    normalized = (values - center) / spread
    if not np.isfinite(normalized).all():
        raise ValueError("normalized features are nonfinite")
    return np.ascontiguousarray(normalized, dtype=np.float32)


def bounded_gaussian_loss(mean: Tensor, raw_logvar: Tensor, target: Tensor) -> Tensor:
    """Return twice the fixed clipped-target, |y|-weighted Gaussian NLL."""
    if mean.shape != target.shape or raw_logvar.shape != target.shape:
        raise ValueError("mean, logvariance, and target shapes must match")
    y = target.float().clamp(-2.0, 2.0)
    s = raw_logvar.float().clamp(LOGVAR_MIN, LOGVAR_MAX)
    weight = y.abs()
    numerator = (weight * (torch.exp(-s) * (mean.float() - y).square() + s)).sum()
    return numerator / weight.sum().clamp_min(1e-8)


def epoch_permutation(size: int, epoch: int) -> Tensor:
    """Return the fixed seed-plus-epoch row permutation shared by both arms."""
    if size < 1 or epoch < 0:
        raise ValueError("permutation size must be positive and epoch nonnegative")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED + epoch)
    return torch.randperm(size, generator=generator)


def state_fingerprint(model: MeanVarianceMLP) -> str:
    """Hash a sorted model state without exposing its parameters in JSON."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def predict_rows(
    model: MeanVarianceMLP, features: np.ndarray, batch_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Predict mean and raw logvariance in bounded CPU batches."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != INPUT_WIDTH or batch_size < 1:
        raise ValueError("prediction input must have shape (N,224) and positive batch size")
    means: list[np.ndarray] = []
    logvars: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(values[start : start + batch_size]))
            mean, logvar = model.forward_all(batch)
            means.append(mean.cpu().numpy())
            logvars.append(logvar.cpu().numpy())
    if not means:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    return np.concatenate(means), np.concatenate(logvars)


def initialize_pair() -> tuple[MeanVarianceMLP, MeanVarianceMLP, dict[str, str]]:
    """Build byte-identical arms from one seeded initialization and zero logvar head."""
    torch.manual_seed(SEED)
    initial = MeanVarianceMLP()
    with torch.no_grad():
        initial.logvar_head.weight.zero_()
        initial.logvar_head.bias.zero_()
    control, candidate = MeanVarianceMLP(), MeanVarianceMLP()
    state = {key: value.detach().clone() for key, value in initial.state_dict().items()}
    control.load_state_dict(state)
    candidate.load_state_dict(state)
    fingerprints = {
        "initial": state_fingerprint(initial),
        "control_initial": state_fingerprint(control),
        "candidate_initial": state_fingerprint(candidate),
    }
    if len(set(fingerprints.values())) != 1:
        raise RuntimeError("paired models do not share byte-identical initialization")
    return control, candidate, fingerprints
