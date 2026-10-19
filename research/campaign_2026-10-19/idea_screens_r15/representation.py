"""Deterministic tied dictionary initialization and full-history representations."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import Tensor, nn

RAW_WIDTH = 112
CODE_WIDTH = 16
TOP_K = 4
FEATURE_WIDTH = 256
HALF_LIFE = 32
ALPHA = 1.0 - 0.5 ** (1.0 / HALF_LIFE)


class DictionaryAutoencoder(nn.Module):
    """Bias-free 112→16→112 dictionary with deterministic signed top-k codes."""

    def __init__(self, sparse: bool) -> None:
        super().__init__()
        self.encoder = nn.Linear(RAW_WIDTH, CODE_WIDTH, bias=False)
        self.decoder = nn.Linear(CODE_WIDTH, RAW_WIDTH, bias=False)
        self.sparse = sparse

    def codes(self, values: Tensor) -> Tensor:
        """Return dense or stable signed top-four code values."""
        code = self.encoder(values)
        if not self.sparse:
            return code
        indices = torch.argsort(code.abs(), dim=1, descending=True, stable=True)[:, :TOP_K]
        mask = torch.zeros_like(code).scatter_(1, indices, 1.0)
        return code * mask.detach()

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return code, reconstruction, and exactly retained input residual."""
        code = self.codes(values)
        reconstructed = self.decoder(code)
        return code, reconstructed, values - reconstructed

    def normalize_decoder_columns(self) -> None:
        """Project decoder columns to unit L2 norm, rejecting collapsed columns."""
        with torch.no_grad():
            norms = torch.linalg.vector_norm(self.decoder.weight, dim=0)
            if not torch.isfinite(norms).all().item() or (norms <= 1e-12).any().item():
                raise FloatingPointError("decoder has a nonfinite or collapsed column")
            self.decoder.weight.div_(norms.unsqueeze(0))


class R15Predictor(nn.Module):
    """Fixed 256→64 GELU mean predictor with an unused zero variance head."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Linear(FEATURE_WIDTH, 64)
        self.mean_head = nn.Linear(64, 2)
        self.logvar_head = nn.Linear(64, 2)

    def _hidden(self, features: Tensor) -> Tensor:
        return torch.nn.functional.gelu(self.trunk(features.float()))

    def forward_mean(self, features: Tensor) -> Tensor:
        return self.mean_head(self._hidden(features))

    def forward_all(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self._hidden(features)
        return self.mean_head(hidden), self.logvar_head(hidden)


def fingerprint(module: nn.Module) -> str:
    """Hash module state tensors without serializing parameter values to JSON."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialize_dictionary_pair(seed: int) -> tuple[DictionaryAutoencoder, DictionaryAutoencoder]:
    """Create dense/sparse dictionaries from one normalized decoder draw."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    decoder = torch.randn((RAW_WIDTH, CODE_WIDTH), generator=generator)
    decoder /= torch.linalg.vector_norm(decoder, dim=0, keepdim=True)
    state = {"encoder.weight": decoder.T.contiguous(), "decoder.weight": decoder.contiguous()}
    dense, sparse = DictionaryAutoencoder(False), DictionaryAutoencoder(True)
    dense.load_state_dict(state)
    sparse.load_state_dict(state)
    return dense, sparse


def fit_normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit float64 population moments and apply the frozen 1e-6 scale floor."""
    matrix = np.asarray(values)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("normalizer requires a finite two-dimensional fit matrix")
    mean = matrix.mean(axis=0, dtype=np.float64)
    scale = np.maximum(matrix.std(axis=0, dtype=np.float64), 1e-6)
    return mean, scale


def build_sequence_features(
    raw: np.ndarray, raw_mean: np.ndarray, raw_scale: np.ndarray, model: DictionaryAutoencoder
) -> tuple[np.ndarray, dict[str, object]]:
    """Build reset-local 256-wide dictionary/residual plus signed-EMA history."""
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != RAW_WIDTH or not np.isfinite(values).all():
        raise ValueError("expected finite complete (T,112) canonical F0 inputs")
    normalized = ((values.astype(np.float64) - raw_mean) / raw_scale).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise FloatingPointError("nonfinite normalized dictionary input")
    base = np.empty((len(values), RAW_WIDTH + CODE_WIDTH), dtype=np.float32)
    active = np.zeros(CODE_WIDTH, dtype=np.int64)
    reconstruction_sq = 0.0
    residual_identity_max_abs = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 2048):
            stop = min(start + 2048, len(values))
            batch = torch.from_numpy(np.ascontiguousarray(normalized[start:stop]))
            codes, rebuilt, residual = model(batch)
            if (
                not torch.isfinite(rebuilt).all().item()
                or not torch.isfinite(residual).all().item()
            ):
                raise FloatingPointError("nonfinite dictionary representation")
            identity_error = (batch - (rebuilt + residual)).abs().max().item()
            residual_identity_max_abs = max(residual_identity_max_abs, float(identity_error))
            base[start:stop, :CODE_WIDTH] = codes.numpy()
            base[start:stop, CODE_WIDTH:] = residual.numpy()
            active += (codes != 0).sum(dim=0).cpu().numpy()
            reconstruction_sq += float((rebuilt - batch).double().square().sum().item())
    features = np.empty((len(values), FEATURE_WIDTH), dtype=np.float32)
    features[:, : RAW_WIDTH + CODE_WIDTH] = base
    net = np.zeros(base.shape[1], dtype=np.float64)
    previous = base[0].astype(np.float64)
    for row in range(1, len(base)):
        current = base[row].astype(np.float64)
        delta = current - previous
        net = (1.0 - ALPHA) * net + ALPHA * delta
        features[row, RAW_WIDTH + CODE_WIDTH :] = net.astype(np.float32)
        previous = current
    # Row zero is a reset boundary: its signed innovation EMA is exactly zero.
    features[0, RAW_WIDTH + CODE_WIDTH :] = 0.0
    metadata = {
        "rows": int(len(values)),
        "active_code_count_by_factor": active.tolist(),
        "reconstruction_mse": reconstruction_sq / (len(values) * RAW_WIDTH),
        "residual_identity_max_abs": residual_identity_max_abs,
        "ema_alpha": ALPHA,
        "ema_reset_at_sequence_start": True,
    }
    if metadata["residual_identity_max_abs"] > 1e-5:
        raise FloatingPointError("retained residual does not reconstruct normalized input")
    return features, metadata
