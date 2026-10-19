"""Fixed-recipe sparse dictionary fitting for experimental input representations."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from wnn.data import WARMUP
from wnn.training.input_representation import CODE_WIDTH, RAW_WIDTH

COMPONENT_SEED = 19081
DICTIONARY_SEED = 19082
ORDER_SEED = 19083
DICTIONARY_EPOCHS = 12
DICTIONARY_BATCH_SIZE = 512
DICTIONARY_LR = 1e-3
DICTIONARY_WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
STRIDE = 40
FIT_COMPONENTS = 64
ARTIFACT_SCHEMA = "wnn.input_representation_fit.v1"
REPRESENTATION_RECIPE = {
    "version": 1,
    "component_selection_seed": COMPONENT_SEED,
    "fit_components": FIT_COMPONENTS,
    "row_stride": STRIDE,
    "warmup_rows": WARMUP,
    "dictionary_initialization_seed": DICTIONARY_SEED,
    "epoch_order_seed": ORDER_SEED,
    "dictionary_epochs": DICTIONARY_EPOCHS,
    "batch_size": DICTIONARY_BATCH_SIZE,
    "optimizer": "AdamW",
    "learning_rate": DICTIONARY_LR,
    "weight_decay": DICTIONARY_WEIGHT_DECAY,
    "gradient_clip_norm": GRAD_CLIP,
    "raw_width": RAW_WIDTH,
    "code_width": CODE_WIDTH,
    "top_k": 4,
    "dictionary_dtype": "float32",
    "normalizer_accumulation": "float64_cpu",
    "normalizer_storage_and_application": "float32",
}


class _SparseDictionary(nn.Module):
    """Bias-free 112→16→112 tied-start dictionary with stable signed top-four codes."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder_weight = nn.Parameter(torch.empty(CODE_WIDTH, RAW_WIDTH))
        self.decoder_weight = nn.Parameter(torch.empty(RAW_WIDTH, CODE_WIDTH))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return stable top-four signed codes and their linear reconstruction."""
        code = F.linear(inputs, self.encoder_weight)
        selected = torch.argsort(code.abs(), dim=1, descending=True, stable=True)[:, :4]
        mask = torch.zeros_like(code).scatter_(1, selected, 1.0)
        code = code * mask.detach()
        return code, F.linear(code, self.decoder_weight)

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        """Project dictionary columns to unit length after each fixed update."""
        norms = torch.linalg.vector_norm(self.decoder_weight, dim=0)
        if not torch.isfinite(norms).all().item() or (norms <= 1e-12).any().item():
            raise FloatingPointError("dictionary decoder has a nonfinite or collapsed column")
        self.decoder_weight.div_(norms.unsqueeze(0))


def representation_recipe_sha256() -> str:
    """Hash the fixed fold dictionary recipe accepted by cache and checkpoint loads."""
    encoded = json.dumps(REPRESENTATION_RECIPE, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def raw_normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute fit-only float64 population moments in bounded row chunks."""
    if values.ndim != 2 or values.shape[1] != RAW_WIDTH or not np.isfinite(values).all():
        raise ValueError("raw normalizer expects finite fit rows shaped (N,112)")
    total = np.zeros(RAW_WIDTH, dtype=np.float64)
    total_sq = np.zeros(RAW_WIDTH, dtype=np.float64)
    for start in range(0, len(values), 16_000):
        block = values[start : start + 16_000].astype(np.float64, copy=False)
        total += block.sum(axis=0, dtype=np.float64)
        total_sq += np.square(block).sum(axis=0, dtype=np.float64)
    mean = total / len(values)
    variance = np.maximum(total_sq / len(values) - np.square(mean), 0.0)
    scale = np.maximum(np.sqrt(variance), 1e-6)
    return mean, scale


def _initialize_dictionary(seed: int) -> _SparseDictionary:
    """Initialize encoder/decoder from one normalized Gaussian decoder draw."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _SparseDictionary()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    decoder = torch.randn((RAW_WIDTH, CODE_WIDTH), generator=generator)
    decoder /= torch.linalg.vector_norm(decoder, dim=0, keepdim=True)
    with torch.no_grad():
        model.encoder_weight.copy_(decoder.T)
        model.decoder_weight.copy_(decoder)
    return model


def fit_dictionary(values: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Fit a frozen top-four dictionary for the fixed twelve-epoch recipe."""
    inputs = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    model = _initialize_dictionary(DICTIONARY_SEED)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=DICTIONARY_LR, weight_decay=DICTIONARY_WEIGHT_DECAY
    )
    attempted = 0
    losses: list[float] = []
    for epoch in range(DICTIONARY_EPOCHS):
        generator = torch.Generator(device="cpu").manual_seed(ORDER_SEED + epoch)
        order = torch.randperm(len(inputs), generator=generator)
        for start in range(0, len(order), DICTIONARY_BATCH_SIZE):
            batch = inputs.index_select(0, order[start : start + DICTIONARY_BATCH_SIZE])
            optimizer.zero_grad(set_to_none=True)
            _, reconstruction = model(batch)
            loss = torch.mean((reconstruction - batch).square())
            if not torch.isfinite(loss).item():
                raise FloatingPointError("nonfinite dictionary reconstruction loss")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
                for parameter in model.parameters()
            ):
                raise FloatingPointError("nonfinite dictionary gradient")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if not torch.isfinite(norm).item():
                raise FloatingPointError("nonfinite dictionary preclip gradient norm")
            optimizer.step()
            model.normalize_decoder_columns()
            if any(not torch.isfinite(parameter).all().item() for parameter in model.parameters()):
                raise FloatingPointError("nonfinite dictionary parameter after update")
            if any(
                isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and not torch.isfinite(value).all().item()
                for state in optimizer.state.values()
                for value in state.values()
            ):
                raise FloatingPointError("nonfinite dictionary optimizer state after update")
            attempted += 1
            losses.append(float(loss.detach()))
    expected_updates = DICTIONARY_EPOCHS * int(np.ceil(len(inputs) / DICTIONARY_BATCH_SIZE))
    if attempted != expected_updates:
        raise RuntimeError("dictionary update count differs from the fixed schedule")
    record = {
        "initialization_seed": DICTIONARY_SEED,
        "epoch_order_seed": ORDER_SEED,
        "epochs": DICTIONARY_EPOCHS,
        "batch_size": DICTIONARY_BATCH_SIZE,
        "optimizer": "AdamW",
        "learning_rate": DICTIONARY_LR,
        "weight_decay": DICTIONARY_WEIGHT_DECAY,
        "gradient_clip_norm": GRAD_CLIP,
        "attempted_and_successful_updates": attempted,
        "mean_training_batch_reconstruction_mse": float(np.mean(losses)),
        "fit_row_count": len(inputs),
    }
    return model.encoder_weight.detach().clone(), model.decoder_weight.detach().clone(), record
