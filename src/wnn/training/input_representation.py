from __future__ import annotations

import hashlib
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from attrs import frozen

RAW_WIDTH = 112
CODE_WIDTH = 16
TOP_K = 4
SLOT_WIDTH = 128
EXTRA_WIDTH = 2 * SLOT_WIDTH
REPRESENTATION_MODES = ("base", "ema", "sparse", "ema_sparse")
HALF_LIFE = 32
EMA_ALPHA = 1.0 - 0.5 ** (1.0 / HALF_LIFE)


@frozen
class FittedInputRepresentation:
    """Frozen fitting-only dictionary and normalization used by appended input slots."""

    raw_mean: torch.Tensor
    raw_scale: torch.Tensor
    encoder_weight: torch.Tensor
    decoder_weight: torch.Tensor
    fit_metadata: dict[str, object]

    def on_device(self, device: torch.device) -> FittedInputRepresentation:
        """Move the small frozen representation state to its model's device."""
        return FittedInputRepresentation(
            raw_mean=self.raw_mean.to(device=device, dtype=torch.float32),
            raw_scale=self.raw_scale.to(device=device, dtype=torch.float32),
            encoder_weight=self.encoder_weight.to(device=device, dtype=torch.float32),
            decoder_weight=self.decoder_weight.to(device=device, dtype=torch.float32),
            fit_metadata=dict(self.fit_metadata),
        )


def representation_input_width(mode: str | None) -> int:
    """Return the appended input width, preserving the legacy omitted-flag layout."""
    if mode is None or mode == "none":
        return 0
    if mode not in REPRESENTATION_MODES:
        raise ValueError(f"unknown input representation mode: {mode!r}")
    return EXTRA_WIDTH


@torch.no_grad()
def build_representation_features(
    raw: torch.Tensor, mode: str, fitted: FittedInputRepresentation
) -> torch.Tensor:
    """Build the fixed 256-wide sparse/residual and causal-history suffix."""
    with torch.autocast(device_type=raw.device.type, enabled=False):
        return _build_representation_features_fp32(raw, mode, fitted)


@torch.no_grad()
def _build_representation_features_fp32(
    raw: torch.Tensor, mode: str, fitted: FittedInputRepresentation
) -> torch.Tensor:
    """Apply the frozen transform with autocast disabled and FP32 intermediates."""
    representation_input_width(mode)
    if raw.ndim != 3 or raw.shape[-1] != RAW_WIDTH:
        raise ValueError("representation inputs must have shape (batch,time,112)")
    if not torch.isfinite(raw).all().item():
        raise FloatingPointError("nonfinite raw input to frozen representation")
    batch_size, rows, _ = raw.shape
    raw_fp32 = raw.to(dtype=torch.float32)
    output = torch.empty((batch_size, rows, EXTRA_WIDTH), device=raw.device, dtype=torch.float32)
    if mode in ("sparse", "ema_sparse"):
        normalized = (raw_fp32 - fitted.raw_mean.view(1, 1, -1)) / fitted.raw_scale.view(1, 1, -1)
        code = F.linear(normalized, fitted.encoder_weight)
        order = torch.argsort(code.abs(), dim=-1, descending=True, stable=True)[..., :TOP_K]
        mask = torch.zeros_like(code).scatter_(-1, order, 1.0)
        code = code * mask.detach()
        residual = normalized - F.linear(code, fitted.decoder_weight)
        output[..., :CODE_WIDTH].copy_(code)
        output[..., CODE_WIDTH:SLOT_WIDTH].copy_(residual)
    elif mode == "ema":
        output[..., :SLOT_WIDTH].zero_()
        causal_signed_ema(
            raw_fp32,
            output=output[..., SLOT_WIDTH : SLOT_WIDTH + RAW_WIDTH],
        )
        output[..., SLOT_WIDTH + RAW_WIDTH :].zero_()
    else:
        output[..., :SLOT_WIDTH].zero_()
    if mode in ("ema", "ema_sparse"):
        if mode == "ema_sparse":
            causal_signed_ema(output[..., :SLOT_WIDTH], output=output[..., SLOT_WIDTH:])
    else:
        output[..., SLOT_WIDTH:].zero_()
    if not torch.isfinite(output).all().item():
        raise FloatingPointError("nonfinite appended representation features")
    return output


@torch.no_grad()
def causal_signed_ema(
    values: torch.Tensor,
    block_rows: int = 256,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute a reset-local FP32 EMA before recurrent chunking."""
    with torch.autocast(device_type=values.device.type, enabled=False):
        return _causal_signed_ema_fp32(values, block_rows, output)


def _causal_signed_ema_fp32(
    values: torch.Tensor, block_rows: int, output: torch.Tensor | None
) -> torch.Tensor:
    """Evaluate a block-cumsum EMA with a FP32 carry between blocks."""
    if values.ndim != 3 or values.shape[1] < 1 or block_rows < 1:
        raise ValueError("EMA requires a nonempty batch/time/features tensor")
    batch_size, row_count, width = values.shape
    source = values.to(dtype=torch.float32)
    if output is None:
        result = torch.empty(
            (batch_size, row_count, width), device=values.device, dtype=torch.float32
        )
    else:
        if (
            output.shape != (batch_size, row_count, width)
            or output.dtype != torch.float32
            or output.device != values.device
        ):
            raise ValueError("EMA output must be a matching FP32 tensor on the input device")
        result = output
    result.zero_()
    alpha = torch.tensor(EMA_ALPHA, device=values.device, dtype=torch.float32)
    beta = torch.tensor(1.0 - EMA_ALPHA, device=values.device, dtype=torch.float32)
    previous = torch.zeros((batch_size, width), device=values.device, dtype=torch.float32)
    for start in range(1, row_count, block_rows):
        stop = min(start + block_rows, row_count)
        delta = source[:, start:stop] - source[:, start - 1 : stop - 1]
        powers = torch.pow(
            beta,
            torch.arange(stop - start, device=values.device, dtype=torch.float32),
        )
        expanded = powers.view(1, -1, 1)
        local = alpha * expanded * torch.cumsum(delta / expanded, dim=1)
        carried = beta * expanded * previous.unsqueeze(1)
        local = local + carried
        result[:, start:stop] = local
        previous = local[:, -1]
    if (
        result.shape != (batch_size, row_count, width)
        or result.dtype != torch.float32
        or not torch.isfinite(result).all().item()
    ):
        raise FloatingPointError("causal EMA output is invalid")
    return result


def extend_input_projection_state(
    parent_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    parent_width: int = RAW_WIDTH + 99,
) -> dict[str, torch.Tensor]:
    """Append zero-effect input columns while preserving an initialized parent function."""
    result = dict(parent_state)
    target_width = parent_width + EXTRA_WIDTH
    source_weight = parent_state.get("inp.weight")
    target_weight = target_state.get("inp.weight")
    if source_weight is None or target_weight is None:
        raise ValueError("representation extension requires plain `inp.weight` tensors")
    if source_weight.ndim != 2 or target_weight.ndim != 2:
        raise ValueError("input projection weights must be matrices")
    if source_weight.shape[0] != target_weight.shape[0]:
        raise ValueError("input projection output width changed during representation extension")
    if source_weight.shape[1] == target_weight.shape[1] == target_width:
        return result
    if source_weight.shape[1] != parent_width or target_weight.shape[1] != target_width:
        raise ValueError("input projection widths do not match the 211-to-467 extension")
    expanded_weight = torch.zeros_like(target_weight)
    expanded_weight[:, :parent_width] = source_weight.to(expanded_weight)
    result["inp.weight"] = expanded_weight
    for name in ("mean", "std"):
        source = parent_state.get(name)
        target = target_state.get(name)
        if source is None or target is None or source.ndim != 1 or target.ndim != 1:
            raise ValueError(f"representation extension requires one-dimensional {name!r} buffers")
        if source.numel() != parent_width or target.numel() != target_width:
            raise ValueError(f"{name} buffer width does not match the 211-to-467 extension")
        expanded = torch.zeros_like(target) if name == "mean" else torch.ones_like(target)
        expanded[:parent_width] = source.to(expanded)
        result[name] = expanded
    return result


def serialize_input_representation(fitted: FittedInputRepresentation) -> dict[str, object]:
    """Create a weights-only-safe checkpoint payload for inference reconstruction."""
    return {
        "schema": "wnn.fitted_input_representation.v1",
        "raw_mean": fitted.raw_mean.detach().cpu().to(torch.float32).contiguous(),
        "raw_scale": fitted.raw_scale.detach().cpu().to(torch.float32).contiguous(),
        "encoder_weight": fitted.encoder_weight.detach().cpu().to(torch.float32).contiguous(),
        "decoder_weight": fitted.decoder_weight.detach().cpu().to(torch.float32).contiguous(),
        "fit_metadata": dict(fitted.fit_metadata),
        "state_sha256": input_representation_sha256(fitted),
    }


def deserialize_input_representation(
    payload: object, device: torch.device | str = "cpu"
) -> FittedInputRepresentation:
    """Validate and restore the fitted transform from a model checkpoint."""
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "wnn.fitted_input_representation.v1"
    ):
        raise ValueError("checkpoint lacks a supported fitted input representation")
    names = ("raw_mean", "raw_scale", "encoder_weight", "decoder_weight")
    values = {name: payload.get(name) for name in names}
    if any(not isinstance(value, torch.Tensor) for value in values.values()):
        raise ValueError("checkpoint representation tensors are incomplete")
    mean, scale, encoder, decoder = (values[name] for name in names)
    if (
        tuple(mean.shape) != (RAW_WIDTH,)
        or tuple(scale.shape) != (RAW_WIDTH,)
        or tuple(encoder.shape) != (CODE_WIDTH, RAW_WIDTH)
        or tuple(decoder.shape) != (RAW_WIDTH, CODE_WIDTH)
    ):
        raise ValueError("checkpoint representation tensor dimensions are invalid")
    if any(value.dtype != torch.float32 for value in (mean, scale, encoder, decoder)):
        raise ValueError("checkpoint representation state must use FP32 tensors")
    if not all(torch.isfinite(value).all().item() for value in (mean, scale, encoder, decoder)):
        raise ValueError("checkpoint representation contains nonfinite values")
    if (scale <= 0).any().item():
        raise ValueError("checkpoint raw input scales must be positive")
    metadata = payload.get("fit_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint representation fit metadata is invalid")
    state = FittedInputRepresentation(mean, scale, encoder, decoder, metadata).on_device(
        torch.device(device)
    )
    expected = payload.get("state_sha256")
    if expected != input_representation_sha256(state):
        raise ValueError("checkpoint representation state digest is invalid")
    return state


def input_representation_sha256(fitted: FittedInputRepresentation) -> str:
    """Hash the exact frozen dictionary and raw normalizer tensors."""
    digest = hashlib.sha256()
    for name, tensor in (
        ("raw_mean", fitted.raw_mean.to(dtype=torch.float32)),
        ("raw_scale", fitted.raw_scale.to(dtype=torch.float32)),
        ("encoder_weight", fitted.encoder_weight.to(dtype=torch.float32)),
        ("decoder_weight", fitted.decoder_weight.to(dtype=torch.float32)),
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()
