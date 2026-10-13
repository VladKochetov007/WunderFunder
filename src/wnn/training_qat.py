from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import numpy as np
import torch
from torch.func import functional_call

from wnn.nn import RNNModel
from wnn.quantization import (
    HEAD_Q8_PRESETS,
    PRESETS,
    bits_for_matrix,
    quantize_model,
    quantize_rows,
    quantize_rows_q4,
    represented_rows,
    represented_rows_q4,
)

QAT_MODES = ("none", *PRESETS)
QAT_LEARN_SCALE_MODES = ("recurrent_w4", "input_w4_recurrent_w8")

_FIXED_SCALE_TRAINING_RULE = "identity STE; detached row scales; no scale, rounding or clipping derivative"
_LEARNED_SCALE_TRAINING_RULE = (
    "identity weight STE; learned row scales with LSQ-inspired clipped scale derivative"
)


def scale_learning_metadata(
    optimizer: str | None = None,
    weight_decay: float | None = None,
) -> dict[str, object]:
    """Return the exact, versioned semantics for the optional LSQ-inspired arm."""
    metadata: dict[str, object] = {
        "version": 1,
        "algorithm": "lsq_inspired_per_output_row_step_sizes_v1",
        "initialization": "fixed_absmax_row_scale_same_initial_codes_and_representation_v1",
        "scale_gradient": "clipped LSQ derivative; per-row g=1/sqrt(input_columns*QP)",
        "weight_gradient": "identity STE, unchanged from fixed-scale QAT; not canonical LSQ clipped STE",
        "scale_parameter": "direct float32 row step; base-LR non-Muon vector group; configured --wd applies",
        "scale_projection": "reject nonfinite; negative step resets to current absmax/QP using min-subnormal floor; zero defers initialization",
        "zero_rows": "scale=0 while latent row is zero; initialize absmax/QP on first nonzero forward",
        "quantizer": "existing nearest-even symmetric Q4/Q8 row quantizer; QP is 7 or 127",
        "paper": "Esser et al., Learned Step Size Quantization, arXiv:1902.08153",
        "reproduction_status": "LSQ-inspired scale-gradient component, not a full LSQ reproduction",
    }
    if optimizer is not None or weight_decay is not None:
        if optimizer not in {"adamw", "muon", "sgd"}:
            raise ValueError("learned-scale metadata requires a supported optimizer")
        if weight_decay is None or not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("learned-scale metadata requires finite nonnegative weight decay")
        metadata["optimizer_group"] = {
            "optimizer": optimizer,
            "learning_rate_group": "base_lr_vector_adamw_group" if optimizer == "muon" else "base_lr",
            "weight_decay": float(weight_decay),
            "weight_decay_semantics": (
                "decoupled_adamw" if optimizer in {"adamw", "muon"} else "coupled_sgd"
            ),
            "muon_rule": "kept out of Muon matrix group; updated by AdamW" if optimizer == "muon" else None,
        }
    return metadata


class _RepresentedWeight(torch.autograd.Function):
    """Exact represented values forward; identity STE to the latent float matrix."""

    @staticmethod
    def forward(ctx: object, latent: torch.Tensor, represented: torch.Tensor) -> torch.Tensor:
        return represented

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient, None


class _LearnedScaleWeight(torch.autograd.Function):
    """Exact represented forward with identity weight STE and LSQ scale gradient."""

    @staticmethod
    def forward(
        ctx: object,
        latent: torch.Tensor,
        scale: torch.Tensor,
        represented: torch.Tensor,
        codes: torch.Tensor,
        bits: int,
    ) -> torch.Tensor:
        ctx.bits = bits
        ctx.save_for_backward(latent, scale, codes)
        return represented

    @staticmethod
    def backward(
        ctx: object, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
        latent, scale, codes = ctx.saved_tensors
        q_pos = float((1 << (ctx.bits - 1)) - 1)
        safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        ratio = latent / safe_scale[:, None]
        derivative = torch.where(
            ratio <= -q_pos,
            -q_pos,
            torch.where(ratio >= q_pos, q_pos, codes.to(ratio.dtype) - ratio),
        )
        normalization = 1.0 / (latent.shape[1] * q_pos) ** 0.5
        scale_gradient = (gradient * derivative).sum(dim=1) * normalization
        return gradient, scale_gradient, None, None, None


def qat_names(model: RNNModel, preset: str) -> tuple[str, ...]:
    """Select exactly the matrices described by the canonical weight-only presets."""
    if preset not in PRESETS:
        raise ValueError("unknown QAT preset")
    recurrent = tuple(
        sorted(
            name
            for name, _ in model.named_parameters()
            if name.startswith(("gru.weight_ih_", "gru.weight_hh_"))
        )
    )
    if not recurrent:
        raise ValueError("QAT requires recurrent matrices")
    return recurrent + (("head.0.weight",) if preset in HEAD_Q8_PRESETS else ())


def _matrix_blocks(name: str, value: torch.Tensor, hidden: int) -> tuple[torch.Tensor, ...]:
    if name != "head.0.weight":
        return (value,)
    return tuple(value.split((hidden, value.shape[1] - hidden), dim=1))


def _quantize_block_with_scales(
    values: np.ndarray, scales: np.ndarray, bits: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize with supplied row steps, initializing deferred all-zero rows."""
    values = np.asarray(values, dtype=np.float32)
    scales = np.asarray(scales)
    q_pos = (1 << (bits - 1)) - 1
    if (
        bits not in (4, 8)
        or values.ndim != 2
        or 0 in values.shape
        or not np.isfinite(values).all()
        or scales.dtype != np.float32
        or scales.shape != (values.shape[0],)
        or not np.isfinite(scales).all()
        or (scales < 0).any()
    ):
        raise ValueError("invalid learned-scale QAT block or row steps")
    maximum = np.max(np.abs(values), axis=1)
    resolved = scales.copy()
    deferred = (resolved == 0) & (maximum != 0)
    resolved[deferred] = np.maximum(
        maximum[deferred] / np.float32(q_pos),
        np.nextafter(np.float32(0), np.float32(1)),
    )
    safe_scales = np.where(resolved == 0, np.float32(1), resolved)
    ratios = values / safe_scales[:, None]
    codes = np.clip(np.rint(ratios), -q_pos, q_pos).astype(np.int8)
    represented = np.ascontiguousarray(codes.astype(np.float32) * resolved[:, None])
    return codes, resolved, represented


def configure_qat(model: RNNModel, preset: str, learn_scales: bool = False) -> None:
    """Enable weight-only QAT and optionally register learned row-step parameters."""
    if preset not in QAT_MODES:
        raise ValueError("unknown QAT mode")
    if learn_scales and preset not in QAT_LEARN_SCALE_MODES:
        raise ValueError("learned scales require a versioned Q4 QAT preset")
    if preset == "none":
        if learn_scales:
            raise ValueError("learned scales require QAT")
        return
    if getattr(model, "_recurrent_rank", 0):
        raise ValueError("combined QAT and factorized recurrence is not implemented")
    if type(model) is not RNNModel or type(model.gru) is not torch.nn.GRU or model.enc != "linear":
        raise ValueError("bounded QAT supports the plain GRU with linear input encoder")
    if any(value.dtype != torch.float32 for value in model.state_dict().values()):
        raise ValueError("QAT requires float32 parameters and normalization buffers")
    model._qat_preset = preset
    model._inference_fp32 = True
    model._qat_names = qat_names(model, preset)
    model._qat_bits = {name: bits_for_matrix(name, preset) for name in model._qat_names}
    model._qat_cache = {}
    model._qat_learn_scales = bool(learn_scales)
    if learn_scales:
        parameters = dict(model.named_parameters())
        scale_keys: list[tuple[str, int, int]] = []
        scale_values: list[torch.nn.Parameter] = []
        for name in model._qat_names:
            value = parameters[name]
            bits = model._qat_bits[name]
            quantizer = quantize_rows_q4 if bits == 4 else quantize_rows
            for index, block in enumerate(_matrix_blocks(name, value, model.inp.out_features)):
                _, initial_scale = quantizer(block.detach().cpu().numpy())
                scale_values.append(
                    torch.nn.Parameter(torch.from_numpy(initial_scale.copy()).to(value.device))
                )
                scale_keys.append((name, index, bits))
        model.add_module("_qat_scale_params", torch.nn.ParameterList(scale_values))
        model._qat_scale_keys = tuple(scale_keys)
        model._qat_scale_lookup = {
            (name, index): position
            for position, (name, index, _) in enumerate(model._qat_scale_keys)
        }


def invalidate_qat_cache(model: torch.nn.Module) -> None:
    """Discard deterministic represented matrices after a latent optimizer/EMA update."""
    if hasattr(model, "_qat_cache"):
        model._qat_cache.clear()


def project_qat_scales_(model: torch.nn.Module) -> None:
    """Keep learned steps finite and nonnegative after every optimizer attempt."""
    if not getattr(model, "_qat_learn_scales", False):
        return
    parameters = dict(model.named_parameters())
    hidden = model.inp.out_features
    with torch.no_grad():
        for position, (name, index, bits) in enumerate(model._qat_scale_keys):
            scale = model._qat_scale_params[position]
            if not torch.isfinite(scale).all().item():
                raise FloatingPointError(f"nonfinite learned QAT scale: {name}.{index}")
            negative = scale < 0
            if negative.any().item():
                block = _matrix_blocks(name, parameters[name], hidden)[index]
                # Match quantize_rows[_q4] exactly, including its positive
                # min-subnormal floor. torch.finfo(...).tiny is min-normal.
                maximum = np.max(np.abs(block.detach().cpu().numpy()), axis=1)
                q_pos = np.float32((1 << (bits - 1)) - 1)
                reset = np.maximum(
                    maximum / q_pos,
                    np.nextafter(np.float32(0), np.float32(1)),
                )
                reset = np.where(maximum == 0, np.float32(0), reset).astype(np.float32)
                reset_tensor = torch.from_numpy(reset).to(scale.device)
                scale.copy_(torch.where(negative, reset_tensor, scale))


def _initialize_deferred_rows_(value: torch.Tensor, scale: torch.Tensor, bits: int) -> None:
    current = scale.detach().cpu().numpy()
    if current.dtype != np.float32 or not np.isfinite(current).all() or (current < 0).any():
        raise ValueError("invalid learned-scale QAT row steps")
    if not (current == 0).any():
        return
    values = value.detach().cpu().numpy()
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("invalid learned-scale QAT latent block")
    maximum = np.max(np.abs(values), axis=1)
    q_pos = (1 << (bits - 1)) - 1
    resolved = current.copy()
    deferred = (resolved == 0) & (maximum != 0)
    resolved[deferred] = np.maximum(
        maximum[deferred] / np.float32(q_pos),
        np.nextafter(np.float32(0), np.float32(1)),
    )
    if not np.array_equal(resolved, current):
        with torch.no_grad():
            scale.copy_(torch.from_numpy(resolved).to(scale.device))


def _learned_scale_block(
    model: RNNModel,
    name: str,
    index: int,
    value: torch.Tensor,
    bits: int,
) -> torch.Tensor:
    position = model._qat_scale_lookup[(name, index)]
    scale = model._qat_scale_params[position]
    identity = (id(value), value._version, value.data_ptr(), id(scale), scale._version, scale.data_ptr())
    cache_key = (name, index)
    cached = model._qat_cache.get(cache_key)
    if cached is None or cached[0] != identity:
        _initialize_deferred_rows_(value, scale, bits)
        identity = (id(value), value._version, value.data_ptr(), id(scale), scale._version, scale.data_ptr())
        codes, _, represented = _quantize_block_with_scales(
            value.detach().cpu().numpy(), scale.detach().cpu().numpy(), bits
        )
        cached = (
            identity,
            torch.from_numpy(represented).to(value.device),
            torch.from_numpy(codes).to(value.device),
        )
        model._qat_cache[cache_key] = cached
    return _LearnedScaleWeight.apply(value, scale, cached[1], cached[2], bits)


def _represented_matrix(name: str, value: torch.Tensor, hidden: int, bits: int = 8) -> torch.Tensor:
    # Canonical CPU arithmetic deliberately handles subnormals and ties exactly;
    # CUDA fast math may flush subnormal scales. Once-per-update caching bounds
    # the transfer cost while a fresh STE node permits every truncated backward.
    if bits not in (4, 8):
        raise ValueError("QAT matrix bit width must be 4 or 8")
    weights = value.detach().cpu().numpy()
    blocks = np.split(weights, [hidden], axis=1) if name == "head.0.weight" else [weights]
    quantizer = quantize_rows_q4 if bits == 4 else quantize_rows
    restorer = represented_rows_q4 if bits == 4 else represented_rows
    represented = np.concatenate([restorer(*quantizer(block)) for block in blocks], axis=1)
    return torch.from_numpy(np.ascontiguousarray(represented)).to(value.device)


def qat_forward(
    model: RNNModel,
    inputs: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run exact weight-only fake quantization, keeping activations and state float32."""
    if inputs.dtype != torch.float32 or state is not None and state.dtype != torch.float32:
        raise ValueError("QAT inputs and recurrent state must remain float32")
    parameters = dict(model.named_parameters())
    replacements = {}
    for name in model._qat_names:
        value = parameters[name]
        if getattr(model, "_qat_learn_scales", False):
            blocks = _matrix_blocks(name, value, model.inp.out_features)
            replacements[name] = torch.cat(
                [
                    _learned_scale_block(model, name, index, block, model._qat_bits[name])
                    for index, block in enumerate(blocks)
                ],
                dim=1,
            )
            continue
        identity = (id(value), value._version, value.data_ptr())
        cached = model._qat_cache.get(name)
        if cached is None or cached[0] != identity:
            cached = (
                identity,
                _represented_matrix(name, value, model.inp.out_features, model._qat_bits[name]),
            )
            model._qat_cache[name] = cached
        replacements[name] = _RepresentedWeight.apply(value, cached[1])
    try:
        with torch.autocast(inputs.device.type, enabled=False):
            return functional_call(model, replacements, (inputs, state), strict=False)
    finally:
        # nn.GRU caches non-Parameter functional overrides in _flat_weights.
        # Restore its ordinary cache after reparameterization so later EMA
        # copies and parameter mutation cannot retain a completed autograd graph.
        model.gru._init_flat_weights()


def effective_qat_state(
    state: Mapping[str, torch.Tensor],
    preset: str,
    source: str,
    learned_scales: Mapping[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Export exact effective matrices and their original signed code/scale representation.

    EMA/SWA callers pass latent averaged tensors. The output is an inference
    artifact; exact continuation remains in the raw epoch training state.
    Never quantize reconstructed values again when importing into a native engine.
    """
    if learned_scales is not None and preset not in QAT_LEARN_SCALE_MODES:
        raise ValueError("learned row scales require a versioned Q4 QAT preset")
    arrays = {name: value.detach().cpu().numpy() for name, value in state.items()}
    represented, encoded, report = quantize_model(arrays, preset)
    if learned_scales is not None:
        hidden = int(arrays["inp.bias"].size)
        expected_scale_keys = {name for name in encoded if name.endswith(".scales")}
        if set(learned_scales) != expected_scale_keys:
            raise ValueError("learned QAT scales are incomplete or unexpected")
        for name, details in report["selected"].items():
            original = arrays[name]
            blocks = np.split(original, [hidden], axis=1) if name == "head.0.weight" else [original]
            rebuilt = []
            squared_error = 0.0
            for index, (block, bits) in enumerate(zip(blocks, details["block_bits"], strict=True)):
                key = f"{name}.{index}.scales"
                raw_scale = learned_scales[key]
                if not isinstance(raw_scale, torch.Tensor) or raw_scale.dtype != torch.float32:
                    raise ValueError(f"learned QAT row step is not a float32 tensor: {key}")
                codes, scales, block_represented = _quantize_block_with_scales(
                    block, raw_scale.detach().cpu().numpy(), bits
                )
                encoded[f"{name}.{index}.codes"] = codes
                encoded[key] = scales
                rebuilt.append(block_represented)
                error = block_represented.astype(np.float64) - block.astype(np.float64)
                squared_error += float(np.square(error).sum())
            represented[name] = np.ascontiguousarray(np.concatenate(rebuilt, axis=1))
            error_count = original.size
            details["max_abs_error"] = float(np.abs(represented[name].astype(np.float64) - original).max())
            details["rms_error"] = float(np.sqrt(squared_error / error_count))
    metadata = {
        **report,
        "representation": "effective_reconstructed_float32",
        "snapshot_source": source,
        "training_rule": (
            _LEARNED_SCALE_TRAINING_RULE if learned_scales is not None else _FIXED_SCALE_TRAINING_RULE
        ),
        "canonical_arithmetic": "numpy float32 on CPU; represented weight cache invalidated after updates",
        "encoded": {name: torch.from_numpy(value.copy()) for name, value in encoded.items()},
        "latent_selected_sha256": {
            name: hashlib.sha256(arrays[name].tobytes()).hexdigest() for name in report["selected"]
        },
    }
    if learned_scales is not None:
        metadata["scale_learning"] = scale_learning_metadata()
    return {name: torch.from_numpy(value.copy()) for name, value in represented.items()}, metadata


def inference_state(
    model: torch.nn.Module,
    source: str,
) -> tuple[dict[str, torch.Tensor], dict[str, object] | None]:
    """Preserve the legacy state-dict path exactly when QAT is off."""
    preset = getattr(model, "_qat_preset", "none")
    if preset == "none":
        return model.state_dict(), None
    state = model.state_dict()
    if not getattr(model, "_qat_learn_scales", False):
        return effective_qat_state(state, preset, source)
    inference_weights = {name: value for name, value in state.items() if not name.startswith("_qat_scale_params.")}
    learned_scales = {
        f"{name}.{index}.scales": model._qat_scale_params[position].detach()
        for position, (name, index, _) in enumerate(model._qat_scale_keys)
    }
    return effective_qat_state(inference_weights, preset, source, learned_scales)
