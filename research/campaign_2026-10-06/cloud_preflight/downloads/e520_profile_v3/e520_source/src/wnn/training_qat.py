from __future__ import annotations

import hashlib
from collections.abc import Mapping

import numpy as np
import torch
from torch.func import functional_call

from wnn.nn import RNNModel
from wnn.quantization import PRESETS, quantize_model, quantize_rows, represented_rows

QAT_MODES = ("none", *PRESETS)


class _RepresentedWeight(torch.autograd.Function):
    """Exact represented values forward; identity STE to the latent float matrix."""

    @staticmethod
    def forward(ctx: object, latent: torch.Tensor, represented: torch.Tensor) -> torch.Tensor:
        return represented

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient, None


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
    return recurrent + (("head.0.weight",) if preset == "recurrent_head" else ())


def configure_qat(model: RNNModel, preset: str) -> None:
    """Enable weight-only QAT without replacing parameters or changing state-dict keys."""
    if preset not in QAT_MODES:
        raise ValueError("unknown QAT mode")
    if preset == "none":
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
    model._qat_cache = {}


def invalidate_qat_cache(model: torch.nn.Module) -> None:
    """Discard deterministic represented matrices after a latent optimizer/EMA update."""
    if hasattr(model, "_qat_cache"):
        model._qat_cache.clear()


def _represented_matrix(name: str, value: torch.Tensor, hidden: int) -> torch.Tensor:
    # Canonical CPU arithmetic deliberately handles subnormals and ties exactly;
    # CUDA fast math may flush subnormal scales. Once-per-update caching bounds
    # the transfer cost while a fresh STE node permits every truncated backward.
    weights = value.detach().cpu().numpy()
    blocks = np.split(weights, [hidden], axis=1) if name == "head.0.weight" else [weights]
    represented = np.concatenate(
        [represented_rows(*quantize_rows(block)) for block in blocks], axis=1
    )
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
        identity = (id(value), value._version, value.data_ptr())
        cached = model._qat_cache.get(name)
        if cached is None or cached[0] != identity:
            cached = (identity, _represented_matrix(name, value, model.inp.out_features))
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
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Export exact effective float32 matrices and their original signed-byte representation.

    EMA/SWA callers pass latent averaged tensors. The output is an inference
    artifact; exact continuation remains in the raw epoch training state.
    Never quantize reconstructed values again when importing into a native engine.
    """
    arrays = {name: value.detach().cpu().numpy() for name, value in state.items()}
    represented, encoded, report = quantize_model(arrays, preset)
    metadata = {
        **report,
        "representation": "effective_reconstructed_float32",
        "snapshot_source": source,
        "training_rule": "identity STE; detached row scales; no scale, rounding or clipping derivative",
        "canonical_arithmetic": "numpy float32 on CPU; represented weight cache invalidated after updates",
        "encoded": {name: torch.from_numpy(value.copy()) for name, value in encoded.items()},
        "latent_selected_sha256": {
            name: hashlib.sha256(arrays[name].tobytes()).hexdigest() for name in report["selected"]
        },
    }
    return {name: torch.from_numpy(value.copy()) for name, value in represented.items()}, metadata


def inference_state(
    model: torch.nn.Module,
    source: str,
) -> tuple[dict[str, torch.Tensor], dict[str, object] | None]:
    """Preserve the legacy state-dict path exactly when QAT is off."""
    preset = getattr(model, "_qat_preset", "none")
    if preset == "none":
        return model.state_dict(), None
    return effective_qat_state(model.state_dict(), preset, source)
