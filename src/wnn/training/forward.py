from __future__ import annotations

import hashlib
import time

# Historical checkpoint records and experimental backend/batch states have
# heterogeneous layouts; Any is retained only at those compatibility boundaries.
from typing import Any

import torch

from wnn.data import N_FEATURES
from wnn.features import FEATURE_NAMES
from wnn.nn import DualClockModel
from wnn.training_factorized import factorized_forward
from wnn.training_qat import qat_forward, qat_weight_replacements

EV_I0PX = N_FEATURES + FEATURE_NAMES.index("ev_i0px")


def forward_chunk(
    model: torch.nn.Module, inputs: torch.Tensor, sl: slice, state: Any
) -> tuple[torch.Tensor, Any]:
    """Forward chunk."""
    activation_backend = getattr(model, "_activation_qat_backend", None)
    if activation_backend is not None:
        from wnn.activation_qat import require_finite_activation_a8

        chunk = inputs[:, sl]
        require_finite_activation_a8(chunk, "input chunk")
        if state is not None:
            require_finite_activation_a8(state, "recurrent state")
        parameters = dict(model.named_parameters())
        parameters.update(qat_weight_replacements(model))
        compile_timings = getattr(model, "_activation_profile_compile_timings", None)
        compile_started = (
            time.perf_counter()
            if compile_timings is not None and compile_timings["forward"] is None
            else None
        )
        result = activation_backend(chunk, parameters, False, state, True)
        if compile_started is not None:
            if inputs.device.type == "cuda":
                torch.cuda.synchronize(inputs.device)
            compile_timings["forward"] = time.perf_counter() - compile_started
        fingerprints = getattr(model, "_activation_profile_cache_fingerprints", None)
        if fingerprints is not None and len(fingerprints) < 2 and getattr(model, "_qat_names", ()):
            fingerprints.append(
                {
                    name: hashlib.sha256(
                        model._qat_cache[name][1].detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest()
                    for name in model._qat_names
                }
            )
        return result
    if getattr(model, "_recurrent_rank", 0):
        return factorized_forward(model, inputs[:, sl], state)
    if getattr(model, "_qat_preset", "none") != "none":
        return qat_forward(model, inputs[:, sl], state)
    if isinstance(model, DualClockModel):
        return model(inputs[:, sl], inputs[:, sl, EV_I0PX] > 0.5, state)
    return model(inputs[:, sl], state)


def detach_state(state: Any) -> Any:
    """Detach state."""
    if isinstance(state, tuple):
        return tuple(v.detach() for v in state)
    return state.detach()
