"""B300 receiver probes for rolling state, causality and sequence reset."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from bilinear_entmax.receiver_support import ATOL, assert_close


def check_streaming_cache(
    model: nn.Module, sample: Tensor, chunk_rows: int, context_rows: int, mode: str
) -> dict[str, float | int]:
    """Compare explicit streaming to the canonical callback and validate rolling caches."""
    from wnn.training.sequence_attention import TransformerState
    from wnn.training.sequence_precision import forward_context
    from wnn.training.sequence_predict import predict_sequence

    model.eval()
    state: Any = None
    outputs: list[Tensor] = []
    observed_cache_rows = 0
    with torch.no_grad():
        for start in range(0, sample.shape[0], chunk_rows):
            stop = min(start + chunk_rows, sample.shape[0])
            with forward_context(mode, sample.device.type):
                prediction, hints, state = model(sample[None, start:stop], state)
            outputs.append(torch.cat((prediction.float(), hints.float()), dim=-1).squeeze(0))
            if not isinstance(state, TransformerState) or state.next_position != stop:
                raise AssertionError("streaming model returned an invalid causal cache state")
            expected_cache = min(stop, context_rows - 1)
            if len(state.keys) != 2 or len(state.values) != 2:
                raise AssertionError("probe transformer cache layer count differs from two")
            for key, value in zip(state.keys, state.values, strict=True):
                if key.shape != value.shape or key.shape[-2] != expected_cache:
                    raise AssertionError("streaming KV cache length differs from rolling context")
                if key.dtype != torch.float32 or value.dtype != torch.float32:
                    raise AssertionError("streaming KV cache must retain canonical FP32 state")
            observed_cache_rows = max(observed_cache_rows, expected_cache)
    streamed = torch.cat(outputs, dim=0).cpu()
    callback = torch.from_numpy(predict_sequence(model, sample, chunk_rows, sample.device))
    gap = assert_close(streamed, callback, "canonical streaming callback", atol=ATOL)
    return {
        "callback_max_abs": gap,
        "max_cache_rows": observed_cache_rows,
        "next_position": sample.shape[0],
    }


def check_sequence_reset(
    model: nn.Module,
    sample: Tensor,
    chunk_rows: int,
    mode: str,
) -> dict[str, float | int]:
    """Require a fresh ``None`` state to reset positions and cached history."""
    from wnn.training.sequence_attention import TransformerState
    from wnn.training.sequence_precision import forward_context

    model.eval()
    chunk = sample[None, :chunk_rows]
    with torch.no_grad(), forward_context(mode, sample.device.type):
        first, _, first_state = model(chunk, None)
        repeated, _, reset_state = model(chunk, None)
        _, _, continued_state = model(chunk, first_state)
    if not all(
        isinstance(value, TransformerState) for value in (first_state, reset_state, continued_state)
    ):
        raise AssertionError("sequence reset probe returned an invalid Transformer state")
    if first_state.next_position != chunk_rows or reset_state.next_position != chunk_rows:
        raise AssertionError("a fresh model call did not reset the absolute sequence position")
    if continued_state.next_position != 2 * chunk_rows:
        raise AssertionError("continued streaming did not retain its absolute sequence position")
    for state in (first_state, reset_state):
        if any(key.shape[-2] != chunk_rows for key in state.keys):
            raise AssertionError("fresh sequence state has an unexpected cached row count")
    gap = assert_close(first.float(), repeated.float(), "fresh-state reset output")
    return {"fresh_reset_max_abs": gap, "reset_position": reset_state.next_position}
