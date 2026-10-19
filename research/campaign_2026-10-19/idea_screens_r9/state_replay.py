"""Bounded no-gradient recurrent replay primitives."""

from __future__ import annotations

import torch
from torch import nn

from wnn.training.forward import detach_state, forward_chunk

MAX_FORWARD_CHUNK = 512


def zero_state(model: nn.Module) -> torch.Tensor:
    """Create the canonical zero GRU state for a single CPU sequence."""
    recurrent = getattr(model, "gru", None)
    if recurrent is None or not hasattr(recurrent, "hidden_size"):
        raise ValueError("diagnostic model must expose its canonical GRU")
    return torch.zeros(
        (int(recurrent.num_layers), 1, int(recurrent.hidden_size)),
        dtype=torch.float32,
        device="cpu",
    )


def check_state(model: nn.Module, state: torch.Tensor) -> None:
    """Require finite detached CPU FP32 state with the checkpoint's exact shape."""
    expected = zero_state(model).shape
    if (
        state.shape != expected
        or state.dtype != torch.float32
        or state.device.type != "cpu"
        or state.requires_grad
        or not torch.isfinite(state).all()
    ):
        raise ValueError("recurrent state must be finite detached CPU FP32 with canonical shape")


def clone_state(state: torch.Tensor) -> torch.Tensor:
    """Clone a detached state so branch rollouts cannot mutate one another."""
    if state.requires_grad:
        raise ValueError("diagnostic recurrent states must be detached")
    return state.detach().clone()


def advance(
    model: nn.Module,
    inputs: torch.Tensor,
    state: torch.Tensor | None,
    start: int,
    stop: int,
    category: str,
    counters: dict[str, int],
) -> torch.Tensor:
    """Replay an interval in bounded chunks without constructing a gradient graph."""
    if start < 0 or stop < start or stop > inputs.shape[1]:
        raise ValueError("invalid causal input interval")
    cursor = start
    with torch.no_grad():
        while cursor < stop:
            end = min(cursor + MAX_FORWARD_CHUNK, stop)
            _, next_state = forward_chunk(model, inputs, slice(cursor, end), state)
            state = detach_state(next_state)
            check_state(model, state)
            counters[f"{category}_rows"] += end - cursor
            counters["model_forward_calls"] += 1
            cursor = end
    if state is None:
        raise ValueError("empty replay interval has no explicit initial state")
    return state
