"""Complete-sequence loss for a learned initializer with a frozen GRU backbone."""

from __future__ import annotations

from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from wnn.state_initialization import PREFIX_ROWS, InitializationState, InitializedGRU


def _loss_chunk(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    hidden: torch.Tensor | None,
    *,
    model: InitializedGRU,
    position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute one pure chunk with its original position during backward."""
    state = None if hidden is None else InitializationState(hidden=hidden, position=position)
    predictions, state = model(inputs, state)
    error = predictions[..., : targets.shape[-1]] - targets
    return (weights * error.square()).sum(), state.hidden


def initializer_sequence_loss(
    model: InitializedGRU,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    *,
    chunk_rows: int = 2000,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    """Backpropagate all scored chunks to the once-per-sequence initializer.

    The caller supplies frozen distillation targets and original-label weights,
    including its scoring/propensity treatment. Warmup rows 0..98 have zero loss
    here but are processed normally. Every backbone parameter must be frozen.
    This is one complete-sequence objective/optimizer update, not the ordinary
    ten-update TBPTT recipe. Its denominator is total sequence-batch weight,
    whereas the ordinary trainer normalizes each chunk separately. Chunk
    boundaries retain the hidden-state graph. The caller must check that the
    supplied targets/weights are finite and weights are nonnegative.

    Non-reentrant checkpointing trades recomputation for saved activations. The
    first chunk must include row 99 so later boundaries need only hidden state;
    prefix statistics are consumed inside that first chunk. No optimizer step
    may occur until the returned scalar has completed backward.
    """
    if model.mode == "zero":
        raise ValueError("zero state has no initializer parameters to fit")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise ValueError("initializer-only loss requires a frozen backbone")
    if chunk_rows <= PREFIX_ROWS:
        raise ValueError("first chunk must include the prefix adjustment at row 99")
    if inputs.ndim != 3 or inputs.shape[1] <= PREFIX_ROWS:
        raise ValueError("inputs must contain synchronized sequences including scored rows")
    if targets.ndim != 3 or targets.shape[:2] != inputs.shape[:2]:
        raise ValueError("targets must share the input batch and time dimensions")
    if targets.shape[-1] != 2 or weights.shape != targets.shape:
        raise ValueError("targets and weights must have two prediction components")
    if targets.requires_grad or weights.requires_grad:
        raise ValueError("targets and weights must be fixed supervision")
    # Inputs and supervision never enter the initializer except through the
    # wrapper's causal inputs. Only the loss receives targets and weights.
    scored = torch.arange(inputs.shape[1], device=weights.device) >= PREFIX_ROWS
    weights = weights * scored[None, :, None]
    hidden = None
    total = inputs.new_zeros(())
    for start in range(0, inputs.shape[1], chunk_rows):
        stop = min(start + chunk_rows, inputs.shape[1])
        function = partial(_loss_chunk, model=model, position=start)
        arguments = (inputs[:, start:stop], targets[:, start:stop], weights[:, start:stop], hidden)
        if checkpoint_chunks:
            loss, hidden = checkpoint(
                function, *arguments, use_reentrant=False, preserve_rng_state=False
            )
        else:
            loss, hidden = function(*arguments)
        total = total + loss
    return total / weights.sum().clamp_min(1e-8)
