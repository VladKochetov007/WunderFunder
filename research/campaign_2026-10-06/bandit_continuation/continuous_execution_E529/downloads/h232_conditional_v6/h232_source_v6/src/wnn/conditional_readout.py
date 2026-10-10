"""Low-rank readouts over a frozen recurrent state and causal event context."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wnn.data import N_FEATURES
from wnn.features import FEATURE_NAMES, event_features_torch

CONTEXT_FEATURE_NAMES = tuple(FEATURE_NAMES)
CONTEXT_WIDTH = len(CONTEXT_FEATURE_NAMES)
READOUT_MODES = ("hidden_only", "additive", "multiplicative", "conditional_residual")


def evaluation_masks(
    row_groups: Iterable[int], evaluation_groups: Iterable[int], selected_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Bind a prediction-row mask to exact evaluated groups and scored rows.

    The group mask is aligned to exported row-group order. Its broadcast
    conjunction with the supplied scored mask is the only allowed held-out
    grader mask; the unqualified scored mask also includes fit-development rows.
    """
    groups = np.asarray(tuple(map(int, row_groups)), dtype=np.int64)
    evaluated = np.asarray(tuple(map(int, evaluation_groups)), dtype=np.int64)
    selected = np.asarray(selected_mask, dtype=np.bool_)
    if groups.ndim != 1 or evaluated.ndim != 1 or not len(groups):
        raise ValueError("row groups must be a nonempty one-dimensional array")
    if len(np.unique(groups)) != len(groups) or len(np.unique(evaluated)) != len(evaluated):
        raise ValueError("row and evaluation group ids must be unique")
    if selected.ndim != 2 or selected.shape[0] != len(groups):
        raise ValueError("selected mask must have shape (exported groups, rows)")
    if not set(evaluated.tolist()).issubset(groups.tolist()):
        raise ValueError("export does not contain every frozen evaluation group")
    group_mask = np.isin(groups, evaluated)
    if int(group_mask.sum()) != len(evaluated):
        raise ValueError("exported evaluation-group mask cardinality changed")
    return group_mask, selected & group_mask[:, None]


def parent_inputs_from_raw(raw: Tensor) -> tuple[Tensor, Tensor]:
    """Build the E511 raw-plus-event input and return the ordered 99-column context."""
    if raw.ndim != 3 or raw.shape[-1] != N_FEATURES:
        raise ValueError(f"raw input must have shape (batch, rows, {N_FEATURES})")
    if CONTEXT_WIDTH != 99:
        raise RuntimeError(f"expected the frozen 99-column event context, found {CONTEXT_WIDTH}")
    raw = raw.float()
    context = event_features_torch(raw)
    return torch.cat((raw, context), dim=-1), context


def primary_pair_wmse(
    prediction: Tensor, target: Tensor, row_weight: Tensor, weight_label: Tensor
) -> Tensor:
    """E511 chunk-normalized loss on clipped labels and original-label magnitudes."""
    if (
        prediction.shape != target.shape
        or weight_label.shape != target.shape
        or prediction.ndim != 3
        or prediction.shape[-1] != 2
    ):
        raise ValueError(
            "prediction, target and weight label must have equal (batch, rows, 2) shapes"
        )
    if row_weight.shape != target.shape[:2]:
        raise ValueError("row weights must align with batch and row dimensions")
    clipped = target.float().clamp(-2.0, 2.0)
    weights = weight_label.float().clamp(-2.0, 2.0).abs() * row_weight.float()[..., None]
    squared_error = (prediction.float() - clipped).square()
    return (weights * squared_error).sum() / weights.sum().clamp_min(1e-8)


def freeze_parent(model: nn.Module) -> None:
    """Put the parent's full trunk and auxiliary output head in fixed inference mode."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def iter_frozen_parent_chunks(
    model: nn.Module,
    inputs: Tensor,
    context: Tensor,
    chunk_rows: int,
    *,
    amp_enabled: bool,
    amp_dtype: torch.dtype = torch.float16,
) -> Iterator[tuple[slice, Tensor, Tensor, Tensor]]:
    """Yield one frozen trunk chunk at a time; never retain a full hidden-state tensor.

    ``model`` is the plain GRU parent: its ``encode``, ``gru`` and ``head`` are
    invoked in the same order as ``RNNModel.forward``. The full parent head,
    including its auxiliary outputs, is evaluated and returned in ``base``.
    ``hidden`` is the exact ``[recurrent, projection]`` representation consumed
    by the parent's own readout.
    """
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if inputs.ndim != 3 or context.ndim != 3 or inputs.shape[:2] != context.shape[:2]:
        raise ValueError("parent inputs and event context must be aligned (batch, rows, features)")
    if context.shape[-1] != CONTEXT_WIDTH:
        raise ValueError(f"event context must contain exactly {CONTEXT_WIDTH} columns")
    if not hasattr(model, "mean") or not hasattr(model, "std") or not hasattr(model, "gru"):
        raise TypeError("conditional readout requires the frozen plain-GRU parent")

    state: Tensor | tuple[Tensor, ...] | None = None
    for start in range(0, inputs.shape[1], chunk_rows):
        sl = slice(start, min(start + chunk_rows, inputs.shape[1]))
        autocast = (
            torch.autocast(device_type=inputs.device.type, dtype=amp_dtype, enabled=True)
            if amp_enabled
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            projected = model.encode((inputs[:, sl] - model.mean) / model.std)
            recurrent, state = model.gru(projected, state)
            hidden = torch.cat((recurrent, projected), dim=-1)
            base = model.head(hidden)
        if isinstance(state, tuple):
            state = tuple(value.detach() for value in state)
        elif state is not None:
            state = state.detach()
        yield sl, base.float(), hidden.float(), context[:, sl].float()


class _ReadoutControl(nn.Module):
    """Equal-size rank-r control; unused branches remain deliberately present."""

    def __init__(self, mode: str, hidden_width: int, context_width: int, rank: int) -> None:
        super().__init__()
        if mode not in READOUT_MODES:
            raise ValueError(f"unknown conditional readout mode: {mode}")
        if min(hidden_width, context_width, rank) <= 0:
            raise ValueError("readout widths must be positive")
        self.mode = mode
        self.hidden_projection = nn.Linear(hidden_width, rank, bias=False)
        self.context_projection = nn.Linear(context_width, rank, bias=False)
        self.fusion = nn.Linear(3 * rank, rank)
        self.output = nn.Linear(rank, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, anchor: Tensor, hidden: Tensor, context: Tensor) -> Tensor:
        anchor = anchor.float()
        hidden_rank = self.hidden_projection(hidden.float())
        context_rank = self.context_projection(context.float())
        zeros = torch.zeros_like(hidden_rank)
        if self.mode == "hidden_only":
            fusion_input = torch.cat((hidden_rank, zeros, zeros), dim=-1)
        elif self.mode == "additive":
            fusion_input = torch.cat((hidden_rank + context_rank, zeros, zeros), dim=-1)
        elif self.mode == "multiplicative":
            fusion_input = torch.cat((zeros, zeros, hidden_rank * context_rank), dim=-1)
        else:
            fusion_input = torch.cat(
                (hidden_rank, context_rank, hidden_rank * context_rank), dim=-1
            )
        correction = self.output(F.gelu(self.fusion(fusion_input)))
        if self.mode == "multiplicative":
            return anchor * (1.0 + torch.tanh(correction))
        return anchor + correction


class ConditionalReadoutHeads(nn.Module):
    """Four separate rank-r controls around a fixed two-target parent prediction.

    The modules have identical parameter counts. All final pair projections are
    zero-initialized, so every endpoint equals the anchor before its first update.
    """

    def __init__(self, hidden_width: int, rank: int = 32) -> None:
        super().__init__()
        self.rank = rank
        self.controls = nn.ModuleDict(
            {
                mode: _ReadoutControl(mode, hidden_width, CONTEXT_WIDTH, rank)
                for mode in READOUT_MODES
            }
        )

    def forward_all(
        self, parent_output: Tensor, hidden: Tensor, context: Tensor
    ) -> dict[str, Tensor]:
        if parent_output.shape[-1] < 2:
            raise ValueError("parent checkpoint must preserve its two primary outputs")
        anchor = parent_output[..., :2].float()
        outputs = {"anchor": anchor}
        outputs.update(
            {mode: control(anchor, hidden, context) for mode, control in self.controls.items()}
        )
        return outputs

    def parameter_counts(self) -> dict[str, int]:
        """Return per-control parameter counts for the frozen recipe receipt."""
        return {
            name: sum(parameter.numel() for parameter in control.parameters())
            for name, control in self.controls.items()
        }
