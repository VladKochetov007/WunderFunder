from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
from torch.func import functional_call

from wnn.nn import RNNModel


class GRUMatrixFactors(nn.Module):
    """Balanced factors of one stacked reset/update/new gate matrix."""

    def __init__(self, left: torch.Tensor, right: torch.Tensor) -> None:
        super().__init__()
        self.left = nn.Parameter(left)
        self.right = nn.Parameter(right)

    def effective(self) -> torch.Tensor:
        """Materialize the learned matrix without changing the factor coordinates."""
        return self.left @ self.right


def balanced_svd(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Initialize FP32 balanced factors using canonical-sign CPU float64 SVD.

    Both factors receive sqrt(singular value), so AdamW starts from balanced
    coordinates. Sign canonicalization removes the arbitrary paired sign;
    repeated singular subspaces still depend on the bound numerical stack.
    """
    if weight.ndim != 2 or not 1 <= rank <= min(weight.shape):
        raise ValueError("factor rank must be positive and at most the matrix minimum dimension")
    source = weight.detach().cpu().double()
    if not torch.isfinite(source).all():
        raise ValueError("SVD initialization requires finite weights")
    u, singular, vh = torch.linalg.svd(source, full_matrices=False)
    u, vh, singular = u[:, :rank], vh[:rank], singular[:rank]
    pivot = vh.abs().argmax(dim=1)
    signs = vh[torch.arange(rank), pivot].sign()
    roots = singular.sqrt()
    left = (u * (roots * signs)).float()
    right = (vh * (roots * signs)[:, None]).float()
    residual = source - left.double() @ right.double()
    norm = float(torch.linalg.vector_norm(source))
    report = {
        "source_sha256": hashlib.sha256(weight.detach().cpu().numpy().tobytes()).hexdigest(),
        "relative_frobenius_error": float(torch.linalg.vector_norm(residual)) / norm
        if norm
        else 0.0,
        "rank": rank,
        "shape": list(weight.shape),
    }
    return left.to(weight.device), right.to(weight.device), report


def configure_factorized(model: RNNModel, rank: int) -> None:
    """Replace trainable GRU matrices by factors, retaining dense reference buffers.

    Factors live below ``gru`` so existing trunk freeze prefixes remain valid.
    Dense reference buffers are only initialization evidence and shape/storage
    placeholders for the fused GRU; every active forward overrides all of them.
    """
    if rank == 0:
        return
    if getattr(model, "_qat_preset", "none") != "none":
        raise ValueError("combined QAT and factorized recurrence is not implemented")
    if type(model) is not RNNModel or type(model.gru) is not nn.GRU or model.enc != "linear":
        raise ValueError("factorized recurrence supports plain GRU with a linear input encoder")
    if hasattr(model.gru, "factors"):
        raise ValueError("model is already factorized")
    if any(value.dtype != torch.float32 for value in model.state_dict().values()):
        raise ValueError("factorized recurrence requires FP32 parameters and buffers")
    names = [
        name
        for name, _ in model.gru.named_parameters()
        if name.startswith(("weight_ih_", "weight_hh_"))
    ]
    factors, reports = {}, {}
    for name in names:
        weight = getattr(model.gru, name)
        left, right, report = balanced_svd(weight, rank)
        factors[name] = GRUMatrixFactors(left, right)
        reports[f"gru.{name}"] = report
    for name in names:
        weight = model.gru._parameters.pop(name)
        model.gru.register_buffer(name, weight.detach())
    model.gru.factors = nn.ModuleDict(factors)
    model.gru._init_flat_weights()
    model._recurrent_rank = rank
    model._factorization_initialization = reports
    model._inference_fp32 = True


def factorized_forward(
    model: RNNModel, inputs: torch.Tensor, state: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize factors for fused GRU training; recurrent state remains FP32."""
    if inputs.dtype != torch.float32 or state is not None and state.dtype != torch.float32:
        raise ValueError("factorized inputs and recurrent state must remain FP32")
    with torch.autocast(inputs.device.type, enabled=False):
        replacements = {
            f"gru.{name}": factors.effective() for name, factors in model.gru.factors.items()
        }
        try:
            return functional_call(model, replacements, (inputs, state), strict=False)
        finally:
            model.gru._init_flat_weights()


def parameter_budget(n_in: int, n_out: int, hidden: int, layers: int, rank: int) -> dict:
    """Count learned parameters and report the closest integer-width dense control."""
    if min(n_in, n_out, hidden, layers, rank) < 1 or rank > hidden:
        raise ValueError("positive dimensions and rank <= hidden are required")
    linear = n_in + n_out + 6 * layers + 2
    target = 2 * hidden**2 + (8 * layers * rank + linear) * hidden + n_out
    quadratic = 6 * layers + 2
    root = (-linear + math.sqrt(linear**2 + 4 * quadratic * (target - n_out))) / (2 * quadratic)
    candidates = [max(1, math.floor(root)), max(1, math.ceil(root))]

    def count(width: int) -> int:
        return quadratic * width**2 + linear * width + n_out

    width = min(candidates, key=lambda candidate: (abs(count(candidate) - target), candidate))
    return {
        "factorized_learned_parameters": target,
        "same_width_dense_parameters": count(hidden),
        "matched_dense_hidden": width,
        "matched_dense_parameters": count(width),
        "matched_dense_relative_difference": (count(width) - target) / target,
        "frozen_reference_buffer_elements": 6 * layers * hidden**2,
    }


def factorized_inference_state(model: RNNModel, source: str) -> tuple[dict, dict]:
    """Export effective dense weights plus factors; make no native runtime claim.

    EMA/SWA average latent factors, then multiply. This differs from averaging
    the effective dense matrices and is part of the declared parameterization.
    Materialization uses the model device so its exact FP32 values match forward.
    """
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("gru.factors.")
    }
    factors = {}
    with torch.no_grad(), torch.autocast(next(model.parameters()).device.type, enabled=False):
        for name, pair in model.gru.factors.items():
            full_name = f"gru.{name}"
            state[full_name] = pair.effective().detach().cpu()
            factors[full_name] = {
                "left": pair.left.detach().cpu().clone(),
                "right": pair.right.detach().cpu().clone(),
            }
    metadata = {
        "scheme": "stacked_gru_low_rank_uv_v1",
        "rank": model._recurrent_rank,
        "matrix_convention": "weight = left @ right; stacked PyTorch reset,update,new rows",
        "factors": factors,
        "initialization": model._factorization_initialization,
        "snapshot_source": source,
        "averaging": "latent factors; effective product formed after averaging",
        "training_compute": "materialized dense FP32 GRU",
        "native_runtime_validated": False,
        "parameter_budget": parameter_budget(
            model.inp.in_features,
            model.head[2].out_features,
            model.inp.out_features,
            model.gru.num_layers,
            model._recurrent_rank,
        ),
    }
    return state, metadata
