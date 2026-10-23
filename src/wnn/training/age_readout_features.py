"""Canonical feature assembly for the cached field-age residual screen."""

from __future__ import annotations

import inspect

import numpy as np
import torch
from torch import Tensor

from wnn.conditional_readout import parent_inputs_from_raw
from wnn.research_features import (
    FIELD_AGE_CAP,
    StreamingFieldAgeFeatures,
    research_features_torch,
)

AGE_DENOMINATOR = np.float32(np.log1p(FIELD_AGE_CAP))


def normalized_base(raw: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Build the frozen parent-normalized 211 input columns for one sequence."""
    if raw.ndim != 2 or raw.shape[1] != 112:
        raise ValueError("raw sequence must have shape (rows,112)")
    if mean.shape != (211,) or std.shape != (211,) or torch.any(std <= 0):
        raise ValueError("parent normalization must contain 211 positive-scale columns")
    base, _ = parent_inputs_from_raw(raw.unsqueeze(0))
    return ((base.squeeze(0) - mean) / std).contiguous()


def normalized_age(raw: Tensor, mode: str) -> Tensor:
    """Return capped reset-local log age scaled to the fixed [0,1] interval."""
    if mode not in ("group_age", "field_age"):
        raise ValueError("age mode must be group_age or field_age")
    values = research_features_torch(raw.unsqueeze(0), mode).squeeze(0)
    return (values / AGE_DENOMINATOR).contiguous()


def validate_age_invariants() -> dict[str, object]:
    """Check prefix, reset, ABA, group repetition, cap and mask-free age semantics."""
    rng = np.random.default_rng(20261023)
    raw = rng.normal(size=(257, 112)).astype(np.float32)
    raw[1::11, :22] = raw[::11, :22][: len(raw[1::11])]
    report: dict[str, object] = {}
    batches: dict[str, np.ndarray] = {}
    for mode in ("group_age", "field_age"):
        batch = research_features_torch(torch.from_numpy(raw[None]), mode)[0].numpy()
        batches[mode] = batch
        stream = StreamingFieldAgeFeatures(mode)
        streamed = stream.update_chunk(raw)
        if not np.array_equal(batch, streamed):
            raise AssertionError(f"{mode} batch and streaming features differ")
        prefix = research_features_torch(torch.from_numpy(raw[:129][None]), mode)[0].numpy()
        if not np.array_equal(batch[:129], prefix):
            raise AssertionError(f"{mode} features changed when a future suffix was removed")
        stream.reset()
        if not np.array_equal(stream.update_chunk(raw[:32]), batch[:32]):
            raise AssertionError(f"{mode} reset did not restore sequence-start state")
        report[mode] = {
            "batch_stream_exact": True,
            "prefix_exact": True,
            "reset_exact": True,
            "range": [float(batch.min()), float(batch.max())],
        }
    aba = np.zeros((5, 112), dtype=np.float32)
    aba[1] = 1.0
    aba[2] = 0.0
    for mode in ("group_age", "field_age"):
        ages = research_features_torch(torch.from_numpy(aba[None]), mode)[0]
        if not torch.equal(ages[2], torch.zeros(112)):
            raise AssertionError(f"{mode} A-B-A return did not reset age")
        capped = np.zeros((FIELD_AGE_CAP + 2, 112), dtype=np.float32)
        cap_features = research_features_torch(torch.from_numpy(capped[None]), mode)[0]
        normalized_cap = cap_features[-1] / float(AGE_DENOMINATOR)
        if not bool(torch.all(cap_features[-1] == float(AGE_DENOMINATOR))):
            raise AssertionError(f"{mode} cap differs from FIELD_AGE_CAP")
        if not bool(torch.all(normalized_cap == 1.0)):
            raise AssertionError(f"{mode} fixed age scale differs from [0,1] contract")
    for start, end in ((0, 22), (22, 44), (44, 52), (52, 74), (74, 96), (96, 104), (104, 112)):
        if not np.all(
            batches["group_age"][:, start:end] == batches["group_age"][:, start : start + 1]
        ):
            raise AssertionError("group-age columns do not repeat the group clock")
    signature = inspect.signature(research_features_torch)
    if any("mask" in name or "score" in name for name in signature.parameters):
        raise AssertionError("age feature API unexpectedly consumes a mask or score flag")
    report["mask_invariance"] = "feature API receives only raw inputs and fixed mode"
    report["aba_exact"] = True
    report["cap_exact"] = True
    return report
