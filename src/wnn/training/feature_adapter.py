"""Train appended causal input columns while preserving a fixed parent backbone."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from wnn.training.state import TrainingRun


def configure_feature_adapter(run: TrainingRun) -> None:
    """Mask old input gradients and freeze every other parent parameter."""
    args = run.options.args
    if not getattr(args, "research_feature_adapter_only", False):
        return
    extension = run.data.provenance.get("research_feature_extension")
    if not args.init or not isinstance(extension, dict):
        raise ValueError("feature-adapter-only requires a zero-extended parent initializer")
    if (
        args.wd != 0
        or bool(args.freeze)
        or args.qat != "none"
        or args.numeric_map != "none"
        or args.precision != "fp32"
        or args.optim != "adamw"
        or args.resume_state is not None
    ):
        raise ValueError(
            "feature-adapter-only requires zero decay, no generic freeze and an unmapped FP32 model"
        )
    model = run.network.model
    added = int(extension["added_columns"])
    weight = model.inp.weight
    if (
        added <= 0
        or added >= weight.shape[1]
        or args.online_norm
        or args.prop_input
        or args.heuristics
    ):
        raise ValueError("feature-adapter-only supports a raw/event parent with a new suffix")
    boundary = weight.shape[1] - added
    if (
        extension.get("parent_mode") != "none"
        or extension.get("child_mode") != "causal_geometry_discrete"
        or added != 60
        or boundary != 211
    ):
        raise ValueError(
            "feature-adapter-only requires the audited raw/event 211 + geometry 60 layout"
        )
    if bool(torch.count_nonzero(weight[:, boundary:]).item()):
        raise ValueError("new feature columns must start at exactly zero")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    weight.requires_grad_(True)
    mask = weight.new_zeros((1, weight.shape[1]))
    mask[:, boundary:] = 1
    weight.register_hook(lambda gradient: gradient * mask)
    run.data.provenance["feature_adapter_only"] = {
        "parameter": "inp.weight",
        "trainable_columns": [boundary, weight.shape[1]],
        "old_columns": "zero gradients; fresh optimizer; weight decay zero",
        "all_other_parameters": "requires_grad=False",
        "purpose": "isolate learned causal features from parent-backbone continuation drift",
    }
