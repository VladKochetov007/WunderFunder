"""Train appended causal input columns while preserving a fixed parent backbone."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from wnn.training.state import TrainingRun


def configure_feature_adapter(run: TrainingRun) -> None:
    """Freeze the parent backbone, mask old input gradients, and record objective provenance."""
    args = run.options.args
    adapter_only = bool(getattr(args, "research_feature_adapter_only", False))
    objective = getattr(args, "research_feature_adapter_objective", "inherited")
    if objective not in ("inherited", "target-only"):
        raise ValueError("feature-adapter objective must be inherited or target-only")
    if objective == "target-only" and not adapter_only:
        raise ValueError("target-only objective requires --research-feature-adapter-only")
    if objective == "target-only" and (
        args.loss != "wmse"
        or bool(getattr(args, "distill", ""))
        or bool(getattr(args, "distill_cache_root", None))
        or bool(getattr(args, "final_distill_initializer", False))
        or float(getattr(args, "distill_family_weight", 0.0)) != 0.0
        or float(getattr(args, "td_alpha", 0.0)) != 0.0
        or float(getattr(args, "ema_teacher", 0.0)) != 0.0
        or float(getattr(args, "ce_mse", 0.0)) != 0.0
    ):
        raise ValueError("target-only objective requires plain WMSE without added teacher losses")
    if not adapter_only:
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
    run.data.provenance["feature_adapter_objective"] = {
        "mode": objective,
        "main_loss": args.loss,
        "distillation": getattr(args, "distill", ""),
        "main_loss_modifiers": {
            name: getattr(args, name, None) for name in ("ce_mse", "td_alpha", "loss_norm")
        },
        "auxiliary_terms": "inherited" if objective == "inherited" else "skipped",
        "inherited_auxiliary_flags": {
            name: getattr(args, name, None)
            for name in (
                "aux",
                "aux_weight",
                "aux_next",
                "aux_next_k",
                "aux_struct",
                "aux_tier",
                "aux_mask",
                "aux_sign",
                "aux_quote",
                "gate_ce",
                "ema_teacher",
            )
        },
        "output_heads": "retained from parent; all parameters except appended input columns frozen",
    }
