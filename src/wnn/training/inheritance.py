from __future__ import annotations

import argparse

import torch

from wnn.research_features import (
    FIELD_AGE_FEATURE_MODES,
    MOMENT_FEATURE_MODES,
    research_feature_names,
)


def inherit_init_arguments(args: argparse.Namespace, checkpoint: dict, options: list[str]) -> None:
    """Preserve the parent's feature construction and model/output dimensions during FT."""
    fields = (
        "hidden",
        "encoder_width",
        "recurrent_width",
        "readout_width",
        "head_layers",
        "layers",
        "dropout",
        "arch",
        "decay_init",
        "memory_arm",
        "cell",
        "enc",
        "tok_dim",
        "att_layers",
        "conv_k",
        "readout_activation",
        "tcn_mix",
        "event_hidden",
        "expert_hidden",
        "num_experts",
        "specialist_route",
        "hash_seed",
        "adapter_rank",
        "adapter_mode",
        "prefix_rows",
        "innovation_state_width",
        "innovation_gain_conditioning",
        "online_norm",
        "raw_only",
        "heuristics",
        "research_features",
        "research_feature_control",
        "privileged_mode",
        "privileged_horizon",
        "sentinel_values",
        "sentinel_indicators",
        "sentinel_base_width",
        "prop_input",
        "aux",
        "aux_struct",
        "aux_next",
        "aux_next_k",
        "aux_tier",
        "aux_mask",
        "aux_sign",
        "aux_quote",
        "quote_cache_root",
    )
    source = checkpoint["args"]
    for field in fields:
        if field in source:
            option = "--" + field.replace("_", "-")
            explicit = any(value == option or value.startswith(option + "=") for value in options)
            if explicit and getattr(args, field) != source[field]:
                if field == "research_features" and can_extend_research_features(
                    source[field], getattr(args, field)
                ):
                    continue
                if field == "research_feature_control" and can_extend_research_features(
                    source.get("research_features", "none"), args.research_features
                ):
                    continue
                raise ValueError(f"{option} disagrees with the parent checkpoint")
            setattr(args, field, source[field])


def can_extend_research_features(parent_mode: str, child_mode: str) -> bool:
    """Allow an append-only causal representation to extend a matching parent."""
    appendable_modes = (*MOMENT_FEATURE_MODES, *FIELD_AGE_FEATURE_MODES)
    if child_mode not in appendable_modes or parent_mode not in ("none", *appendable_modes):
        return False
    parent_names = research_feature_names(parent_mode)
    child_names = research_feature_names(child_mode)
    return len(parent_names) < len(child_names) and child_names[: len(parent_names)] == parent_names


def expand_research_feature_state(
    source_state: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    parent_mode: str,
    child_mode: str,
    insert_at: int,
) -> dict[str, torch.Tensor]:
    """Zero-extend causal input columns while preserving the parent function."""
    if not can_extend_research_features(parent_mode, child_mode):
        raise ValueError("initializer research features are not a causal prefix of the child")
    added = len(research_feature_names(child_mode)) - len(research_feature_names(parent_mode))
    expanded = {}
    for name, target in target_state.items():
        source = source_state.get(name)
        if source is None:
            raise ValueError(f"initializer lacks model state {name!r}")
        if source.shape == target.shape:
            expanded[name] = source
        elif name in ("mean", "std") and source.ndim == target.ndim == 1:
            if source.shape[0] + added != target.shape[0]:
                raise ValueError("initializer normalization width does not match causal features")
            value = target.clone()
            value[:insert_at] = source[:insert_at].to(value)
            value[insert_at + added :] = source[insert_at:].to(value)
            value[insert_at : insert_at + added] = 0.0 if name == "mean" else 1.0
            expanded[name] = value
        elif name == "inp.weight" and source.ndim == target.ndim == 2:
            if source.shape[0] != target.shape[0] or source.shape[1] + added != target.shape[1]:
                raise ValueError(
                    "initializer input projection width does not match causal features"
                )
            value = target.clone()
            value[:, :insert_at] = source[:, :insert_at].to(value)
            value[:, insert_at + added :] = source[:, insert_at:].to(value)
            value[:, insert_at : insert_at + added] = 0.0
            expanded[name] = value
        else:
            raise ValueError(
                f"initializer state shape changed outside causal input columns: {name}"
            )
    return expanded


def validate_campaign_checkpoint(
    checkpoint: dict, args: argparse.Namespace, *, final_initializer: bool = False
) -> None:
    """Reject initialization or propensity weights fitted outside this clean outer fold."""
    if final_initializer:
        source = checkpoint.get("args", {})
        if (
            source.get("train_source") != "train+valid"
            or not source.get("include_heldout_valid")
            or source.get("split_manifest") is not None
            or (not source.get("no_eval_during_fit"))
            or (source.get("checkpoint_policy") != "fixed")
            or (checkpoint.get("epoch") != source.get("checkpoint_epoch"))
        ):
            raise ValueError("final adaptation requires a fixed all-data initializer")
        return
    if args.split_manifest:
        source = checkpoint.get("args", {})
        if (
            source.get("split_manifest_sha256") != args.split_manifest_sha256
            or not source.get("no_eval_during_fit")
            or source.get("checkpoint_policy") != "fixed"
            or (checkpoint.get("epoch") != source.get("checkpoint_epoch"))
        ):
            raise ValueError("source checkpoint lacks matching clean-fold fitting provenance")
