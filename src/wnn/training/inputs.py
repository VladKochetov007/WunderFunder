from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

import numpy as np
import torch

from wnn.data import N_FEATURES, SequenceFile
from wnn.features import event_features_torch, online_zscore_torch
from wnn.privileged import privileged_inputs
from wnn.research_features import (
    MOMENT_FEATURE_MODES,
    fit_causal_moment_prior,
    research_feature_names,
    research_features_torch,
    validate_causal_moment_prior,
)
from wnn.training.environment import TrainingEnvironment


def build_inputs(
    x: torch.Tensor, *, environment: TrainingEnvironment | None = None
) -> torch.Tensor:
    """Assemble checkpoint-compatible feature blocks for one sequence batch.

    Propensity uses only base, optional online-normalized and extra input
    columns. Research and privileged features join the model inputs separately;
    sentinel preprocessing runs after the final join. Differentiable raw inputs
    retain the staged graph to preserve floating-point gradient accumulation.

    Args:
        x: Batch-by-time input tensor with raw columns followed by optional extras.
        environment: Explicit feature configuration and propensity dependency.

    Returns:
        Model inputs in the feature order expected by existing checkpoints.
    """
    environment = environment if environment is not None else TrainingEnvironment()
    raw, extra = (x[..., :N_FEATURES], x[..., N_FEATURES:])
    if environment.inputs.research_features == "tl3_lead" and extra.shape[-1] != 0:
        raise ValueError("tl3_lead requires the canonical 112 raw input columns without extras")
    ev = event_features_torch(raw)
    if environment.inputs.raw_only:
        ev = torch.zeros_like(ev)
    if (
        environment.inputs.research_features in MOMENT_FEATURE_MODES
        and (not environment.inputs.online_norm)
        and (not environment.inputs.heuristics)
        and (environment.inputs.privileged_mode == "none")
        and (environment.inputs.propensity_model is None)
        and (environment.inputs.sentinel_preprocessor is None)
        and (extra.shape[-1] == 0)
    ):
        base_width = N_FEATURES + ev.shape[-1]
        feature_width = len(research_feature_names(environment.inputs.research_features))
        inputs = raw.new_empty((*raw.shape[:2], base_width + feature_width))
        inputs[..., :N_FEATURES].copy_(raw)
        inputs[..., N_FEATURES:base_width].copy_(ev)
        del ev
        research_features_torch(
            raw,
            environment.inputs.research_features,
            environment.inputs.research_feature_prior,
            output=inputs[..., base_width:],
            control=environment.inputs.research_feature_control,
        )
        return inputs
    if x.requires_grad:
        return _build_staged_inputs(raw, extra, ev, environment)
    input_parts = [raw, ev]
    if environment.inputs.online_norm:
        input_parts.append(online_zscore_torch(raw))
    propensity_parts = (
        list(input_parts) if environment.inputs.propensity_model is not None else None
    )
    if environment.inputs.research_features != "none":
        input_parts.append(
            research_features_torch(
                raw,
                environment.inputs.research_features,
                environment.inputs.research_feature_prior,
                control=environment.inputs.research_feature_control,
                event=None if environment.inputs.raw_only else ev,
            )
        )
    if environment.inputs.privileged_mode != "none":
        input_parts.append(
            privileged_inputs(
                raw, environment.inputs.privileged_mode, environment.inputs.privileged_horizon
            )
        )
    if extra.shape[-1]:
        input_parts.append(extra)
        if propensity_parts is not None:
            propensity_parts.append(extra)
    if environment.inputs.propensity_model is not None:
        assert propensity_parts is not None
        prop_inputs = torch.cat(propensity_parts, dim=-1)
        input_dtype = input_parts[0].dtype
        for part in input_parts[1:]:
            input_dtype = torch.promote_types(input_dtype, part.dtype)
        input_parts.append(
            environment.propensity(environment.inputs.propensity_model, prop_inputs)[..., None].to(
                input_dtype
            )
        )
    inputs = torch.cat(input_parts, dim=-1)
    if environment.inputs.sentinel_preprocessor is not None:
        inputs = environment.inputs.sentinel_preprocessor.torch(inputs)
    return inputs


def _build_staged_inputs(
    raw: torch.Tensor,
    extra: torch.Tensor,
    event: torch.Tensor,
    environment: TrainingEnvironment,
) -> torch.Tensor:
    """Retain the legacy autograd accumulation order for differentiable inputs."""
    inputs = torch.cat([raw, event], dim=-1)
    if environment.inputs.online_norm:
        inputs = torch.cat([inputs, online_zscore_torch(raw)], dim=-1)
    propensity_inputs = inputs
    if environment.inputs.research_features != "none":
        inputs = torch.cat(
            [
                inputs,
                research_features_torch(
                    raw,
                    environment.inputs.research_features,
                    environment.inputs.research_feature_prior,
                    control=environment.inputs.research_feature_control,
                    event=None if environment.inputs.raw_only else event,
                ),
            ],
            dim=-1,
        )
    if environment.inputs.privileged_mode != "none":
        inputs = torch.cat(
            [
                inputs,
                privileged_inputs(
                    raw, environment.inputs.privileged_mode, environment.inputs.privileged_horizon
                ),
            ],
            dim=-1,
        )
    if extra.shape[-1]:
        inputs = torch.cat([inputs, extra], dim=-1)
        if environment.inputs.propensity_model is not None:
            propensity_inputs = torch.cat([propensity_inputs, extra], dim=-1)
    if environment.inputs.propensity_model is not None:
        inputs = torch.cat(
            [
                inputs,
                environment.propensity(environment.inputs.propensity_model, propensity_inputs)[
                    ..., None
                ].to(inputs.dtype),
            ],
            dim=-1,
        )
    if environment.inputs.sentinel_preprocessor is not None:
        inputs = environment.inputs.sentinel_preprocessor.torch(inputs)
    return inputs


def fit_research_feature_prior(
    train_split: str,
    train_groups: np.ndarray,
    valid_groups: np.ndarray | None,
    seed: int,
    *,
    environment: TrainingEnvironment | None = None,
) -> dict[str, object]:
    """Fit the causal prior from a deterministic sample of fitting groups only."""
    environment = environment if environment is not None else TrainingEnvironment()
    members = [(train_split, int(group)) for group in train_groups]
    if valid_groups is not None:
        members.extend(("valid", int(group)) for group in valid_groups)
    members = sorted(set(members))
    if not members:
        raise ValueError("causal feature prior needs at least one fitting group")
    rng = np.random.default_rng(seed + 20261012)
    if len(members) > 32:
        selected_indices = np.sort(rng.choice(len(members), size=32, replace=False))
        selected = [members[int(index)] for index in selected_indices]
    else:
        selected = members
    files: dict[str, SequenceFile] = {}

    def fitting_sequences() -> Iterable[np.ndarray]:
        """Fitting sequences."""
        for split, group in selected:
            if split not in files:
                files[split] = environment.sequence_file(
                    environment.root / "datasets" / f"{split}.parquet"
                )
            yield files[split].read(group, targets=False, mask=False).features

    prior = fit_causal_moment_prior(fitting_sequences())
    encoded_members = [{"split": split, "row_group": group} for split, group in selected]
    member_text = json.dumps(encoded_members, sort_keys=True, separators=(",", ":"))
    prior["fit_members"] = encoded_members
    prior["fit_members_sha256"] = hashlib.sha256(member_text.encode("ascii")).hexdigest()
    prior["sample_policy"] = "uniform_seeded_source_qualified_32_groups_v1"
    prior["sample_seed"] = seed + 20261012
    return validate_causal_moment_prior(prior)
