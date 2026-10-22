from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from warm_i1_preflight_identity import verify_recipe_and_files, verify_setup_provenance

if TYPE_CHECKING:
    from wnn.training.state import TrainingRun


def _new_run(argv: list[str], project_root: Path, source_root: Path) -> TrainingRun:
    """Parse the frozen command into the canonical training run container."""
    from wnn.training.cli import build_parser
    from wnn.training.environment import TrainingEnvironment
    from wnn.training.state import TrainingRun

    environment = TrainingEnvironment(root=project_root, source_root=source_root, argv=argv[1:])
    run = TrainingRun(environment)
    run.options.ap = build_parser()
    run.options.args = run.options.ap.parse_args(argv[1:])
    return run


def _run_setup(
    run: TrainingRun,
) -> tuple[str, str, dict[str, object], dict[str, object]]:
    """Execute the canonical setup path and return RNG identities around normalization."""
    import torch

    from wnn.run_identity import sample_order_identity
    from wnn.training.data_caches import bind_caches
    from wnn.training.data_loaders import build_loaders
    from wnn.training.data_membership import select_membership
    from wnn.training.provenance import record_provenance
    from wnn.training.setup_lineage import configure_runtime, load_lineage, validate_membership
    from wnn.training.setup_model import build_model, load_initializer
    from wnn.training.setup_model_runtime import configure_activation_backend, record_model_identity
    from wnn.training.setup_preprocessing import configure_initializer_teacher, fit_preprocessing
    from wnn.training.validation_inputs import validate_distillation, validate_inputs
    from wnn.training.validation_training import validate_clocks, validate_models

    validate_inputs(run)
    validate_distillation(run)
    validate_models(run)
    load_lineage(run)
    validate_membership(run)
    configure_runtime(run)
    validate_clocks(run)
    select_membership(run)
    bind_caches(run)
    build_loaders(run)
    record_provenance(run)
    build_model(run)
    load_initializer(run)
    rng_before = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    sample_before = sample_order_identity(run.data.train_loader)
    fit_preprocessing(run)
    rng_after = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    sample_after = sample_order_identity(run.data.train_loader)
    configure_initializer_teacher(run)
    configure_activation_backend(run)
    record_model_identity(run)
    return rng_before, rng_after, sample_before, sample_after


def _verify_zero_extension(run: TrainingRun, parent_spec: dict[str, object]) -> dict[str, object]:
    """Require exact parent state, 211-prefix buffers and zero input/readout appends."""
    import torch

    from wnn.run_identity import tensor_identity
    from wnn.training.i1_warm_normalization import PARENT_INPUT_WIDTH

    parent = run.network.init
    if parent is None:
        raise ValueError("canonical setup did not retain the actual initializer")
    if run.network.n_in != 367 or run.network.n_out != 25:
        raise ValueError("child tensor layout must be 367 inputs and 25 output rows")
    parent_state = cast(dict[str, torch.Tensor], parent["model"])
    child_state = run.network.model.state_dict()
    if parent_state["mean"].shape != (PARENT_INPUT_WIDTH,):
        raise ValueError("actual parent is not a 211-slot input model")
    if tensor_identity(parent_state["mean"])["bytes_sha256"] != parent_spec["mean_identity"]:
        raise ValueError("actual parent mean bytes differ from the frozen plan")
    if tensor_identity(parent_state["std"])["bytes_sha256"] != parent_spec["std_identity"]:
        raise ValueError("actual parent scale bytes differ from the frozen plan")
    if parent_state["head.2.weight"].shape[0] != 23:
        raise ValueError("actual parent is not the expected 23-row readout")
    if parent_state["inp.weight"].shape[1] != PARENT_INPUT_WIDTH:
        raise ValueError("actual parent input projection is not 211 slots")
    if set(parent_state) != set(child_state):
        raise ValueError("parent and child state keys differ beyond the supported append")

    for name, before in parent_state.items():
        after = child_state[name]
        if name in ("mean", "std"):
            if not torch.equal(before, after[:PARENT_INPUT_WIDTH]):
                raise ValueError(f"inherited normalization prefix changed: {name}")
        elif name == "inp.weight":
            if not torch.equal(before, after[:, :PARENT_INPUT_WIDTH]):
                raise ValueError("inherited input projection changed")
            if torch.count_nonzero(after[:, PARENT_INPUT_WIDTH:]).item():
                raise ValueError("appended input weights are not exactly zero")
        elif name in ("head.2.weight", "head.2.bias"):
            if not torch.equal(before, after[:23]) or torch.count_nonzero(after[23:]).item():
                raise ValueError(f"23-to-25 readout is not an exact zero append: {name}")
        elif not torch.equal(before, after):
            raise ValueError(f"unexpected parent tensor change at initialization: {name}")

    appended_std = run.network.model.std[PARENT_INPUT_WIDTH:]
    if not bool(torch.isfinite(appended_std).all()) or bool((appended_std < 0.001).any()):
        raise ValueError("appended normalization scales must be finite and at least 0.001")
    return {
        "parent_tensor_layout": {
            "input_width": int(parent_state["mean"].shape[0]),
            "output_rows": int(parent_state["head.2.weight"].shape[0]),
        },
        "child_tensor_layout": {
            "input_width": int(run.network.n_in),
            "output_rows": int(run.network.n_out),
        },
        "zero_extension_verified": True,
        "normalization_prefix_verified": True,
    }


def setup_arm(
    argv: list[str],
    run_name: str,
    project_root: Path,
    source_root: Path,
    plan: dict[str, object],
    cell: dict[str, object],
    order_expectation: dict[str, object] | None = None,
) -> tuple[dict[str, object], TrainingRun]:
    """Run create-only canonical setup and remove its required empty run directory."""
    out_dir = project_root / "artifacts" / "runs" / run_name
    if out_dir.exists():
        raise FileExistsError(f"create-only preflight run directory already exists: {out_dir}")
    try:
        return _setup_arm_body(
            argv, run_name, project_root, source_root, plan, cell, order_expectation
        )
    finally:
        if out_dir.is_dir():
            if any(out_dir.iterdir()):
                raise RuntimeError(f"preflight unexpectedly wrote run artifacts: {out_dir}")
            out_dir.rmdir()


def _setup_arm_body(
    argv: list[str],
    run_name: str,
    project_root: Path,
    source_root: Path,
    plan: dict[str, object],
    cell: dict[str, object],
    order_expectation: dict[str, object] | None = None,
) -> tuple[dict[str, object], TrainingRun]:
    """Run source-bound canonical setup and return identities before fitting."""
    run = _new_run(argv, project_root, source_root)
    args = run.options.args
    parent_spec, cache_spec, propensity_spec, split_spec, parent_record = verify_recipe_and_files(
        args, cell, project_root
    )
    rng_before, rng_after, sample_before, sample_after = _run_setup(run)
    cache, prior_sources = verify_setup_provenance(
        run.data.provenance,
        plan,
        parent_spec,
        cache_spec,
        propensity_spec,
        split_spec,
        parent_record,
        run,
        order_expectation,
    )
    tensor_layout = _verify_zero_extension(run, parent_spec)
    provenance = run.data.provenance
    warm_norm = cast(dict[str, object], provenance["i1_warm_normalization"])
    prior_json = json.dumps(
        provenance.get("causal_feature_prior"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    result = {
        "run_name": run_name,
        "arm": str(args.research_feature_control),
        "initial_checkpoint_sha256": provenance["initial_checkpoint_sha256"],
        "split_manifest_sha256": provenance["split_manifest_sha256"],
        "distillation_cache": cache,
        "propensity_checkpoint_sha256": provenance["propensity_checkpoint_sha256"],
        "fit_membership_identity": provenance["fit_membership_identity"],
        "initial_parameter_sha256": provenance["initial_parameter_sha256"],
        "normalization_identity": provenance["normalization_identity"],
        "appended_mean_identity": warm_norm["appended_mean_identity"],
        "appended_std_identity": warm_norm["appended_std_identity"],
        "feature_prior_source_sha256": prior_sources,
        "feature_prior_sha256": hashlib.sha256(prior_json.encode("utf-8")).hexdigest(),
        "torch_rng_before_normalization_sha256": rng_before,
        "torch_rng_after_normalization_sha256": rng_after,
        "sample_order_before_normalization": sample_before,
        "sample_order_after_normalization": sample_after,
        **tensor_layout,
    }
    return result, run
