from __future__ import annotations

from pathlib import Path

import numpy as np

from wnn import student_frontier
from wnn.i1_history import I1_HISTORY_MODE, i1_history_input_mask_indices
from wnn.research_features import (
    PRIOR_FEATURE_MODES,
    research_feature_control_indices,
    research_feature_control_width,
    validate_research_feature_prior,
)
from wnn.splits import file_sha256
from wnn.training.initializer import final_initializer_state_sha256, make_final_initializer_teacher
from wnn.training.inputs import fit_research_feature_prior
from wnn.training.normalization import fit_e540_numeric_preprocessing, fit_normalization
from wnn.training.state import TrainingRun
from wnn.training_factorized import configure_factorized
from wnn.training_qat import configure_qat


def fit_preprocessing(run: TrainingRun) -> None:
    """Fit or inherit preprocessing using only the selected fitting groups."""
    environment = run.environment
    ap = run.options.ap
    args = run.options.args
    campaign_split = run.data.campaign_split
    device = run.options.device
    extra_valid = run.data.extra_valid
    frontier_contract = run.data.frontier_contract
    init = run.network.init
    initializer_teacher = run.network.initializer_teacher
    model = run.network.model
    provenance = run.data.provenance
    saved_training = run.data.saved_training
    train_idx = run.data.train_idx
    train_loader = run.data.train_loader
    train_split = run.data.train_split
    if environment.inputs.research_features in PRIOR_FEATURE_MODES:
        if (
            init is not None
            and init.get("args", {}).get("research_features", "none")
            == environment.inputs.research_features
        ):
            parent_prior = init.get("research_feature_prior")
            if not isinstance(parent_prior, dict):
                ap.error("feature initializer lacks its fitting-only prior")
            environment.inputs.research_feature_prior = validate_research_feature_prior(
                environment.inputs.research_features, parent_prior
            )
        else:
            environment.inputs.research_feature_prior = fit_research_feature_prior(
                train_split,
                np.asarray(train_idx),
                None if extra_valid is None else np.asarray(extra_valid),
                args.seed,
                environment=environment,
            )
        provenance["causal_feature_prior"] = environment.inputs.research_feature_prior
        prior_sources = [environment.source_root / "src/wnn/research_features.py"]
        if environment.inputs.research_features == "causal_geometry_discrete":
            prior_sources.append(environment.source_root / "src/wnn/moment_geometry/online.py")
        if environment.inputs.research_features == I1_HISTORY_MODE:
            prior_sources.extend(
                (
                    environment.source_root / "src/wnn/i1_history.py",
                    environment.source_root / "src/wnn/i1_history_scan.py",
                    environment.source_root / "src/wnn/i1_history_streaming.py",
                    environment.source_root / "src/wnn/features.py",
                    environment.source_root / "src/wnn/training/inputs.py",
                )
            )
        provenance["causal_feature_prior_source_sha256"] = {
            str(path.relative_to(environment.source_root)): file_sha256(path)
            for path in prior_sources
        }
        zero_width = research_feature_control_width(
            environment.inputs.research_features, environment.inputs.research_feature_control
        )
        zeroed_indices = research_feature_control_indices(
            environment.inputs.research_features, environment.inputs.research_feature_control
        )
        if environment.inputs.research_features == I1_HISTORY_MODE:
            model_input_indices = i1_history_input_mask_indices(
                environment.inputs.research_feature_control, bool(args.drop_i1)
            )
            provenance["causal_feature_control"] = {
                "kind": environment.inputs.research_feature_control,
                "drop_i1": bool(args.drop_i1),
                "zeroed_feature_count": len(model_input_indices),
                "zeroed_feature_indices": list(zeroed_indices),
                "zeroed_model_input_indices": list(model_input_indices),
                "mask_point": "after stored model normalization",
                "effective_capacity": (
                    "matched_full_width" if model_input_indices else "full_features"
                ),
            }
        else:
            provenance["causal_feature_control"] = {
                "kind": environment.inputs.research_feature_control,
                "zeroed_feature_count": zero_width,
                "zeroed_feature_indices": list(zeroed_indices),
                "effective_capacity": "matched_full_width" if zero_width else "full_features",
            }
    elif environment.inputs.research_features != "none":
        mode = environment.inputs.research_features
        zeroed_indices = research_feature_control_indices(
            mode, environment.inputs.research_feature_control
        )
        provenance["causal_feature_control"] = {
            "kind": environment.inputs.research_feature_control,
            "zeroed_feature_count": len(zeroed_indices),
            "zeroed_feature_indices": list(zeroed_indices),
            "effective_capacity": "matched_full_width" if zeroed_indices else "full_features",
        }
    if not args.init and saved_training is None:
        if frontier_contract is None:
            fit_normalization(model, train_loader, device, environment=environment)
        else:
            mean, std, normalization_source = student_frontier.load_normalization_buffers(
                environment.root, frontier_contract, campaign_split
            )
            student_frontier.install_normalization(model, mean, std)
            provenance["student_frontier"]["normalization_source"] = normalization_source
    if args.numeric_map != "none":
        fit_groups_train = train_idx.tolist() if train_split == "train" else []
        fit_groups_valid = (
            train_idx.tolist()
            if train_split == "valid"
            else extra_valid.tolist()
            if extra_valid is not None
            else []
        )
        numeric_preprocessing = fit_e540_numeric_preprocessing(
            model,
            fit_groups_train,
            fit_groups_valid,
            campaign_split.sha256,
            environment=environment,
        )
        if numeric_preprocessing is None:
            raise RuntimeError("E540 numeric map disappeared during model construction")
        provenance["numeric_preprocessing"] = numeric_preprocessing
        provenance["trainer_source_sha256"] = file_sha256(
            environment.source_root / "scripts/train_rnn.py"
        )
        provenance["model_source_sha256"] = file_sha256(environment.source_root / "src/wnn/nn.py")
    configure_factorized(model, args.recurrent_rank)
    if args.precision == "fp32":
        model._inference_fp32 = True
    initializer_teacher = None
    run.network.initializer_teacher = initializer_teacher


def configure_initializer_teacher(run: TrainingRun) -> None:
    """Clone a fixed FP32 teacher before configuring any student QAT."""
    environment = run.environment
    args = run.options.args
    initializer_teacher = run.network.initializer_teacher
    model = run.network.model
    provenance = run.data.provenance
    if args.final_distill_initializer:
        initializer_teacher = make_final_initializer_teacher(model)
        cache_provenance = provenance["distillation_cache"]
        provenance["final_initializer_distillation"] = {
            "version": 1,
            "initializer_reference": args.init,
            "initializer_checkpoint_sha256": provenance["initial_checkpoint_sha256"],
            "initializer_fp32_state_sha256": final_initializer_state_sha256(initializer_teacher),
            "source_sha256": {
                "trainer": file_sha256(Path(environment.entrypoint)),
                "model": file_sha256(environment.source_root / "src/wnn/nn.py"),
                "features": file_sha256(environment.source_root / "src/wnn/features.py"),
                "qat": file_sha256(environment.source_root / "src/wnn/training_qat.py"),
                "quantization": file_sha256(environment.source_root / "src/wnn/quantization.py"),
            },
            "student_qat": args.qat,
            "student_qat_learn_scales": args.qat_learn_scales,
            "teacher_creation": "deepcopy after initializer and preprocessing load with _inference_fp32=true, before student QAT configuration",
            "teacher_execution": "independent FP32 model, eval, requires_grad=false, no_grad forward; held fixed for the full fit",
            "cache_main_values": "validated cache values are loaded into each batch, copied to a private device tensor, then fully overwritten chunk-by-chunk before the main loss; cache files remain unchanged and cached main values are not used as targets",
            "cache_metadata_sha256": cache_provenance["metadata_sha256"],
            "cache_verified_files": cache_provenance["verified_files"],
            "causal_state": "initializer hidden state resets to None for each sequence batch; ordered chunks consume only current/past inputs and carry detached hidden state forward",
            "target_formula": "0.1*clamp(label,-2,2) + 0.9*clamp(initializer_prediction[...,0:2],-2,2)",
            "main_loss_weights": "abs(clamp(label,-2,2))*main_valid; configured auxiliary terms are unchanged",
            "continuation": "unsupported; --final-model-adaptation requires an epoch-zero start and the initializer teacher is rebuilt from the pinned source checkpoint",
        }
    configure_qat(model, args.qat, args.qat_learn_scales)
    run.network.initializer_teacher = initializer_teacher
