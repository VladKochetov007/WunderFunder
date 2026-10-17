from __future__ import annotations

import importlib
from collections.abc import Mapping
from functools import wraps

# Checkpoint payloads and experimental backend states have heterogeneous schemas.
from typing import Any

from wnn.training.compatibility_classes import environment_bound_class
from wnn.training.environment import InputConfiguration, ReadoutConfiguration, TrainingEnvironment
from wnn.training.forward import forward_chunk
from wnn.training.models import make_model
from wnn.training.runner import run_training
from wnn.training_dynamics import build_optimizer

EXPORTS = {
    "serialized_args": "checkpoints",
    "inference_checkpoint_fields": "checkpoints",
    "profile_update_evidence": "profiling",
    "StageTimer": "profiling",
    "Sequences": "datasets",
    "thin_rows": "row_views",
    "true_mask_weights": "targets",
    "episode_redundancy_weight": "targets",
    "main_row_weights": "targets",
    "bind_fold_teachers": "cache_identity",
    "prop_cache_path": "cache_identity",
    "CudaPrefetcher": "prefetch",
    "FastEMA": "averaging",
    "build_inputs": "inputs",
    "fit_research_feature_prior": "inputs",
    "next_change_labels": "targets",
    "next_label_change": "targets",
    "kth_change_labels": "targets",
    "shifted_targets": "targets",
    "calibrate_distill_reference": "calibration",
    "fit_normalization": "normalization",
    "iter_e540_fit_features": "normalization",
    "fit_e540_numeric_preprocessing": "normalization",
    "load_component_fit_subset": "membership",
    "load_e540_fit_subset": "membership",
    "forward_chunk": "forward",
    "detach_state": "forward",
    "final_initializer_distillation_error": "initializer",
    "final_initializer_state_sha256": "initializer",
    "final_initializer_distillation_target": "initializer",
    "make_final_initializer_teacher": "initializer",
    "replace_final_initializer_chunk": "initializer",
    "apply_block_memory_defaults": "model_validation",
    "block_memory_rejection": "model_validation",
    "independent_width_configuration_error": "model_validation",
    "make_model": "models",
    "predict": "evaluation",
    "evaluate": "evaluation",
    "configure_checkpoint_policy": "checkpoints",
    "checkpoint_selected": "checkpoints",
    "checkpoint_model": "checkpoints",
    "inherit_init_arguments": "inheritance",
    "can_extend_research_features": "inheritance",
    "expand_research_feature_state": "inheritance",
    "validate_campaign_checkpoint": "inheritance",
    "resume_contract": "checkpoints",
}
INPUT_FIELDS = {
    "INPUT_PROPENSITY": "propensity_model",
    "ONLINE_NORM": "online_norm",
    "HEURISTICS": "heuristics",
    "RAW_ONLY": "raw_only",
    "RESEARCH_FEATURES": "research_features",
    "RESEARCH_FEATURE_PRIOR": "research_feature_prior",
    "RESEARCH_FEATURE_CONTROL": "research_feature_control",
    "PRIVILEGED_MODE": "privileged_mode",
    "PRIVILEGED_HORIZON": "privileged_horizon",
    "SENTINEL_PREPROCESSOR": "sentinel_preprocessor",
}
ENVIRONMENT_EXPORTS = {
    "thin_rows",
    "fit_research_feature_prior",
    "resume_contract",
    "predict",
    "iter_e540_fit_features",
    "fit_normalization",
    "calibrate_distill_reference",
    "Sequences",
    "build_inputs",
    "main_row_weights",
    "bind_fold_teachers",
    "fit_e540_numeric_preprocessing",
    "evaluate",
    "prop_cache_path",
    "inference_checkpoint_fields",
    "replace_final_initializer_chunk",
}
DEPENDENCY_EXPORTS = {
    "argparse": ("argparse", None),
    "copy": ("copy", None),
    "hashlib": ("hashlib", None),
    "json": ("json", None),
    "math": ("math", None),
    "os": ("os", None),
    "platform": ("platform", None),
    "resource": ("resource", None),
    "sys": ("sys", None),
    "time": ("time", None),
    "Iterable": ("collections.abc", "Iterable"),
    "Path": ("pathlib", "Path"),
    "Any": ("typing", "Any"),
    "np": ("numpy", None),
    "torch": ("torch", None),
    "F": ("torch.nn.functional", None),
    "DataLoader": ("torch.utils.data", "DataLoader"),
    "Dataset": ("torch.utils.data", "Dataset"),
    "WeightedRandomSampler": ("torch.utils.data", "WeightedRandomSampler"),
    "student_frontier": ("wnn", "student_frontier"),
    "BlockMemoryAttention": ("wnn.block_memory_attention", "BlockMemoryAttention"),
    "CORRUPT_TRAIN_ROW_GROUPS": ("wnn.data", "CORRUPT_TRAIN_ROW_GROUPS"),
    "N_FEATURES": ("wnn.data", "N_FEATURES"),
    "SEQUENCE_LENGTH": ("wnn.data", "SEQUENCE_LENGTH"),
    "TARGET_QUANTA": ("wnn.data", "TARGET_QUANTA"),
    "WARMUP": ("wnn.data", "WARMUP"),
    "SequenceFile": ("wnn.data", "SequenceFile"),
    "market_class": ("wnn.data", "market_class"),
    "append_family_output_head": ("wnn.distillation", "append_family_output_head"),
    "extend_family_output_head": ("wnn.distillation", "extend_family_output_head"),
    "four_family_weighted_mse": ("wnn.distillation", "four_family_weighted_mse"),
    "validate_distillation_cache": ("wnn.distillation_cache", "validate_distillation_cache"),
    "EventContentModel": ("wnn.event_content", "EventContentModel"),
    "FEATURE_NAMES": ("wnn.features", "FEATURE_NAMES"),
    "N_EVENT_FEATURES": ("wnn.features", "N_EVENT_FEATURES"),
    "event_features_torch": ("wnn.features", "event_features_torch"),
    "online_zscore_torch": ("wnn.features", "online_zscore_torch"),
    "block_aggregate_score": ("wnn.metric", "block_aggregate_score"),
    "block_scores": ("wnn.metric", "block_scores"),
    "score": ("wnn.metric", "score"),
    "DualClockModel": ("wnn.nn", "DualClockModel"),
    "GatedRNNModel": ("wnn.nn", "GatedRNNModel"),
    "PooledCorrMemory": ("wnn.nn", "PooledCorrMemory"),
    "RNNModel": ("wnn.nn", "RNNModel"),
    "TCNModel": ("wnn.nn", "TCNModel"),
    "TwoClockModel": ("wnn.nn", "TwoClockModel"),
    "weighted_corr_loss": ("wnn.nn", "weighted_corr_loss"),
    "weighted_huber_loss": ("wnn.nn", "weighted_huber_loss"),
    "weighted_mse_loss": ("wnn.nn", "weighted_mse_loss"),
    "weighted_pooled_corr_loss": ("wnn.nn", "weighted_pooled_corr_loss"),
    "NumericFeatureMap": ("wnn.numeric_ple", "NumericFeatureMap"),
    "fit_training_quantile_knots": ("wnn.numeric_ple", "fit_training_quantile_knots"),
    "GradStats": ("wnn.optim", "GradStats"),
    "rnn_init_": ("wnn.optim", "rnn_init_"),
    "N_CODES": ("wnn.outcomes", "N_CODES"),
    "N_SP_AUX": ("wnn.outcomes", "N_SP_AUX"),
    "ReadoutMoments": ("wnn.outcomes", "ReadoutMoments"),
    "outcome_codes": ("wnn.outcomes", "outcome_codes"),
    "readout": ("wnn.outcomes", "readout"),
    "sequence_class": ("wnn.outcomes", "sequence_class"),
    "structure_labels": ("wnn.outcomes", "structure_labels"),
    "PRIVILEGED_MODES": ("wnn.privileged", "PRIVILEGED_MODES"),
    "load_inner_manifest": ("wnn.privileged", "load_inner_manifest"),
    "privileged_inputs": ("wnn.privileged", "privileged_inputs"),
    "validate_crossfit_cache": ("wnn.privileged", "validate_crossfit_cache"),
    "load_propensity": ("wnn.propensity", "load_propensity"),
    "propensity": ("wnn.propensity", "propensity"),
    "policy_metadata": ("wnn.quantization", "policy_metadata"),
    "FIELD_AGE_FEATURE_MODES": ("wnn.research_features", "FIELD_AGE_FEATURE_MODES"),
    "MOMENT_FEATURE_MODES": ("wnn.research_features", "MOMENT_FEATURE_MODES"),
    "RESEARCH_FEATURE_MODES": ("wnn.research_features", "RESEARCH_FEATURE_MODES"),
    "fit_causal_moment_prior": ("wnn.research_features", "fit_causal_moment_prior"),
    "research_feature_control_width": ("wnn.research_features", "research_feature_control_width"),
    "research_feature_names": ("wnn.research_features", "research_feature_names"),
    "research_features_torch": ("wnn.research_features", "research_features_torch"),
    "validate_causal_moment_prior": ("wnn.research_features", "validate_causal_moment_prior"),
    "configure_torch": ("wnn.resources", "configure_torch"),
    "cpu_threads": ("wnn.resources", "cpu_threads"),
    "dataloader_budget": ("wnn.resources", "dataloader_budget"),
    "EpochUpdateCounter": ("wnn.run_identity", "EpochUpdateCounter"),
    "group_identity": ("wnn.run_identity", "group_identity"),
    "normalization_identity": ("wnn.run_identity", "normalization_identity"),
    "sample_order_identity": ("wnn.run_identity", "sample_order_identity"),
    "scaler_source_identity": ("wnn.run_identity", "scaler_source_identity"),
    "INDICATOR_MODES": ("wnn.sentinel", "INDICATOR_MODES"),
    "VALUE_MODES": ("wnn.sentinel", "VALUE_MODES"),
    "SentinelInputProjection": ("wnn.sentinel", "SentinelInputProjection"),
    "SentinelPreprocessor": ("wnn.sentinel", "SentinelPreprocessor"),
    "validate_sentinel_fit": ("wnn.sentinel_calibration", "validate_fit"),
    "PrefixAdapterGRUModel": ("wnn.specialist", "PrefixAdapterGRUModel"),
    "SpecialistGRUModel": ("wnn.specialist", "SpecialistGRUModel"),
    "CampaignSplit": ("wnn.splits", "CampaignSplit"),
    "dev_halves": ("wnn.splits", "dev_halves"),
    "file_sha256": ("wnn.splits", "file_sha256"),
    "load_campaign_split": ("wnn.splits", "load_campaign_split"),
    "valid_row_groups": ("wnn.splits", "valid_row_groups"),
    "PackedSpreadCache": ("wnn.spread_aux", "PackedSpreadCache"),
    "structural_loss": ("wnn.spread_aux", "structural_loss"),
    "validate_spread_cache": ("wnn.spread_aux", "validate_spread_cache"),
    "validate_spread_parent": ("wnn.spread_aux", "validate_spread_parent"),
    "StructuredStateModel": ("wnn.structured_state", "StructuredStateModel"),
    "DistilledLoss": ("wnn.training_controls", "DistilledLoss"),
    "blended_target": ("wnn.training_controls", "blended_target"),
    "detach_boundary_offset": ("wnn.training_controls", "detach_boundary_offset"),
    "distilled_macro_losses": ("wnn.training_controls", "distilled_macro_losses"),
    "ema_decay_per_update": ("wnn.training_controls", "ema_decay_per_update"),
    "shuffled_masses": ("wnn.training_controls", "shuffled_masses"),
    "credit_edge_scales": ("wnn.training_credit", "credit_edge_scales"),
    "Dynamics": ("wnn.training_dynamics", "Dynamics"),
    "build_optimizer": ("wnn.training_dynamics", "build_optimizer"),
    "build_scheduler": ("wnn.training_dynamics", "build_scheduler"),
    "switch_optimizer": ("wnn.training_dynamics", "switch_optimizer"),
    "add_dynamics_arguments": ("wnn.training_dynamics", "add_arguments"),
    "configure_factorized": ("wnn.training_factorized", "configure_factorized"),
    "factorized_forward": ("wnn.training_factorized", "factorized_forward"),
    "factorized_inference_state": ("wnn.training_factorized", "factorized_inference_state"),
    "QAT_LEARN_SCALE_MODES": ("wnn.training_qat", "QAT_LEARN_SCALE_MODES"),
    "QAT_MODES": ("wnn.training_qat", "QAT_MODES"),
    "configure_qat": ("wnn.training_qat", "configure_qat"),
    "inference_state": ("wnn.training_qat", "inference_state"),
    "invalidate_qat_cache": ("wnn.training_qat", "invalidate_qat_cache"),
    "project_qat_scales_": ("wnn.training_qat", "project_qat_scales_"),
    "qat_forward": ("wnn.training_qat", "qat_forward"),
    "qat_weight_replacements": ("wnn.training_qat", "qat_weight_replacements"),
    "scale_learning_metadata": ("wnn.training_qat", "scale_learning_metadata"),
    "EpochOrderSampler": ("wnn.training_sampling", "EpochOrderSampler"),
    "init_qmc_": ("wnn.training_sampling", "init_qmc_"),
    "atomic_save_training_state": ("wnn.training_state", "atomic_save_training_state"),
    "capture_training_state": ("wnn.training_state", "capture_training_state"),
    "load_training_state": ("wnn.training_state", "load_training_state"),
    "loader_state": ("wnn.training_state", "loader_state"),
    "restore_rng": ("wnn.training_state", "restore_rng"),
    "restore_training_state": ("wnn.training_state", "restore_training_state"),
    "origin_windows": ("wnn.training_views", "origin_windows"),
    "update_ema_for_rows": ("wnn.training_views", "update_ema_for_rows"),
}


class LegacyTrainerAPI:
    """Adapt historical imported settings at explicit API boundaries only."""

    def __init__(self, namespace: Mapping[str, Any]) -> None:
        self.namespace = namespace
        self._class_exports: dict[str, type[Any]] = {}

    def environment(self) -> TrainingEnvironment:
        """Snapshot caller settings without mutating another module."""
        ns = self.namespace
        values = {field: ns[flag] for flag, field in INPUT_FIELDS.items()}
        return TrainingEnvironment(
            source_root=ns["SOURCE_ROOT"],
            root=ns["ROOT"],
            inputs=InputConfiguration(**values),
            readout=ReadoutConfiguration(ns["READOUT"], ns["READOUT_TILTED"]),
            sequence_length=ns["SEQUENCE_LENGTH"],
            warmup=ns["WARMUP"],
            corrupt_train_groups=frozenset(ns["CORRUPT_TRAIN_ROW_GROUPS"]),
            sequence_file=ns["SequenceFile"],
            valid_row_groups=ns["valid_row_groups"],
            dev_halves=ns["dev_halves"],
            propensity=ns["propensity"],
            forward_chunk=ns.get("forward_chunk", forward_chunk),
            build_optimizer=ns.get("build_optimizer", build_optimizer),
            make_model=ns.get("make_model", make_model),
            prediction=ns.get("predict"),
        )

    def attribute(self, name: str) -> Any:
        """Resolve documented exports, including custom-name importlib consumers."""
        if name in DEPENDENCY_EXPORTS:
            module_name, attribute = DEPENDENCY_EXPORTS[name]
            imported = importlib.import_module(module_name)
            return imported if attribute is None else getattr(imported, attribute)
        if name not in EXPORTS:
            raise AttributeError(f"No training export named {name!r}")
        target = getattr(importlib.import_module(f"wnn.training.{EXPORTS[name]}"), name)
        if name not in ENVIRONMENT_EXPORTS:
            return target
        if isinstance(target, type):
            return self._class_export(name, target)

        @wraps(target)
        def invoke(*args: Any, **kwargs: Any) -> Any:
            """Adapt a legacy call to explicit environment injection."""
            kwargs.setdefault("environment", self.environment())
            return target(*args, **kwargs)

        return invoke

    def _class_export(self, name: str, target: type[Any]) -> type[Any]:
        """Inject legacy settings while keeping class-valued exports as classes."""
        if name not in self._class_exports:
            self._class_exports[name] = environment_bound_class(
                target,
                self.environment,
                str(self.namespace.get("__name__", target.__module__)),
            )
        return self._class_exports[name]

    def main(self) -> None:
        """Run with explicitly captured legacy settings and I/O dependencies."""
        run_training(environment=self.environment())
