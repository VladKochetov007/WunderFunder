from __future__ import annotations

import json
import math
import os

import torch

from wnn.quantization import policy_metadata
from wnn.run_identity import group_identity, normalization_identity, scaler_source_identity
from wnn.splits import file_sha256
from wnn.training.reporting import emit
from wnn.training.state import TrainingRun
from wnn.training_qat import scale_learning_metadata


def configure_activation_backend(run: TrainingRun) -> None:
    """Attach the optional compiled activation backend without changing its tensor graph."""
    ap = run.options.ap
    args = run.options.args
    device = run.options.device
    model = run.network.model
    provenance = run.data.provenance
    if args.activation_backend != "none":
        from wnn.activation_qat import ActivationQATModel

        if args.qat_learn_scales:
            ap.error("activation scan currently profiles fixed-scale weight QAT only")
        if not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
            ap.error("activation scan backend requires TORCHINDUCTOR_CACHE_DIR on physical storage")
        if device.type == "cuda":
            import torch._inductor.config as inductor_config

            inductor_config.compile_threads = 1
        finite_model_tensors = [torch.isfinite(value).all() for value in model.parameters()] + [
            torch.isfinite(model.mean).all(),
            torch.isfinite(model.std).all(),
        ]
        if not bool(torch.stack(finite_model_tensors).all().item()):
            raise ValueError("activation scan requires finite model parameters and normalization")
        activation_model = ActivationQATModel(
            model, quantize_activations=args.activation_backend == "scan_a8"
        )
        object.__setattr__(
            model, "_activation_qat_backend", torch.compile(activation_model, fullgraph=True)
        )
        if args.activation_profile_first_chunk:
            object.__setattr__(model, "_activation_profile_cache_fingerprints", [])
            object.__setattr__(
                model, "_activation_profile_compile_timings", {"forward": None, "backward": None}
            )
        provenance["activation_backend"] = {
            "kind": args.activation_backend,
            "implementation": "compiled torch higher-order scan",
            "activation_scope": "post-encoder/per-layer input-to-hidden and previous-hidden recurrent matvec vectors"
            if args.activation_backend == "scan_a8"
            else "same scan recurrence without activation quantization",
            "state": "fp32 recurrent state",
            "excluded": ["raw input projection/I16 export", "head hidden/projection A8 blocks"],
            "weight_qat": args.qat,
        }


def record_model_identity(run: TrainingRun) -> None:
    """Record preprocessing/QAT identity, freeze prefixes and calculate the unchanged update budget."""
    environment = run.environment
    args = run.options.args
    model = run.network.model
    provenance = run.data.provenance
    saved_training = run.data.saved_training
    total = run.network.total
    train_ds = run.data.train_ds
    train_loader = run.data.train_loader
    update_rows = run.options.update_rows
    if saved_training is not None:
        model.load_state_dict(saved_training["model"])
    if args.qat != "none":
        provenance["qat"] = {
            **policy_metadata(args.qat),
            "gradient": "identity weight STE; learned row scale derivative"
            if args.qat_learn_scales
            else "identity STE; scales detached; no scale/round/clamp derivative",
            "averaging": "latent float parameters, quantized for each effective forward/export",
            "training_state": "raw latent float weights; derived quantization cache not serialized",
        }
        if args.qat_learn_scales:
            provenance["qat"]["scale_learning"] = scale_learning_metadata(args.optim, args.wd)
    if args.recurrent_rank:
        provenance["factorization"] = {
            "rank": args.recurrent_rank,
            "initialization": model._factorization_initialization,
            "averaging": "latent factors; products formed after averaging",
            "training_compute": "materialized dense FP32 GRU; native runtime not validated",
        }
    provenance["normalization_identity"] = normalization_identity(model)
    provenance["update_counter_implementation"] = scaler_source_identity()
    provenance["run_identity_source_sha256"] = file_sha256(
        environment.source_root / "src/wnn/run_identity.py"
    )
    provenance["fit_membership_identity"] = {
        "train": group_identity(provenance["fit_row_groups"]["train"]),
        "valid": group_identity(provenance["fit_row_groups"]["valid"]),
        "valid_repeat": args.valid_repeat,
        "dataset_length": len(train_ds),
    }
    if args.profile_steps and args.activation_backend != "none":
        provenance["profile_source_hashes"] = {
            "trainer": file_sha256(environment.source_root / "scripts/train_rnn.py"),
            "activation_backend": file_sha256(
                environment.source_root / "src/wnn/activation_qat.py"
            ),
            "weight_qat": file_sha256(environment.source_root / "src/wnn/training_qat.py"),
            "gru_model": file_sha256(environment.source_root / "src/wnn/nn.py"),
        }
        emit(
            "PROFILE_RUN_PROVENANCE",
            json.dumps(
                {
                    "initial_checkpoint_sha256": provenance.get("initial_checkpoint_sha256"),
                    "split_manifest_sha256": provenance.get("split_manifest_sha256"),
                    "fit_subset_manifest_sha256": provenance.get("fit_subset_manifest_sha256"),
                    "fit_subset_members_sha256": provenance.get("fit_subset_members_sha256"),
                    "fit_membership_identity": provenance["fit_membership_identity"],
                    "normalization_identity": provenance["normalization_identity"],
                    "distillation_cache": provenance.get("distillation_cache"),
                    "propensity_checkpoint_sha256": provenance.get("propensity_checkpoint_sha256"),
                    "propensity_cache": provenance.get("propensity_cache"),
                    "source_sha256": provenance["profile_source_hashes"],
                    "activation_backend": provenance.get("activation_backend"),
                    "weight_qat": provenance.get("qat"),
                }
            ),
            flush=True,
        )
    frozen = tuple(v for v in args.freeze.split(",") if v)
    for pname, prm in model.named_parameters():
        if frozen and pname.startswith(frozen):
            prm.requires_grad_(False)
    if args.qat_learn_scales:
        parameters = dict(model.named_parameters())
        for position, (name, _, _) in enumerate(model._qat_scale_keys):
            model._qat_scale_params[position].requires_grad_(parameters[name].requires_grad)
    total = len(train_loader) * math.ceil(environment.sequence_length / update_rows) * args.epochs
    run.network.total = total
