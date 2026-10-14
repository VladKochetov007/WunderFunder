"""Strict consumer-side contract for the CPU student architecture frontier.

This module deliberately does not alter the producer's historical distillation
cache contract. It validates one separately versioned recipe after that cache
has passed its original validator, then copies only fit-derived normalization
buffers from the pinned reference checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.run_identity import group_identity, tensor_identity
from wnn.splits import CampaignSplit, file_sha256
from wnn.student_frontier_order import (
    members_sha256,
    positions_sha256,
    seeded_epoch_positions,
    source_qualified_members,
    source_qualified_target_positions,
)

CONTRACT_FORMAT = "wnn.cpu-student-frontier-consumer-contract"
CONTRACT_VERSION = 1
MAX_CONTRACT_BYTES = 256_000
MAX_PROPENSITY_SIDECAR_BYTES = 64_000
MAX_PROVENANCE_BYTES = 8_000_000


def _read_bounded_json(path: Path, max_bytes: int, description: str) -> dict[str, Any]:
    """Parse a small provenance JSON only after checking its file-size bound."""
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"{description} exceeds its {max_bytes}-byte read bound")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"{description} changed while being read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def load_contract(path: Path) -> tuple[dict[str, Any], str]:
    """Read a bounded consumer recipe and return its contents and exact digest."""
    path = path.resolve(strict=True)
    size = path.stat().st_size
    if size <= 0 or size > MAX_CONTRACT_BYTES:
        raise ValueError("student-frontier consumer contract exceeds the 256 KB bound")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError("student-frontier consumer contract changed while being read")
    contract = json.loads(raw)
    if (not isinstance(contract, dict)
            or contract.get("format") != CONTRACT_FORMAT
            or contract.get("version") != CONTRACT_VERSION):
        raise ValueError("unsupported student-frontier consumer contract")
    if not isinstance(contract.get("id"), str) or not contract["id"]:
        raise ValueError("student-frontier consumer contract needs a stable id")
    return contract, hashlib.sha256(raw).hexdigest()


def validate_stage_args(
    args: Any, contract: dict[str, Any], stage_name: str, root: Path
) -> dict[str, Any]:
    """Reject any trainer setting outside the selected frozen consumer stage."""
    stages = contract.get("stages")
    if not isinstance(stages, dict) or stage_name not in stages:
        raise ValueError(f"student-frontier stage is not admitted: {stage_name}")
    stage = stages[stage_name]
    if not isinstance(stage, dict) or stage.get("kind") not in {"profile", "fit"}:
        raise ValueError("student-frontier stage must declare profile or fit kind")

    shapes = {tuple(pair) for pair in contract.get("architecture_ladder", [])}
    shape = (int(args.hidden), int(args.layers))
    if shape not in shapes:
        raise ValueError(f"student-frontier shape is not admitted: {shape}")
    recipe = contract["recipe"]
    optimizer = recipe["optimizer"]
    loader = recipe["loader"]

    expected: dict[str, object] = {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "dropout": 0.0,
        "train_source": "train+valid",
        "train_seqs": 10607,
        "valid_train_fraction": 1.0,
        "valid_repeat": 1,
        "valid_fold": 0,
        "n_valid_folds": 4,
        "full_valid": True,
        "include_heldout_valid": False,
        "class_balance": False,
        "train_classes": "",
        "epoch_samples": 0,
        "batch": int(recipe["batch_size"]),
        "chunk": 2000,
        "update_rows": 0,
        "detach_rows": 0,
        "batch_order": "seeded",
        "antithetic": False,
        "seed": int(stage["seed"]),
        "precision": "fp32",
        "epochs": int(stage["epochs"]),
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": int(stage["checkpoint_epoch"]),
        "no_eval_during_fit": True,
        "init": "",
        "inherit_init_preprocessing": False,
        "rnn_init": "default",
        "loss": "wmse",
        "loss_norm": "chunk",
        "distill_cache_variant": "homogeneous",
        "distill_cache_scale": 1.0,
        "distill_alpha": 0.9,
        "distill_alpha_far": None,
        "distill_boot_h": 0,
        "distill_reanchor": 0.0,
        "distill_denominator": "chunk",
        "distill_family_weight": 0.0,
        "since_weight": 0.0,
        "propensity": contract["propensity"]["run"],
        "prop_cache": True,
        "prop_power": 2.0,
        "prop_eps": float(contract["propensity"]["epsilon"]),
        "valid_true_mask": 0.03,
        "prop_input": "",
        "mask_weight": False,
        "class_weight": "",
        "aux": "",
        "aux_next": 0.0,
        "aux_next_k": "",
        "aux_tier": 0.0,
        "aux_mask": 0.0,
        "aux_sign": 0.0,
        "aux_quote": 0.0,
        "aux_struct": 0.0,
        "ema_teacher": 0.0,
        "td_alpha": 0.0,
        "swa_frac": 0.0,
        "eval_ema": False,
        "qat": "none",
        "recurrent_rank": 0,
        "numeric_map": "none",
        "online_norm": False,
        "heuristics": False,
        "raw_only": False,
        "research_features": "none",
        "privileged_mode": "none",
        "quote_cache_root": "",
        "spread_cache_root": "",
        "sentinel_base_width": 0,
        "sentinel_indicators": "none",
        "freeze": "",
        "optim": "adamw",
        "sched": "onecycle",
        "optim_switch": "",
        "head_lr_mult": 1.0,
        "warmup_frac": float(recipe["schedule"]["warmup_fraction"]),
        "no_gradient_clip": False,
        "grad_noise": 0.0,
        "zoneout": 0.0,
        "momentum": 0.0,
        "lr": float(optimizer["learning_rate"]),
        "beta2": float(optimizer["beta2"]),
        "wd": float(optimizer["weight_decay"]),
        "clip": float(optimizer["gradient_clip_norm"]),
        "fused_adam": bool(optimizer["fused"]),
        "sync_free": bool(optimizer["sync_free"]),
        "pin_memory": bool(loader["pin_memory"]),
        "deterministic": True,
        "save_training_state": False,
        "resume_state": None,
        "max_batches": 1 if stage["kind"] == "profile" else 0,
        "profile_steps": 1 if stage["kind"] == "profile" else 0,
        "profile_warmup": 0 if stage["kind"] == "profile" else 5,
        "grad_stats": True,
    }
    for name, value in expected.items():
        actual = getattr(args, name, None)
        if actual != value:
            raise ValueError(
                f"student-frontier {stage_name} requires {name}={value!r}, got {actual!r}"
            )

    prefix = stage.get("run_name_prefix")
    if not isinstance(prefix, str) or not str(args.name).startswith(prefix):
        raise ValueError("student-frontier run name is outside the frozen stage namespace")

    source_root = Path(contract["source_root"]).resolve()
    if source_root != Path(__file__).resolve().parents[2]:
        raise ValueError("student-frontier consumer code root differs from its contract")
    expected_sources = contract["source_sha256"]
    if not isinstance(expected_sources, dict) or not expected_sources:
        raise ValueError("student-frontier consumer recipe lacks source hashes")
    actual_sources = {
        relative_path: file_sha256(source_root / relative_path)
        for relative_path in expected_sources
    }
    if actual_sources != expected_sources:
        raise ValueError("student-frontier consumer source hashes differ from its contract")

    split = contract["split"]
    if Path(args.split_manifest).resolve() != (root / split["path"]).resolve():
        raise ValueError("student-frontier split manifest path differs from its contract")
    cache = contract["cache"]
    cache_path = Path(args.distill_cache_root)
    if not cache_path.is_absolute():
        cache_path = root / cache_path
    if cache_path.resolve() != (root / cache["root"]).resolve():
        raise ValueError("student-frontier distillation cache root differs from its contract")
    normalization = contract["normalization_source"]
    if contract.get("feature_width") != 211 or normalization.get("feature_width") != 211:
        raise ValueError("student-frontier common normalization must have exactly 211 features")
    return stage


def validate_training_order(
    args: Any,
    contract: dict[str, Any],
    stage: dict[str, Any],
    split: CampaignSplit,
    train_groups: np.ndarray,
    valid_groups: np.ndarray,
    train_loader: Any,
) -> dict[str, Any]:
    """Bind the actual epoch sampler to one source-qualified, fixed batch order."""
    recipe = contract["recipe"]
    batch_size = int(recipe["batch_size"])
    seed = int(stage["seed"])
    epoch = int(stage["checkpoint_epoch"])
    if (args.batch_order != "seeded" or args.valid_repeat != 1
            or train_loader.batch_size != batch_size or not train_loader.drop_last):
        raise ValueError("student-frontier requires its exact seeded drop-last loader")
    sampler = train_loader.sampler
    if type(sampler).__name__ != "EpochOrderSampler":
        raise ValueError("student-frontier loader must use EpochOrderSampler")

    positions = seeded_epoch_positions(len(train_loader.dataset), seed, batch_size, epoch)
    prior_epoch = sampler.epoch
    sampler.set_epoch(epoch)
    actual_positions = [int(position) for position in sampler]
    sampler.set_epoch(prior_epoch)
    if actual_positions != positions:
        raise ValueError("trainer sampler order differs from the frozen order implementation")

    all_members = source_qualified_members(train_groups, valid_groups, positions)
    batches = len(positions) // batch_size
    exposed = batches * batch_size
    fit_members = all_members[:exposed]
    dropped_members = all_members[exposed:]
    fit_train = set(map(int, train_groups))
    fit_valid = set(map(int, valid_groups))
    eval_valid = set(map(int, split.evaluation_groups))
    if (not set(map(int, train_groups)).issubset(split.train_groups)
            or not set(map(int, valid_groups)).issubset(split.valid_groups)):
        raise ValueError("student-frontier ordered membership differs from its source-qualified split")
    if any(
        group not in (fit_train if source == "train" else fit_valid)
        or source == "valid" and group in eval_valid
        for source, group in fit_members
    ):
        raise ValueError("student-frontier sampler includes a non-fitting source-qualified group")

    def batch_digest(members: list[tuple[str, int]]) -> str:
        encoded = "\n".join(
            f"{index // batch_size}:{source}:{group}"
            for index, (source, group) in enumerate(members)
        ).encode("ascii")
        return hashlib.sha256(encoded).hexdigest()

    observed = {
        "batch_size": batch_size,
        "seed": seed,
        "epoch": epoch,
        "fit_groups_before_drop_last": len(positions),
        "exposed_sequence_count": exposed,
        "dropped_sequence_count": len(dropped_members),
        "batches_per_epoch": batches,
        "chunks_per_sequence": int(recipe["sequence_rows"]) // int(recipe["chunk_rows"]),
        "optimizer_updates_per_epoch": (
            batches * int(recipe["sequence_rows"]) // int(recipe["chunk_rows"])
        ),
        "train_group_order_sha256": group_identity(train_groups)["ordered_int64_le_sha256"],
        "valid_group_order_sha256": group_identity(valid_groups)["ordered_int64_le_sha256"],
        "sampler_positions_sha256": positions_sha256(positions),
        "sampler_source_rows_sha256": members_sha256(all_members),
        "retained_source_rows_sha256": members_sha256(fit_members),
        "batch_membership_sha256": batch_digest(fit_members),
        "dropped_tail_positions": positions[exposed:],
        "dropped_tail_members": [list(member) for member in dropped_members],
        "same_order_and_omissions_across_shapes": True,
    }
    expected = contract["sample_order"]
    for key, value in observed.items():
        if expected.get(key) != value:
            raise ValueError(
                f"student-frontier source-qualified sample order differs at {key}: "
                f"expected {expected.get(key)!r}, got {value!r}"
            )
    if int(stage["expected_updates"]) != observed["optimizer_updates_per_epoch"]:
        raise ValueError("student-frontier stage update budget differs from the actual loader")
    return observed


def validate_cache_binding(
    root: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    split: CampaignSplit,
    cache_validation: dict[str, Any],
) -> dict[str, Any]:
    """Bind the strict legacy cache result to the separate frontier recipe."""
    split_contract = contract["split"]
    if split.fold != split_contract["fold"] or split.sha256 != split_contract["sha256"]:
        raise ValueError("student-frontier split identity differs from its contract")
    if Path(split.path).resolve() != (root / split_contract["path"]).resolve():
        raise ValueError("student-frontier split path differs from its contract")
    actual_membership = {
        "train": group_identity(split.train_groups),
        "valid": group_identity(split.valid_groups),
        "evaluation_valid": group_identity(split.evaluation_groups),
    }
    if actual_membership != split_contract["groups"]:
        raise ValueError("student-frontier source-qualified fitting/evaluation groups changed")
    split.assert_fit_groups(split.train_groups, split.valid_groups)
    # Row-group numbers are source-local. A train row-group may have the same
    # integer as a valid row-group; only compare memberships within one source.
    if set(split.valid_groups) & set(split.evaluation_groups):
        raise ValueError("student-frontier fitting and evaluation memberships overlap")

    cache_contract = contract["cache"]
    metadata = cache_validation["metadata"]
    if cache_validation["metadata_sha256"] != cache_contract["metadata_sha256"]:
        raise ValueError("student-frontier producer metadata hash differs from its contract")
    if (metadata.get("cache_profile") != cache_contract["profile"]
            or metadata.get("format") != cache_contract["format"]
            or metadata.get("version") != cache_contract["version"]
            or metadata.get("fold") != split.fold
            or metadata.get("split_manifest_sha256") != split.sha256
            or metadata.get("family_order") != ["strong"]
            or metadata.get("homogeneous_family") != "strong"):
        raise ValueError("student-frontier cache profile or fold identity changed")
    if metadata.get("target_contract") != cache_contract["target_contract"]:
        raise ValueError("student-frontier teacher-target arithmetic differs")
    if metadata.get("student_contract") != cache_contract["historical_student_contract"]:
        raise ValueError("historical producer student contract was rewritten")
    if metadata.get("adapter_contract_sha256") != cache_contract["adapter_contract_sha256"]:
        raise ValueError("student-frontier cache adapter ancestry differs")
    if (metadata.get("fit_groups") != {
            "train": list(split.train_groups), "valid": list(split.valid_groups)}
            or metadata.get("evaluation_valid_groups") != list(split.evaluation_groups)):
        raise ValueError("student-frontier cache fitting/evaluation memberships changed")

    verified_targets = cache_validation["verified_files"]
    for name, expected in cache_contract["targets"].items():
        entry = metadata.get("files", {}).get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"student-frontier cache is missing target {name}")
        for field in ("path", "sha256", "bytes", "shape", "dtype", "row_groups_sha256"):
            if entry.get(field) != expected.get(field):
                raise ValueError(f"student-frontier target identity changed: {name}.{field}")
        if verified_targets.get(name) != expected["sha256"]:
            raise ValueError(f"student-frontier target payload was not verified: {name}")

    try:
        cache_rows = source_qualified_target_positions(
            metadata["files"]["main_homogeneous_train"]["row_groups"],
            metadata["files"]["main_homogeneous_valid"]["row_groups"],
            split.train_groups,
            split.valid_groups,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("student-frontier cache rows do not map to source-qualified fit groups") from error
    if len(cache_rows) != len(split.train_groups) + len(split.valid_groups):
        raise ValueError("student-frontier cache rows omit a source-qualified fitting group")

    source_records = metadata.get("sources", [])
    expected_sources = cache_contract["teachers"]
    if len(source_records) != len(expected_sources):
        raise ValueError("student-frontier teacher source count changed")
    verified_sources = []
    for source, expected in zip(source_records, expected_sources, strict=True):
        identity = {key: source.get(key) for key in expected}
        if identity != expected:
            raise ValueError(f"student-frontier teacher ancestry changed: {source.get('run')}")
        run = str(expected["run"])
        checkpoint = root / "artifacts/runs" / run / f"model_ep{expected['epoch']}.pt"
        provenance = root / "artifacts/runs" / run / "provenance.json"
        if (file_sha256(checkpoint) != expected["checkpoint_sha256"]
                or file_sha256(provenance) != expected["provenance_sha256"]):
            raise ValueError(f"student-frontier teacher artifact changed: {run}")
        verified_sources.append(expected)

    propensity = contract["propensity"]
    propensity_model = root / "artifacts/runs" / propensity["run"] / "model.pt"
    propensity_provenance = root / "artifacts/runs" / propensity["run"] / "provenance.json"
    if (file_sha256(propensity_model) != propensity["checkpoint_sha256"]
            or file_sha256(propensity_provenance) != propensity["provenance_sha256"]):
        raise ValueError("student-frontier propensity model or provenance changed")
    verified_propensity_caches = {}
    for split_name, expected in propensity["caches"].items():
        cache_path = root / expected["path"]
        sidecar_path = root / expected["sidecar_path"]
        sidecar = _read_bounded_json(
            sidecar_path, MAX_PROPENSITY_SIDECAR_BYTES, f"{split_name} propensity sidecar"
        )
        if (file_sha256(cache_path) != expected["sha256"]
                or cache_path.stat().st_size != expected["bytes"]
                or file_sha256(sidecar_path) != expected["sidecar_sha256"]
                or sidecar.get("model_sha256") != propensity["checkpoint_sha256"]
                or sidecar.get("split") != split_name):
            raise ValueError(f"student-frontier propensity cache changed: {split_name}")
        verified_propensity_caches[split_name] = expected

    return {
        "contract_id": contract["id"],
        "contract_sha256": contract_sha256,
        "stage": None,
        "split_sha256": split.sha256,
        "source_qualified_group_identities": actual_membership,
        "cache_metadata_sha256": cache_validation["metadata_sha256"],
        "verified_target_files": verified_targets,
        "source_qualified_target_rows": len(cache_rows),
        "source_qualified_target_order": {
            "train": group_identity(metadata["files"]["main_homogeneous_train"]["row_groups"]),
            "valid": group_identity(metadata["files"]["main_homogeneous_valid"]["row_groups"]),
        },
        "target_formula": cache_contract["target_contract"]["formula"],
        "teacher_sources": verified_sources,
        "propensity": {
            "run": propensity["run"],
            "checkpoint_sha256": propensity["checkpoint_sha256"],
            "provenance_sha256": propensity["provenance_sha256"],
            "cache_files": verified_propensity_caches,
            "power": propensity["power"],
            "valid_true_mask_eps": propensity["valid_true_mask_eps"],
        },
        "target_semantics": contract["target_semantics"],
    }


def load_normalization_buffers(
    root: Path,
    contract: dict[str, Any],
    split: CampaignSplit,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Load only mean/std from the pinned H256 reference; never load its weights."""
    source = contract["normalization_source"]
    run = source["run"]
    checkpoint_path = root / "artifacts/runs" / run / source["checkpoint_file"]
    provenance_path = root / "artifacts/runs" / run / "provenance.json"
    if (file_sha256(checkpoint_path) != source["checkpoint_sha256"]
            or file_sha256(provenance_path) != source["provenance_sha256"]):
        raise ValueError("student-frontier normalization source changed")
    provenance = _read_bounded_json(
        provenance_path, MAX_PROVENANCE_BYTES, "normalization provenance"
    )
    if (provenance.get("split_manifest_sha256") != split.sha256
            or provenance.get("fit_membership_identity") != source["fit_membership_identity"]
            or provenance.get("normalization_identity") != source["normalization_identity"]):
        raise ValueError("student-frontier normalization provenance changed")
    fit_groups = provenance.get("fit_row_groups", {})
    if (set(fit_groups.get("train", [])) != set(split.train_groups)
            or set(fit_groups.get("valid", [])) != set(split.valid_groups)
            or set(provenance.get("evaluation_row_groups", [])) != set(split.evaluation_groups)):
        raise ValueError("student-frontier normalization source used different fold membership")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != source["epoch"]:
        raise ValueError("student-frontier normalization source checkpoint epoch changed")
    state = checkpoint.get("model")
    if not isinstance(state, dict) or "mean" not in state or "std" not in state:
        raise ValueError("student-frontier normalization source lacks mean/std buffers")
    mean = state["mean"].detach().to(dtype=torch.float32, device="cpu").clone()
    std = state["std"].detach().to(dtype=torch.float32, device="cpu").clone()
    del state, checkpoint
    identity = {
        "mean": tensor_identity(mean),
        "std": tensor_identity(std),
    }
    if (mean.shape != (211,) or std.shape != (211,)
            or identity != source["normalization_identity"]):
        raise ValueError("student-frontier H256 normalization buffers differ from the frozen identity")
    return mean, std, {
        "run": run,
        "epoch": source["epoch"],
        "checkpoint_file": source["checkpoint_file"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "provenance_sha256": source["provenance_sha256"],
        "normalization_identity": identity,
        "loaded_fields_only": ["mean", "std"],
        "learned_weights_loaded": False,
    }


def install_normalization(
    model: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    """Copy the common FP32 normalization buffers to a new student model."""
    if mean.shape != (211,) or std.shape != (211,):
        raise ValueError("student-frontier normalization buffers must each have width 211")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
        raise ValueError("student-frontier normalization buffers must be finite and positive")
    model.mean.copy_(mean.to(device=model.mean.device, dtype=model.mean.dtype))
    model.std.copy_(std.to(device=model.std.device, dtype=model.std.dtype))
