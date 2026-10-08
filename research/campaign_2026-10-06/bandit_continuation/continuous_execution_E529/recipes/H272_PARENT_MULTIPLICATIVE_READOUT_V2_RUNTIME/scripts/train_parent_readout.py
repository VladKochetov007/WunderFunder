"""Validate, profile, fit, and export the bounded H272 parent readout pair.

This V2 runtime is separate from the frozen H272/H233 packages. It trains the
same multiplicative rank-32 head and a parameter-matched hidden-only control on
one source-qualified parent recipe at a time. Expensive commands require a
root-issued, source-bound authorization receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[5]
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

import train_rnn as trainer  # noqa: E402
from wnn.conditional_readout import (  # noqa: E402
    CONTEXT_FEATURE_NAMES,
    CONTEXT_WIDTH,
    ConditionalReadoutHeads,
    direct_label_target,
    evaluation_masks,
    freeze_parent,
    iter_frozen_parent_chunks,
    primary_pair_wmse,
)

from wnn.data import N_FEATURES, SEQUENCE_LENGTH, WARMUP  # noqa: E402
from wnn.propensity import load_propensity  # noqa: E402
from wnn.splits import file_sha256, load_campaign_split  # noqa: E402

SOURCE_LOCK = PACKAGE_ROOT / "H272_PARENT_READOUT_V2_SOURCE_LOCK_V1_20261008.json"
RECIPE_PATHS = {
    "cleanfold": PACKAGE_ROOT / "recipes/H272_PARENT_READOUT_V2_CLEANFOLD_20261008.json",
    "full_data": PACKAGE_ROOT / "recipes/H272_PARENT_READOUT_V2_FULL_DATA_20261008.json",
}
MODES = ("hidden_only", "multiplicative")
SEED = 3
BATCH_SIZE = 8
CHUNK_ROWS = 2_000
EPOCHS = 2
CHECKPOINT_EPOCH = 1
RANK = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
GRADIENT_CLIP_NORM = 1.0
PROFILE_WARMUP_CHUNKS = 2
PROFILE_TIMED_CHUNKS = 8
EXPECTED_BATCHES = {"cleanfold": 1_492, "full_data": 1_513}


def _sha256_group_ids(groups: list[int] | tuple[int, ...]) -> str:
    return hashlib.sha256(np.asarray(groups, dtype="<i8").tobytes()).hexdigest()


def _validate_parent_output_count(actual: int, bound: int) -> None:
    if bound not in (23, 31) or actual not in (23, 31) or actual != bound:
        raise ValueError(f"parent output width must match its bound 23/31 contract, got {actual}")


def _qualified_fit_overlap(
    train_groups: tuple[int, ...],
    valid_groups: tuple[int, ...],
    evaluation_groups: tuple[int, ...],
    component_by_identity: dict[tuple[str, int], int],
) -> tuple[set[tuple[str, int]], set[int]]:
    fit_identity = {("train", group) for group in train_groups}
    fit_identity.update(("valid", group) for group in valid_groups)
    eval_identity = {("valid", group) for group in evaluation_groups}
    direct_overlap = fit_identity & eval_identity
    fit_components = {component_by_identity[item] for item in fit_identity}
    eval_components = {component_by_identity[item] for item in eval_identity}
    return direct_overlap, fit_components & eval_components


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _verify_source_lock() -> dict[str, Any]:
    if not SOURCE_LOCK.is_file():
        raise FileNotFoundError(f"reviewed source lock is required: {SOURCE_LOCK}")
    lock = json.loads(SOURCE_LOCK.read_text())
    if lock.get("schema") != "h272-parent-readout-source-lock-v1":
        raise ValueError("unsupported parent-readout source lock")
    for relative, wanted in lock["source_sha256"].items():
        path = PACKAGE_ROOT / relative
        actual = file_sha256(path)
        if actual != wanted:
            raise ValueError(f"source changed: {relative}: {actual} != {wanted}")
    for relative, wanted in lock["native_source_sha256"].items():
        actual = file_sha256(REPO_ROOT / relative)
        if actual != wanted:
            raise ValueError(f"native serving source changed: {relative}")
    for relative, wanted in lock["baseline_h272_source_sha256"].items():
        actual = file_sha256(REPO_ROOT / relative)
        if actual != wanted:
            raise ValueError(f"frozen H272 reference changed: {relative}")
    for key, path in RECIPE_PATHS.items():
        relative = str(path.relative_to(PACKAGE_ROOT))
        actual = file_sha256(path)
        if lock["recipe_sha256"].get(key) != actual:
            raise ValueError(f"recipe changed or unbound: {relative}")
    return lock


def _load_recipe(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    matches = [
        name for name, recipe_path in RECIPE_PATHS.items() if resolved == recipe_path.resolve()
    ]
    if len(matches) != 1:
        raise ValueError("only the two source-bound V2 parent recipes are accepted")
    recipe = json.loads(resolved.read_text())
    if recipe.get("schema") != "h272-parent-readout-recipe-v2":
        raise ValueError("unsupported parent-readout recipe schema")
    if recipe.get("recipe_id") != f"E529-H272-PARENT-READOUT-V2-{matches[0].upper()}-20261008":
        raise ValueError("recipe identity does not match its fixed path")
    fixed = recipe["training"]
    if SEQUENCE_LENGTH % CHUNK_ROWS:
        raise ValueError("sequence length is not divisible by the fixed chunk size")
    batches = EXPECTED_BATCHES[matches[0]]
    updates_per_epoch = batches * (SEQUENCE_LENGTH // CHUNK_ROWS)
    expected = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "chunk_rows": CHUNK_ROWS,
        "epochs": EPOCHS,
        "selected_checkpoint_epoch": CHECKPOINT_EPOCH,
        "rank": RANK,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "modes": list(MODES),
        "readout_teacher_coefficient": 0.0,
        "expected_batches_per_epoch": batches,
        "expected_updates_per_epoch_per_head": updates_per_epoch,
        "expected_total_updates_per_head": updates_per_epoch * EPOCHS,
        "optimizer": "AdamW",
        "schedule": "constant",
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "workers": 0,
        "drop_last": True,
        "head_initialization": (
            "Seed 3; instantiate the copied H272 four-control module in its original order; "
            "train only hidden_only and multiplicative parameters."
        ),
    }
    if fixed != expected:
        changed = sorted(
            key for key in set(fixed) | set(expected) if fixed.get(key) != expected.get(key)
        )
        raise ValueError(f"fixed optimizer, target, or matched-control contract changed: {changed}")
    _verify_source_lock()
    return recipe


def _verify_authorization(
    recipe_path: Path, command: str, authorization_path: Path | None
) -> dict[str, Any]:
    if authorization_path is None:
        raise PermissionError(f"{command} requires a root-issued authorization receipt")
    authorization = json.loads(authorization_path.read_text())
    lock = _verify_source_lock()
    expected = {
        "status": "approved",
        "command": command,
        "recipe_id": json.loads(recipe_path.read_text())["recipe_id"],
        "recipe_sha256": file_sha256(recipe_path),
        "source_lock_sha256": file_sha256(SOURCE_LOCK),
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise PermissionError(f"authorization field {key} is not bound to this source")
    if authorization.get("source_lock_source_sha256") != lock["source_sha256"]:
        raise PermissionError("authorization source hashes differ from the reviewed source lock")
    return authorization


def _verify_parent_contract(
    data_root: Path, recipe: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], tuple[int, ...], tuple[int, ...], Any]:
    parent = recipe["parent"]
    checkpoint_path = data_root / parent["checkpoint"]
    provenance_path = data_root / parent["provenance"]
    alias_path = data_root / parent["alias_checkpoint"]
    split_path = data_root / recipe["split_manifest"]["path"]
    for path, pin, label in (
        (checkpoint_path, parent["checkpoint_sha256"], "selected parent checkpoint"),
        (provenance_path, parent["provenance_sha256"], "parent provenance"),
        (alias_path, parent["alias_checkpoint_sha256"], "parent model.pt alias"),
        (split_path, recipe["split_manifest"]["sha256"], "fold-0 split manifest"),
    ):
        actual = file_sha256(path)
        if actual != pin:
            raise ValueError(f"{label} SHA-256 changed: {actual} != {pin}")

    provenance = json.loads(provenance_path.read_text())
    if provenance.get("selected_epoch") != parent["epoch"] or parent["epoch"] != 3:
        raise ValueError("parent is not the bound fixed epoch-3 checkpoint")
    if provenance.get("seed") != parent["seed"] or parent["seed"] != 4:
        raise ValueError("parent seed differs from the source-bound seed-4 recipe")
    if provenance.get("run_identity_source_sha256") != parent["source_identity_sha256"]:
        raise ValueError("parent run/source identity changed")
    selected_hash = provenance.get("checkpoint_sha256", {}).get("model_ep3.pt")
    if selected_hash != parent["checkpoint_sha256"]:
        raise ValueError("parent provenance does not select the bound model_ep3 checkpoint")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    alias = torch.load(alias_path, map_location="cpu", weights_only=False)
    if len(CONTEXT_FEATURE_NAMES) != 99 or tuple(CONTEXT_FEATURE_NAMES) != tuple(
        trainer.FEATURE_NAMES
    ):
        raise ValueError("parent 99-column event-feature order differs from train_rnn")
    if checkpoint.get("epoch") != 3 or alias.get("epoch") != 3:
        raise ValueError("model_ep3.pt/model.pt alias does not select fixed epoch 3")
    if checkpoint["model"].keys() != alias["model"].keys() or any(
        not torch.equal(checkpoint["model"][name], alias["model"][name])
        for name in checkpoint["model"]
    ):
        raise ValueError("model.pt alias tensors differ from the selected model_ep3 tensors")
    if checkpoint.get("n_in") != 211:
        raise ValueError("parent checkpoint input width must be exactly 211")
    n_out = int(checkpoint["model"]["head.2.weight"].shape[0])
    bound_n_out = int(parent["expected_n_out"])
    _validate_parent_output_count(n_out, bound_n_out)
    if tuple(checkpoint["model"]["head.0.weight"].shape) != (256, 512):
        raise ValueError("parent readout input must be the [recurrent, projection] 512-vector")
    if tuple(checkpoint["model"]["head.2.weight"].shape) != (bound_n_out, 256):
        raise ValueError("parent output tensor shape differs from the bound 23/31 contract")
    args = checkpoint["args"]
    for name, wanted in parent["args_fields"].items():
        if args.get(name) != wanted:
            raise ValueError(f"parent training argument {name} drifted: {args.get(name)!r}")
    if (
        recipe["mode"] == "cleanfold"
        and args.get("split_manifest_sha256") != recipe["split_manifest"]["sha256"]
    ):
        raise ValueError("cleanfold parent args do not bind the reviewed split-manifest bytes")
    if checkpoint.get("provenance", {}).get("fit_row_groups") != provenance.get("fit_row_groups"):
        raise ValueError("checkpoint-embedded and sidecar parent membership differ")
    for name in ("mean", "std"):
        value = checkpoint["model"][name].detach().cpu().contiguous().numpy()
        actual = hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest()
        expected = provenance["normalization_identity"][name]["bytes_sha256"]
        if actual != expected:
            raise ValueError(f"parent {name} tensor differs from its provenance identity")

    fit = provenance.get("fit_row_groups", {})
    train_groups = tuple(map(int, fit.get("train", ())))
    valid_groups = tuple(map(int, fit.get("valid", ())))
    identity = provenance.get("fit_membership_identity", {})
    expected_membership = parent["membership"]
    for split_name, groups in (("train", train_groups), ("valid", valid_groups)):
        actual = {
            "count": len(groups),
            "ordered_int64_le_sha256": _sha256_group_ids(groups),
        }
        if actual != identity.get(split_name) or actual != expected_membership[split_name]:
            raise ValueError(f"parent {split_name} fit membership differs from its exact recipe")
    if len(set(train_groups)) != len(train_groups) or len(set(valid_groups)) != len(valid_groups):
        raise ValueError("parent fit row groups are not unique within their source namespace")
    split = load_campaign_split(split_path, data_root)
    if split.sha256 != recipe["split_manifest"]["sha256"] or split.fold != 0:
        raise ValueError("fold-0 split membership changed")
    if recipe["mode"] == "cleanfold":
        if provenance.get("split_manifest_sha256") != split.sha256:
            raise ValueError("cleanfold parent provenance does not bind fold 0")
        if set(train_groups) != set(split.train_groups) or set(valid_groups) != set(
            split.valid_groups
        ):
            raise ValueError("cleanfold fit rows are not exactly the frozen fold-0 source sets")
        split.assert_fit_groups(train_groups, valid_groups)
        component_of = {(source, group): component for source, group, component in split.components}
        direct_overlap, component_overlap = _qualified_fit_overlap(
            train_groups, valid_groups, split.evaluation_groups, component_of
        )
        if direct_overlap:
            raise ValueError("cleanfold fit contains source-qualified evaluation rows")
        if component_overlap:
            raise ValueError("cleanfold parent/readout fit components overlap evaluation")
        expected_teacher_records = recipe["parent"]["teacher_prediction_ancestry"]
        if args.get("distill_provenance") != expected_teacher_records:
            raise ValueError("cleanfold parent teacher prediction ancestry changed")
        for teacher in recipe["parent"]["teacher_ancestry"]:
            checkpoint_path = data_root / "artifacts/runs" / teacher["run"] / "model_ep3.pt"
            provenance_path = data_root / "artifacts/runs" / teacher["run"] / "provenance.json"
            if file_sha256(checkpoint_path) != teacher["checkpoint_sha256"]:
                raise ValueError(f"parent teacher checkpoint changed: {teacher['run']}")
            if file_sha256(provenance_path) != teacher["provenance_sha256"]:
                raise ValueError(f"parent teacher provenance changed: {teacher['run']}")
    else:
        if (
            len(train_groups) != expected_membership["train_group_count"]
            or len(valid_groups) != expected_membership["valid_group_count"]
        ):
            raise ValueError("deployment parent fit-group counts changed")
        # The all-data model deliberately includes previously exposed fold-0 rows.
        # This recipe is deployment-only and has no fold-0 grade command.
        overlap = len(set(valid_groups) & set(split.evaluation_groups))
        if overlap != parent["fold0_exposed_valid_overlap_groups"]:
            raise ValueError("full-data exposed-row overlap differs from the reviewed audit")
    return checkpoint, provenance, train_groups, valid_groups, split


def _verify_propensity(data_root: Path, recipe: dict[str, Any], split: Any) -> dict[str, Path]:
    prop = recipe["propensity"]
    checkpoint = data_root / prop["checkpoint"]
    if file_sha256(checkpoint) != prop["checkpoint_sha256"]:
        raise ValueError("propensity checkpoint changed")

    provenance_path = prop.get("provenance")
    if provenance_path is not None:
        path = data_root / provenance_path
        if file_sha256(path) != prop.get("provenance_sha256"):
            raise ValueError("propensity provenance changed")
        provenance = json.loads(path.read_text())
        if (
            provenance.get("args", {}).get("name") != prop["run"]
            or provenance.get("selected_epoch") != 3
            or provenance.get("checkpoint_sha256", {}).get("model.pt") != prop["checkpoint_sha256"]
            or provenance.get("split_manifest_sha256") != recipe["split_manifest"]["sha256"]
            or provenance.get("args", {}).get("split_manifest_sha256")
            != recipe["split_manifest"]["sha256"]
        ):
            raise ValueError(
                "propensity producer provenance differs from its bound run/model/split"
            )
        fit_rows = provenance.get("fit_row_groups", {})
        if set(fit_rows) != {"valid"}:
            raise ValueError("fold-0 propensity fit-source namespaces changed")
        fit_valid = tuple(map(int, fit_rows["valid"]))
        eval_valid = tuple(map(int, provenance.get("evaluation_row_groups", ())))
        if fit_valid != tuple(split.valid_groups) or eval_valid != tuple(split.evaluation_groups):
            raise ValueError("propensity producer row membership differs from frozen fold 0")
        component_by_identity = {
            (source, group): component for source, group, component in split.components
        }
        fit_components = {component_by_identity[("valid", group)] for group in fit_valid}
        eval_components = {component_by_identity[("valid", group)] for group in eval_valid}
        if fit_components & eval_components:
            raise ValueError("fold-0 propensity fit components overlap its held-out evaluation")
    elif (
        not prop.get("provenance_gap")
        or prop.get("run_record", {}).get("runtime_source_bytes_attested") is not False
    ):
        raise ValueError(
            "unattested propensity ancestry must retain its explicit source uncertainty"
        )

    result: dict[str, Path] = {}
    for split_name in ("train", "valid"):
        record = prop["caches"][split_name]
        path = data_root / record["path"]
        sidecar = path.with_suffix(".json")
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"propensity {split_name} array changed")
        if file_sha256(sidecar) != record["sidecar_sha256"]:
            raise ValueError(f"propensity {split_name} sidecar changed")
        metadata = json.loads(sidecar.read_text())
        if (
            metadata.get("model") != prop["run"]
            or metadata.get("split") != split_name
            or metadata.get("model_sha256") != prop["checkpoint_sha256"]
        ):
            raise ValueError(f"propensity {split_name} cache is bound to another model or split")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != "float32":
            raise ValueError(f"propensity {split_name} array shape/dtype changed")
        if (
            metadata.get("rows") != record["shape"][0]
            or metadata.get("n_filled") != record["n_filled"]
        ):
            raise ValueError(f"propensity {split_name} cache fill count changed")
        result[split_name] = path
    return result


def validate_recipe(recipe_path: Path, data_root: Path) -> dict[str, Any]:
    """Read exact source/checkpoint/split/cache metadata without fitting or scoring."""
    recipe = _load_recipe(recipe_path)
    data_root = data_root.resolve()
    trainer.ROOT = data_root
    checkpoint, provenance, train_groups, valid_groups, split = _verify_parent_contract(
        data_root, recipe
    )
    prop_paths = _verify_propensity(data_root, recipe, split)
    return {
        "recipe_id": recipe["recipe_id"],
        "mode": recipe["mode"],
        "parent_checkpoint_sha256": recipe["parent"]["checkpoint_sha256"],
        "parent_provenance_sha256": recipe["parent"]["provenance_sha256"],
        "parent_source_identity_sha256": provenance["run_identity_source_sha256"],
        "parent_output_count": int(checkpoint["model"]["head.2.weight"].shape[0]),
        "fit_group_counts": {"train": len(train_groups), "valid": len(valid_groups)},
        "fold0_evaluation_groups": len(split.evaluation_groups),
        "cleanfold_prediction_export_path_available": recipe["mode"] == "cleanfold",
        "full_data_score_allowed": False,
        "propensity_checkpoint_sha256": recipe["propensity"]["checkpoint_sha256"],
        "propensity_provenance_sha256": recipe["propensity"].get("provenance_sha256"),
        "propensity_cache_paths": {key: str(value) for key, value in prop_paths.items()},
        "no_training_or_prediction_started": True,
    }


def _load_runtime(
    recipe_path: Path, data_root: Path, device: torch.device
) -> tuple[
    dict[str, Any], torch.nn.Module, ConditionalReadoutHeads, Any, DataLoader, torch.Generator, Any
]:
    recipe = _load_recipe(recipe_path)
    trainer.ROOT = data_root.resolve()
    checkpoint, provenance, train_groups, valid_groups, split = _verify_parent_contract(
        data_root.resolve(), recipe
    )
    prop_paths = _verify_propensity(data_root.resolve(), recipe, split)
    prop_model = load_propensity(data_root / recipe["propensity"]["checkpoint"], device)
    parent = trainer.make_model(
        checkpoint["args"], checkpoint["n_in"], checkpoint["model"]["head.2.weight"].shape[0]
    )
    parent.load_state_dict(checkpoint["model"], strict=True)
    parent.to(device)
    freeze_parent(parent)
    if (
        parent.head[0].in_features != 512
        or CONTEXT_WIDTH != 99
        or len(CONTEXT_FEATURE_NAMES) != 99
        or tuple(CONTEXT_FEATURE_NAMES) != tuple(trainer.FEATURE_NAMES)
    ):
        raise ValueError("parent readout must consume the frozen 512+99 feature contract")

    torch.manual_seed(SEED)
    heads = ConditionalReadoutHeads(parent.head[0].in_features, RANK).to(device)
    selected_parameters = [
        parameter for mode in MODES for parameter in heads.controls[mode].parameters()
    ]
    optimizer = torch.optim.AdamW(selected_parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_data = trainer.Sequences("train", train_groups, None, prop_paths["train"])
    valid_data = trainer.Sequences("valid", valid_groups, None, prop_paths["valid"])
    dataset = ConcatDataset((train_data, valid_data))
    loader_generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=loader_generator,
    )
    expected_batches = recipe["training"]["expected_batches_per_epoch"]
    if len(loader) != expected_batches:
        raise ValueError(f"loader batch count changed: {len(loader)} != {expected_batches}")
    weighting_args = SimpleNamespace(
        mask_weight=False,
        prop_power=2.0,
        prop_eps=0.03,
        valid_true_mask=0.03,
        prop_cache=True,
        class_weight="",
        since_weight=0.0,
        since_floor=0.2,
        episode_weight=0.0,
        episode_floor=0.2,
        sync_free=True,
    )
    context = {
        "parent_checkpoint": checkpoint,
        "parent_provenance": provenance,
        "train_groups": train_groups,
        "valid_groups": valid_groups,
        "split": split,
        "optimizer": optimizer,
        "dataset_length": len(dataset),
        "expected_updates_per_epoch": expected_batches * (SEQUENCE_LENGTH // CHUNK_ROWS),
        "weighting_args": weighting_args,
    }
    return recipe, parent, heads, prop_model, loader, loader_generator, context


def _batch_inputs(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    prop_model: torch.nn.Module,
    weighting_args: SimpleNamespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw, target, scored, prop_cached = (item.to(device, non_blocking=True) for item in batch)
    if target.shape[-1] != 2:
        raise ValueError("readout batches must contain only the original two label channels")
    inputs = trainer.build_inputs(raw)
    context = inputs[..., N_FEATURES : N_FEATURES + CONTEXT_WIDTH]
    label = direct_label_target(target)
    valid = torch.arange(raw.shape[1], device=device)[None].expand(raw.shape[0], -1) >= WARMUP
    row_weights = trainer.main_row_weights(
        inputs, label, valid, scored, prop_model, prop_cached, weighting_args, raw
    )
    return inputs, context, label, label, row_weights


def _update_chunk(
    parent: torch.nn.Module,
    heads: ConditionalReadoutHeads,
    optimizer: torch.optim.Optimizer,
    parent_chunk: tuple[slice, torch.Tensor, torch.Tensor, torch.Tensor],
    target: torch.Tensor,
    label: torch.Tensor,
    row_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sl, parent_output, hidden, event_context = parent_chunk
    if hidden.shape[-1] != 512 or parent.head[0].in_features != 512:
        raise RuntimeError("parent representation must be the 512-wide H272 input")
    outputs = heads.forward_all(parent_output, hidden, event_context)
    if any(parameter.requires_grad for parameter in parent.parameters()):
        raise RuntimeError("frozen parent parameter became trainable")
    optimizer.zero_grad(set_to_none=True)
    losses: dict[str, torch.Tensor] = {}
    for mode in MODES:
        loss = primary_pair_wmse(outputs[mode], target[:, sl], row_weights[:, sl], label[:, sl])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{mode} direct-label loss became nonfinite")
        loss.backward()
        clip_grad_norm_(
            heads.controls[mode].parameters(),
            max_norm=GRADIENT_CLIP_NORM,
            error_if_nonfinite=True,
        )
        losses[mode] = loss.detach()
    optimizer.step()
    return losses


def _state_hashes(heads: ConditionalReadoutHeads) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for mode in MODES:
        digest = hashlib.sha256()
        for name, value in sorted(heads.controls[mode].state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        hashes[mode] = digest.hexdigest()
    return hashes


def _save_checkpoint(
    output_dir: Path,
    recipe_path: Path,
    recipe: dict[str, Any],
    parent: torch.nn.Module,
    heads: ConditionalReadoutHeads,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    epoch: int,
    provenance: dict[str, Any],
) -> Path:
    parent_state = {key: value.detach().cpu().clone() for key, value in parent.state_dict().items()}
    payload = {
        "schema_version": 1,
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": file_sha256(recipe_path),
        "source_lock_sha256": file_sha256(SOURCE_LOCK),
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "parent_model": parent_state,
        "parent_checkpoint_sha256": recipe["parent"]["checkpoint_sha256"],
        "heads": {key: value.detach().cpu().clone() for key, value in heads.state_dict().items()},
        "selected_modes": list(MODES),
        "optimizer": optimizer.state_dict(),
        "loader_generator_state": generator.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "training_config": recipe["training"],
        "provenance": provenance,
    }
    path = output_dir / f"model_ep{epoch}.pt"
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def _build_provenance(
    recipe_path: Path,
    recipe: dict[str, Any],
    context: dict[str, Any],
    source_lock: dict[str, Any],
) -> dict[str, Any]:
    return {
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": file_sha256(recipe_path),
        "source_lock_sha256": file_sha256(SOURCE_LOCK),
        "source_sha256": source_lock["source_sha256"],
        "parent_run": recipe["parent"]["run"],
        "parent_checkpoint_sha256": recipe["parent"]["checkpoint_sha256"],
        "parent_provenance_sha256": recipe["parent"]["provenance_sha256"],
        "parent_source_identity_sha256": recipe["parent"]["source_identity_sha256"],
        "parent_output_count": recipe["parent"]["expected_n_out"],
        "split_manifest_sha256": recipe["split_manifest"]["sha256"],
        "propensity_run": recipe["propensity"]["run"],
        "propensity_checkpoint_sha256": recipe["propensity"]["checkpoint_sha256"],
        "propensity_cache_sha256": {
            split: value["sha256"] for split, value in recipe["propensity"]["caches"].items()
        },
        "fit_row_groups": {
            "train": list(context["train_groups"]),
            "valid": list(context["valid_groups"]),
        },
        "fit_membership_sha256": {
            "train": _sha256_group_ids(context["train_groups"]),
            "valid": _sha256_group_ids(context["valid_groups"]),
        },
        "fit_group_namespace": "source-qualified (train,row_group) and (valid,row_group)",
        "mode": recipe["mode"],
        "evaluation_allowed": recipe["mode"] == "cleanfold",
        "exposed_fold0_overlap_groups": recipe["parent"].get(
            "fold0_exposed_valid_overlap_groups", 0
        ),
        "objective": {
            "target": "clip(original primary pair label, -2, 2)",
            "loss": "H272 primary_pair_wmse per 2000-row chunk",
            "loss_weight": "abs(clipped original label) * train_rnn.main_row_weights",
            "propensity": "p^2 + 0.03; true-mask coefficient 0.03; mask_weight=false",
            "teacher_inputs": "none",
            "parent_trainable": False,
        },
        "readout_modes": list(MODES),
        "random_seed": SEED,
        "expected_updates_per_epoch_per_head": context["expected_updates_per_epoch"],
        "optimizer_completed_updates_per_head": [0, 0],
        "epoch_losses": [],
        "elapsed_epoch_seconds": [],
        "epoch_update_evidence": [],
        "status": "fitting",
    }


def _validate_profile_plan(warmup_chunks: int, timed_chunks: int, cpu_threads: int) -> None:
    if (
        timed_chunks != PROFILE_TIMED_CHUNKS
        or warmup_chunks != PROFILE_WARMUP_CHUNKS
        or cpu_threads != 1
    ):
        raise ValueError("profile must use two warmup and eight timed chunks with one CPU thread")


def run_profile(args: argparse.Namespace) -> None:
    recipe_path = args.recipe.resolve()
    recipe = _load_recipe(recipe_path)
    _verify_authorization(recipe_path, "profile", args.authorization)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("profile authorization requires the admitted CUDA device")
    _validate_profile_plan(args.warmup_chunks, args.steps, args.cpu_threads)
    output_path = args.output.resolve()
    expected_output = args.data_root.resolve() / recipe["paths"]["profile_output"]
    if output_path != expected_output:
        raise ValueError(f"profile output path is source-bound to {expected_output}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite profile output: {output_path}")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats(device)
    recipe, parent, heads, prop_model, loader, _, context = _load_runtime(
        recipe_path, args.data_root.resolve(), device
    )
    batch = next(iter(loader))
    inputs, event_context, label, target, row_weights = _batch_inputs(
        batch, device, prop_model, context["weighting_args"]
    )
    optimizer = context["optimizer"]
    chunks = iter_frozen_parent_chunks(parent, inputs, event_context, CHUNK_ROWS, amp_enabled=True)
    started = time.perf_counter()
    timed = 0
    for index, parent_chunk in enumerate(chunks):
        losses = _update_chunk(parent, heads, optimizer, parent_chunk, target, label, row_weights)
        if index >= args.warmup_chunks:
            timed += 1
        if timed >= args.steps:
            break
    if timed != args.steps:
        raise RuntimeError(f"profile expected {args.steps} timed chunks, observed {timed}")
    elapsed = time.perf_counter() - started
    result = {
        "status": "profile_complete",
        "recipe_id": recipe["recipe_id"],
        "authorization_sha256": file_sha256(args.authorization),
        "device": torch.cuda.get_device_name(device),
        "steps_per_mode": timed,
        "elapsed_seconds": elapsed,
        "last_losses": {key: float(value) for key, value in losses.items()},
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "fit_started": False,
    }
    _json_write(args.output, result)
    print(json.dumps(result, sort_keys=True))


def run_fit(args: argparse.Namespace) -> None:
    recipe_path = args.recipe.resolve()
    recipe = _load_recipe(recipe_path)
    _verify_authorization(recipe_path, "fit", args.authorization)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fit authorization requires the admitted CUDA device")
    output_dir = args.output_dir.resolve()
    expected_output = args.data_root.resolve() / recipe["paths"]["fit_output"]
    if output_dir != expected_output:
        raise ValueError(f"fit output is source-bound to {expected_output}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite fit output: {output_dir}")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    source_lock = _verify_source_lock()
    recipe, parent, heads, prop_model, loader, generator, context = _load_runtime(
        recipe_path, args.data_root.resolve(), device
    )
    output_dir.mkdir(parents=True)
    provenance = _build_provenance(recipe_path, recipe, context, source_lock)
    provenance["fit_authorization_sha256"] = file_sha256(args.authorization)
    provenance["initial_selected_heads_sha256"] = _state_hashes(heads)
    optimizer = context["optimizer"]
    for epoch in range(EPOCHS):
        started = time.perf_counter()
        heads.train()
        loss_sums = {mode: 0.0 for mode in MODES}
        update_count = 0
        for batch in loader:
            inputs, event_context, label, target, row_weights = _batch_inputs(
                batch, device, prop_model, context["weighting_args"]
            )
            for parent_chunk in iter_frozen_parent_chunks(
                parent,
                inputs,
                event_context,
                CHUNK_ROWS,
                amp_enabled=device.type == "cuda",
            ):
                losses = _update_chunk(
                    parent, heads, optimizer, parent_chunk, target, label, row_weights
                )
                for mode in MODES:
                    loss_sums[mode] += float(losses[mode])
                update_count += 1
        expected_updates = context["expected_updates_per_epoch"]
        if update_count != expected_updates:
            raise RuntimeError(f"expected {expected_updates} chunks, observed {update_count}")
        elapsed = time.perf_counter() - started
        provenance["elapsed_epoch_seconds"].append(elapsed)
        provenance["epoch_losses"].append({mode: loss_sums[mode] / update_count for mode in MODES})
        provenance["optimizer_completed_updates_per_head"] = [
            update_count * (epoch + 1),
            update_count * (epoch + 1),
        ]
        provenance["epoch_update_evidence"].append(
            {
                "optimizer_attempts_per_head": update_count,
                "completed_updates_per_head": update_count,
                "amp_skipped_updates": 0,
                "finite_losses": True,
                "finite_gradients": True,
            }
        )
        provenance["selected_epoch"] = CHECKPOINT_EPOCH if epoch == CHECKPOINT_EPOCH else None
        provenance["status"] = "complete" if epoch == CHECKPOINT_EPOCH else "epoch_complete"
        checkpoint_path = _save_checkpoint(
            output_dir,
            recipe_path,
            recipe,
            parent,
            heads,
            optimizer,
            generator,
            epoch,
            provenance,
        )
        provenance.setdefault("checkpoint_sha256", {})[checkpoint_path.name] = file_sha256(
            checkpoint_path
        )
        if epoch == CHECKPOINT_EPOCH:
            alias = output_dir / "model.pt"
            temporary = alias.with_name(alias.name + ".partial")
            torch.save(
                torch.load(checkpoint_path, map_location="cpu", weights_only=False), temporary
            )
            os.replace(temporary, alias)
            provenance["checkpoint_sha256"][alias.name] = file_sha256(alias)
        _json_write(output_dir / "provenance.json", provenance)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "updates_per_head": update_count,
                    "seconds": elapsed,
                    "losses": provenance["epoch_losses"][-1],
                    "checkpoint": str(checkpoint_path),
                    "status": provenance["status"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _validate_fit_completion(fit_provenance: dict[str, Any], recipe: dict[str, Any]) -> None:
    expected_updates = recipe["training"]["expected_total_updates_per_head"]
    updates_per_epoch = recipe["training"]["expected_updates_per_epoch_per_head"]
    evidence = fit_provenance.get("epoch_update_evidence", [])
    if (
        fit_provenance.get("status") != "complete"
        or fit_provenance.get("selected_epoch") != CHECKPOINT_EPOCH
        or fit_provenance.get("optimizer_completed_updates_per_head")
        != [expected_updates, expected_updates]
        or len(evidence) != EPOCHS
        or any(
            item.get("optimizer_attempts_per_head") != updates_per_epoch
            or item.get("completed_updates_per_head") != updates_per_epoch
            or item.get("amp_skipped_updates") != 0
            or item.get("finite_losses") is not True
            or item.get("finite_gradients") is not True
            for item in evidence
        )
    ):
        raise ValueError("selected readout checkpoint lacks complete fixed-update evidence")


def _load_fixed_heads(
    recipe_path: Path,
    data_root: Path,
    device: torch.device,
    parent: torch.nn.Module,
    heads: ConditionalReadoutHeads,
    expected_checkpoint_sha256: str,
) -> Path:
    recipe = _load_recipe(recipe_path)
    checkpoint_path = data_root / recipe["paths"]["fit_output"] / "model_ep1.pt"
    if file_sha256(checkpoint_path) != expected_checkpoint_sha256:
        raise ValueError("fixed epoch-1 readout checkpoint is not the admitted artifact")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        payload.get("recipe_id") != recipe["recipe_id"]
        or payload.get("recipe_sha256") != file_sha256(recipe_path)
        or payload.get("source_lock_sha256") != file_sha256(SOURCE_LOCK)
        or payload.get("epoch") != CHECKPOINT_EPOCH
        or payload.get("selected_modes") != list(MODES)
        or payload.get("parent_checkpoint_sha256") != recipe["parent"]["checkpoint_sha256"]
    ):
        raise ValueError("selected readout checkpoint is not the fixed source-bound epoch 1")
    _validate_fit_completion(payload.get("provenance", {}), recipe)
    parent_state = parent.state_dict()
    if parent_state.keys() != payload["parent_model"].keys() or any(
        not torch.equal(parent_state[key].detach().cpu(), payload["parent_model"][key])
        for key in parent_state
    ):
        raise ValueError("readout checkpoint altered the frozen parent tensors")
    parent.load_state_dict(payload["parent_model"], strict=True)
    freeze_parent(parent)
    heads.load_state_dict(payload["heads"], strict=True)
    heads.to(device).eval()
    return checkpoint_path


def run_export(args: argparse.Namespace) -> None:
    recipe_path = args.recipe.resolve()
    recipe = _load_recipe(recipe_path)
    authorization = _verify_authorization(recipe_path, "export", args.authorization)
    if recipe["mode"] != "cleanfold":
        raise PermissionError("full-data deployment head has no prediction-export or grade path")
    device = torch.device(args.device)
    parent_record = recipe["parent"]
    checkpoint_path = args.data_root / parent_record["checkpoint"]
    if file_sha256(checkpoint_path) != parent_record["checkpoint_sha256"]:
        raise ValueError("parent checkpoint bytes differ from the frozen prediction recipe")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parent = trainer.make_model(
        checkpoint["args"], checkpoint["n_in"], checkpoint["model"]["head.2.weight"].shape[0]
    ).to(device)
    parent.load_state_dict(checkpoint["model"], strict=True)
    freeze_parent(parent)
    heads = ConditionalReadoutHeads(512, RANK).to(device)
    if authorization.get("selected_epoch") != CHECKPOINT_EPOCH:
        raise PermissionError("export authorization must select fixed epoch 1")
    selected_checkpoint_sha256 = authorization.get("selected_checkpoint_sha256")
    if not isinstance(selected_checkpoint_sha256, str):
        raise PermissionError("export authorization must pin the fixed readout checkpoint")
    readout_checkpoint = _load_fixed_heads(
        recipe_path,
        args.data_root,
        device,
        parent,
        heads,
        selected_checkpoint_sha256,
    )
    split = load_campaign_split(
        args.data_root / recipe["split_manifest"]["path"], args.data_root.resolve()
    )
    row_groups = tuple(range(1_873))
    if set(split.valid_groups) & set(split.evaluation_groups) or set(split.valid_groups) | set(
        split.evaluation_groups
    ) != set(row_groups):
        raise ValueError("cleanfold export order is not the complete frozen 1,873-row valid set")
    dataset = trainer.Sequences("valid", row_groups, None, None)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    output_dir = args.output_dir.resolve()
    expected_output = args.data_root.resolve() / recipe["paths"]["prediction_export"]
    if output_dir != expected_output:
        raise ValueError(f"prediction export path is source-bound to {expected_output}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite export directory {output_dir}")
    output_dir.mkdir(parents=True)
    shape = (len(row_groups), SEQUENCE_LENGTH, 2)
    arrays = {
        name: np.lib.format.open_memmap(
            output_dir / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape
        )
        for name in ("parent", "hidden_only", "multiplicative")
    }
    scored_mask = np.lib.format.open_memmap(
        output_dir / "scored_mask.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(len(row_groups), SEQUENCE_LENGTH),
    )
    write_at = 0
    for raw, target, scored in loader:
        batch_size = raw.shape[0]
        raw = raw.to(device)
        inputs = trainer.build_inputs(raw)
        context = inputs[..., N_FEATURES : N_FEATURES + CONTEXT_WIDTH]
        target = target.to(device)
        for row, parent_output, hidden, event_context in iter_frozen_parent_chunks(
            parent, inputs, context, CHUNK_ROWS, amp_enabled=device.type == "cuda"
        ):
            outputs = heads.forward_all(parent_output, hidden, event_context)
            end = write_at + batch_size
            for name in arrays:
                value = outputs[name][..., :2] if name != "parent" else outputs["anchor"]
                arrays[name][write_at:end, row, :] = value.detach().cpu().numpy()
        scored_mask[write_at : write_at + batch_size] = scored.numpy()
        write_at += batch_size
    if write_at != len(row_groups):
        raise RuntimeError(f"export wrote {write_at} groups, expected {len(row_groups)}")
    for array in arrays.values():
        array.flush()
    scored_mask.flush()
    group_mask, selected_mask = evaluation_masks(
        row_groups, split.evaluation_groups, np.asarray(scored_mask)
    )
    np.save(
        output_dir / "row_groups.npy", np.asarray(row_groups, dtype=np.int64), allow_pickle=False
    )
    np.save(output_dir / "evaluation_group_mask.npy", group_mask, allow_pickle=False)
    np.save(output_dir / "evaluation_selected_mask.npy", selected_mask, allow_pickle=False)
    receipt = {
        "status": "cleanfold_predictions_exported_ungraded",
        "recipe_id": recipe["recipe_id"],
        "authorization_sha256": file_sha256(args.authorization),
        "mode": recipe["mode"],
        "parent_checkpoint_sha256": recipe["parent"]["checkpoint_sha256"],
        "readout_checkpoint_sha256": file_sha256(readout_checkpoint),
        "row_groups": len(row_groups),
        "prediction_shape": list(shape),
        "prediction_dtype": "float32",
        "prediction_sha256": {name: file_sha256(output_dir / f"{name}.npy") for name in arrays},
        "evaluation_groups": int(group_mask.sum()),
        "evaluation_group_ids_sha256": _sha256_group_ids(split.evaluation_groups),
        "official_metric": "global_wp_clip2_v1",
        "grade_started": False,
    }
    _json_write(output_dir / "export_receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


def run_export_head(args: argparse.Namespace) -> None:
    """Write only the fixed multiplicative FP32 head for a later native bundle."""
    recipe_path = args.recipe.resolve()
    recipe = _load_recipe(recipe_path)
    authorization = _verify_authorization(recipe_path, "export_head", args.authorization)
    if authorization.get("selected_epoch") != CHECKPOINT_EPOCH:
        raise PermissionError("serving-head export authorization must select fixed epoch 1")
    selected_checkpoint_sha256 = authorization.get("selected_checkpoint_sha256")
    if not isinstance(selected_checkpoint_sha256, str):
        raise PermissionError("serving-head export authorization must pin the readout checkpoint")
    data_root = args.data_root.resolve()
    parent_checkpoint_path = data_root / recipe["parent"]["checkpoint"]
    if file_sha256(parent_checkpoint_path) != recipe["parent"]["checkpoint_sha256"]:
        raise ValueError("parent checkpoint bytes differ from the frozen serving recipe")
    parent_checkpoint = torch.load(parent_checkpoint_path, map_location="cpu", weights_only=False)
    parent = trainer.make_model(
        parent_checkpoint["args"],
        parent_checkpoint["n_in"],
        parent_checkpoint["model"]["head.2.weight"].shape[0],
    )
    parent.load_state_dict(parent_checkpoint["model"], strict=True)
    freeze_parent(parent)
    heads = ConditionalReadoutHeads(512, RANK)
    selected_path = _load_fixed_heads(
        recipe_path,
        data_root,
        torch.device("cpu"),
        parent,
        heads,
        selected_checkpoint_sha256,
    )
    output_dir = args.output_dir.resolve()
    expected_output = data_root / recipe["paths"]["serving_head_export"]
    if output_dir != expected_output:
        raise ValueError(f"serving-head output path is source-bound to {expected_output}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite serving-head export {output_dir}")
    output_dir.mkdir(parents=True)
    state = heads.controls["multiplicative"].state_dict()
    arrays = {
        name: value.detach().cpu().numpy().astype(np.float32, copy=False)
        for name, value in state.items()
    }
    weights_path = output_dir / "multiplicative_head.npz"
    np.savez(weights_path, **arrays)
    sidecar = {
        "status": "head_weights_exported_no_submission_zip",
        "recipe_id": recipe["recipe_id"],
        "authorization_sha256": file_sha256(args.authorization),
        "selected_epoch": CHECKPOINT_EPOCH,
        "recipe_sha256": file_sha256(recipe_path),
        "source_lock_sha256": file_sha256(SOURCE_LOCK),
        "selected_checkpoint_sha256": file_sha256(selected_path),
        "parent_checkpoint_sha256": recipe["parent"]["checkpoint_sha256"],
        "parent_source_identity_sha256": recipe["parent"]["source_identity_sha256"],
        "mode": "multiplicative",
        "weight_file": weights_path.name,
        "weight_file_sha256": file_sha256(weights_path),
        "weight_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "precision": "float32",
        "native_binding": "wnn_native.ConditionalPredictor",
        "public_zip_created": False,
    }
    _json_write(output_dir / "head_export_receipt.json", sidecar)
    print(json.dumps(sidecar, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate source and data identities only")
    validate.add_argument("--recipe", type=Path, required=True)
    validate.add_argument("--data-root", type=Path, required=True)
    validate.set_defaults(
        handler=lambda args: print(
            json.dumps(validate_recipe(args.recipe, args.data_root), sort_keys=True)
        )
    )
    for name, handler in (
        ("profile", run_profile),
        ("fit", run_fit),
        ("export", run_export),
        ("export-head", run_export_head),
    ):
        sub = commands.add_parser(name)
        sub.add_argument("--recipe", type=Path, required=True)
        sub.add_argument("--data-root", type=Path, required=True)
        sub.add_argument("--authorization", type=Path, required=True)
        sub.add_argument("--device", default="cuda")
        if name == "profile":
            sub.add_argument("--warmup-chunks", type=int, default=2)
            sub.add_argument("--steps", type=int, default=8)
            sub.add_argument("--cpu-threads", type=int, default=1)
            sub.add_argument("--output", type=Path, required=True)
        elif name == "fit":
            sub.add_argument("--cpu-threads", type=int, default=1)
            sub.add_argument("--output-dir", type=Path, required=True)
        else:
            sub.add_argument("--output-dir", type=Path, required=True)
        sub.set_defaults(handler=handler)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
