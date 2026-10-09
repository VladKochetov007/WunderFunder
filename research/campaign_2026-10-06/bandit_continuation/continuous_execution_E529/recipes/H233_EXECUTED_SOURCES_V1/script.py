"""Profile, fit and export H233 low-rank readouts over a frozen C1 GRU."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any  # JSON and torch checkpoints have heterogeneous nested schemas.

import numpy as np
import torch
import train_rnn as trainer
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader

from wnn.conditional_readout import (
    CONTEXT_FEATURE_NAMES,
    CONTEXT_WIDTH,
    READOUT_MODES,
    ConditionalReadoutHeads,
    freeze_parent,
    iter_frozen_parent_chunks,
    primary_pair_wmse,
)
from wnn.data import N_FEATURES, SEQUENCE_LENGTH, WARMUP
from wnn.propensity import load_propensity
from wnn.splits import CampaignSplit, file_sha256, load_campaign_split

ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = (
    ROOT
    / "research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/recipes/"
    / "H233_C1S3_CONDITIONAL_READOUT_V1.json"
)
PARENT_RUN = "bd26_c1v2_s3_f0"
PARENT_EPOCH = 1
PARENT_CHECKPOINT = ROOT / "artifacts/runs" / PARENT_RUN / "model_ep1.pt"
PARENT_PROVENANCE = ROOT / "artifacts/runs" / PARENT_RUN / "provenance.json"
SPLIT_MANIFEST = ROOT / "research/campaign_2026-09-29/artifacts/splits/fold0.json"
PROPENSITY_RUN = "c20260929_prop_f0"
TEACHERS = (("c20261003_combo_s3_f0", 0.8125), ("c20261003_combo_s4_f0", 0.8125))
SEED = 3
BATCH_SIZE = 8
CHUNK_ROWS = 2_000
EPOCHS = 2
CHECKPOINT_EPOCH = 1
RANK = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
DISTILL_ALPHA = 0.7
CHECKPOINT_SHA256 = "7d34ee5649e124948f0b651aa786083fa9ee66a01c439b034345a1d9df9ec3f0"
PARENT_PROVENANCE_SHA256 = "60e8884d3d5886dc53dcb2b7d280fef284a702de8769ab333e07e1819d66f28a"
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PARENT_MODEL_PT_SHA256 = "5f21361d47cc1f010abb363e6aca6e3cbf50a3d5d40d88ad81851aed7519fc4d"
PARENT_VALID_PREDICTION_SHA256 = "cb832b1397c75e318551fd6d6f0e22d09f3c4230c3c5ad56cab1a4c4e603c7e8"
PROPENSITY_CHECKPOINT_SHA256 = "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7"
TEACHER_CHECKPOINT_SHA256 = {
    "c20261003_combo_s3_f0": "d4a1e50b1892b4d9edd7c255908b68073a760c059ca80223e98a63c7e023371e",
    "c20261003_combo_s4_f0": "0203b109a1f25f3c82794ad3684b83fe4a4bc815f4b5c1bc4bc1ea450dba0a8d",
}
TEACHER_PREDICTION_SHA256 = {
    "c20261003_combo_s3_f0": {
        "train": "55be3dffe899260f875e7d6460d37a24257067ebedb9273514cb38accc6af086",
        "valid": "aa38f384907e1ad1122e410c2f5a6fb9353da387bbab788b3e9978b05f646d19",
    },
    "c20261003_combo_s4_f0": {
        "train": "6188bba89a0d5e30516c8ec4163746310aa0e4b634dd1bbc3da9a156b8d9bfb2",
        "valid": "f1a3b07072a62dce620dc0c90e9568093367d0aed1afa4e9ec550866f94df12b",
    },
}


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _torch_save(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_recipe(path: Path) -> dict[str, Any]:
    if path.resolve() != RECIPE_PATH.resolve():
        raise ValueError(f"H233 uses its single frozen recipe only: {RECIPE_PATH}")
    recipe = json.loads(path.read_text())
    if recipe.get("recipe_id") != "E529-H233-C1S3-CONDITIONAL-READOUT-V1":
        raise ValueError(f"unexpected H233 recipe identity in {path}")
    for relative, expected in recipe["source_sha256"].items():
        source = ROOT / relative
        actual = file_sha256(source)
        if actual != expected:
            raise ValueError(f"frozen source changed: {relative} ({actual} != {expected})")
    return recipe


def _verify_static_ancestry(
    recipe: dict[str, Any],
) -> tuple[CampaignSplit, dict[str, Any], tuple[int, ...], tuple[int, ...]]:
    pins = recipe["ancestry"]
    checks = (
        (PARENT_CHECKPOINT, CHECKPOINT_SHA256, "C1 seed-3 epoch-1 checkpoint"),
        (PARENT_PROVENANCE, PARENT_PROVENANCE_SHA256, "C1 seed-3 provenance"),
        (SPLIT_MANIFEST, SPLIT_SHA256, "fold-0 split manifest"),
    )
    for path, expected, label in checks:
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"{label} hash changed: {actual} != {expected}")
    if pins["parent_checkpoint_sha256"] != CHECKPOINT_SHA256:
        raise ValueError("recipe checkpoint pin disagrees with the H233 implementation")
    if pins["parent_provenance_sha256"] != PARENT_PROVENANCE_SHA256:
        raise ValueError("recipe parent-provenance pin disagrees with the H233 implementation")
    if pins["split_manifest_sha256"] != SPLIT_SHA256:
        raise ValueError("recipe split pin disagrees with the H233 implementation")
    if pins["propensity_checkpoint_sha256"] != PROPENSITY_CHECKPOINT_SHA256:
        raise ValueError("recipe propensity pin disagrees with the H233 implementation")
    if pins["parent_model_alias_sha256"] != PARENT_MODEL_PT_SHA256:
        raise ValueError("recipe C1 model.pt alias pin disagrees with the H233 implementation")
    if pins["parent_valid_prediction_sha256"] != PARENT_VALID_PREDICTION_SHA256:
        raise ValueError("recipe E511 anchor prediction pin disagrees with the H233 implementation")
    if pins["teacher_checkpoint_sha256"] != TEACHER_CHECKPOINT_SHA256:
        raise ValueError("recipe teacher checkpoint pins disagree with the H233 implementation")
    if pins["teacher_prediction_sha256"] != TEACHER_PREDICTION_SHA256:
        raise ValueError("recipe teacher prediction pins disagree with the H233 implementation")
    for relative, expected in recipe["reference_sha256"].items():
        actual = file_sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(f"frozen recipe reference changed: {relative}")
    if tuple(recipe["feature_order"]) != tuple(CONTEXT_FEATURE_NAMES):
        raise ValueError(
            "frozen recipe's 99-column event context order differs from the implementation"
        )
    split = load_campaign_split(SPLIT_MANIFEST, ROOT)
    if split.sha256 != SPLIT_SHA256 or split.fold != 0:
        raise ValueError("loaded campaign split is not the frozen fold-0 manifest")
    parent_provenance = json.loads(PARENT_PROVENANCE.read_text())
    if parent_provenance.get("split_manifest_sha256") != split.sha256:
        raise ValueError("parent fit did not use the frozen fold-0 manifest")
    fit_groups = parent_provenance["fit_row_groups"]
    train_groups = tuple(map(int, fit_groups["train"]))
    valid_groups = tuple(map(int, fit_groups["valid"]))
    if set(train_groups) != set(split.train_groups) or set(valid_groups) != set(split.valid_groups):
        raise ValueError("C1 fit membership does not exactly equal fold-0 fit membership")
    split.assert_fit_groups(train_groups, valid_groups)
    if set(split.evaluation_groups) & set(valid_groups):
        raise ValueError("evaluation groups entered the parent fit set")
    recipe_counts = recipe["parent_fit_membership"]
    if (len(train_groups), len(valid_groups), len(split.evaluation_groups)) != (
        recipe_counts["train_groups"],
        recipe_counts["valid_fit_groups"],
        recipe_counts["evaluation_groups"],
    ):
        raise ValueError("parent group counts disagree with the frozen H233 recipe")
    return split, parent_provenance, train_groups, valid_groups


def _verify_parent_definition(checkpoint: dict[str, Any]) -> None:
    args = checkpoint["args"]
    fixed = {
        "arch": "gru",
        "cell": "gru",
        "enc": "linear",
        "hidden": 256,
        "layers": 2,
        "raw_only": False,
        "heuristics": False,
        "online_norm": False,
        "research_features": "none",
        "propensity": PROPENSITY_RUN,
        "loss": "wmse",
        "loss_norm": "chunk",
        "distill_alpha": DISTILL_ALPHA,
        "distill": "c20261003_combo_s3_f0*0.8125,c20261003_combo_s4_f0*0.8125",
        "init": "c20261003_combo_s3_f0@ep3",
        "seed": SEED,
        "batch": 16,
        "chunk": CHUNK_ROWS,
        "precision": "fp16",
        "optim": "adamw",
        "sched": "onecycle",
        "prop_power": 2.0,
        "prop_eps": 0.03,
        "valid_true_mask": 0.03,
        "aux_struct": 0.3,
        "aux_next": 0.3,
        "train_source": "train+valid",
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": CHECKPOINT_EPOCH,
    }
    for name, wanted in fixed.items():
        if args.get(name) != wanted:
            raise ValueError(f"C1 parent field {name} changed: {args.get(name)!r} != {wanted!r}")
    if checkpoint.get("epoch") != PARENT_EPOCH:
        raise ValueError(f"expected C1 fixed epoch {PARENT_EPOCH}, found {checkpoint.get('epoch')}")
    if int(checkpoint["n_in"]) != N_FEATURES + CONTEXT_WIDTH:
        raise ValueError("C1 input width does not match raw112 plus ordered event99 features")
    if len(CONTEXT_FEATURE_NAMES) != 99 or tuple(CONTEXT_FEATURE_NAMES) != tuple(
        trainer.FEATURE_NAMES
    ):
        raise ValueError("H233 context feature order differs from train_rnn's parent feature order")
    if checkpoint["model"]["head.2.weight"].shape[0] != 23:
        raise ValueError("C1 parent auxiliary output head is not the expected 23-channel head")


def _load_parent(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if file_sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("parent epoch-1 checkpoint does not match the frozen C1 seed-3 artifact")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _verify_parent_definition(checkpoint)
    model = trainer.make_model(
        checkpoint["args"], checkpoint["n_in"], checkpoint["model"]["head.2.weight"].shape[0]
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    freeze_parent(model)
    return model, checkpoint


def _verify_parent_alias() -> None:
    alias_path = ROOT / "artifacts/runs" / PARENT_RUN / "model.pt"
    if file_sha256(alias_path) != PARENT_MODEL_PT_SHA256:
        raise ValueError("C1 model.pt alias changed from its E511-verified bytes")
    alias = torch.load(alias_path, map_location="cpu", weights_only=False)
    selected = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    if alias.get("epoch") != PARENT_EPOCH:
        raise ValueError("C1 model.pt no longer selects the frozen epoch-1 checkpoint")
    if alias["model"].keys() != selected["model"].keys() or any(
        not torch.equal(alias["model"][key], selected["model"][key]) for key in alias["model"]
    ):
        raise ValueError("C1 model.pt tensors differ from the pinned model_ep1.pt tensors")


def _load_fit_data(
    recipe: dict[str, Any], device: torch.device, batch_size: int = BATCH_SIZE
) -> tuple[
    CampaignSplit,
    dict[str, Any],
    tuple[int, ...],
    tuple[int, ...],
    torch.nn.Module,
    DataLoader,
    torch.Generator,
    dict[str, Any],
]:
    split, parent_provenance, train_groups, valid_groups = _verify_static_ancestry(recipe)
    teacher_bindings = trainer.bind_fold_teachers(TEACHERS, split, ("train", "valid"))
    for binding in teacher_bindings:
        if binding["checkpoint_sha256"] != TEACHER_CHECKPOINT_SHA256[binding["run"]]:
            raise ValueError(f"H233 teacher checkpoint changed: {binding['run']}")
        expected_prediction = TEACHER_PREDICTION_SHA256[binding["run"]][binding["split"]]
        if binding["prediction_sha256"] != expected_prediction:
            raise ValueError(
                f"H233 teacher prediction changed: {binding['run']}/{binding['split']}"
            )
    propensity_path = ROOT / "artifacts/runs" / PROPENSITY_RUN / "model.pt"
    propensity_checkpoint_sha256 = file_sha256(propensity_path)
    if propensity_checkpoint_sha256 != PROPENSITY_CHECKPOINT_SHA256:
        raise ValueError("H233 propensity checkpoint differs from the C1 recipe")
    propensity_model = load_propensity(propensity_path, device)
    prop_cache = {
        name: trainer.prop_cache_path(PROPENSITY_RUN, name) for name in ("train", "valid")
    }
    for name, path in prop_cache.items():
        if file_sha256(path) != recipe["propensity_cache_sha256"][name]:
            raise ValueError(f"H233 propensity cache changed: {name}")
    train_data = trainer.Sequences("train", train_groups, list(TEACHERS), prop_cache["train"])
    valid_data = trainer.Sequences("valid", valid_groups, list(TEACHERS), prop_cache["valid"])
    dataset = ConcatDataset((train_data, valid_data))
    sampler_generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=sampler_generator,
    )
    args = SimpleNamespace(
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
    provenance = {
        "split_manifest_sha256": split.sha256,
        "parent_checkpoint_sha256": CHECKPOINT_SHA256,
        "parent_provenance_sha256": PARENT_PROVENANCE_SHA256,
        "parent_run": PARENT_RUN,
        "parent_epoch": PARENT_EPOCH,
        "fit_row_groups": {"train": list(train_groups), "valid": list(valid_groups)},
        "evaluation_row_groups": list(split.evaluation_groups),
        "teacher_bindings": teacher_bindings,
        "propensity_run": PROPENSITY_RUN,
        "propensity_checkpoint_sha256": propensity_checkpoint_sha256,
        "propensity_cache_sha256": {
            name: recipe["propensity_cache_sha256"][name] for name in prop_cache
        },
        "objective": {
            "kind": "E511 distilled primary-pair weighted MSE, chunk denominator",
            "label": "clip(y, -2, 2)",
            "teacher": "clip(mean(0.8125 * own-C1-fold0 teacher outputs), -2, 2)",
            "target": "0.3 * clipped_label + 0.7 * clipped_teacher",
            "row_weight": "main_row_weights from train_rnn.py: prop^2 + 0.03, true-mask coefficient 0.03",
            "loss_weight": "abs(clipped_label) * row_weight",
            "auxiliary_objectives": "none; frozen 23-output parent head retained; grade only pair 0:2",
        },
        "no_fit_rows_from_evaluation_groups": True,
    }
    return (
        split,
        parent_provenance,
        train_groups,
        valid_groups,
        propensity_model,
        loader,
        sampler_generator,
        {
            "args": args,
            "provenance": provenance,
            "dataset": dataset,
            "loader_length": len(loader),
        },
    )


def _build_batch(
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    propensity_model: torch.nn.Module,
    weighting_args: SimpleNamespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw, target, scored, prop_cached = (item.to(device, non_blocking=True) for item in batch)
    inputs = trainer.build_inputs(raw)
    context = inputs[..., N_FEATURES : N_FEATURES + CONTEXT_WIDTH]
    label = target[..., :2].float().clamp(-2.0, 2.0)
    teacher = target[..., 2:4].float().clamp(-2.0, 2.0)
    blended_target = (1.0 - DISTILL_ALPHA) * label + DISTILL_ALPHA * teacher
    row = torch.arange(raw.shape[1], device=device)[None].expand(raw.shape[0], -1)
    valid = row >= WARMUP
    row_weights = trainer.main_row_weights(
        inputs, label, valid, scored, propensity_model, prop_cached, weighting_args, raw
    )
    return inputs, context, label, blended_target, row_weights


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
    parent_hidden_width = parent.head[0].in_features
    if hidden.shape[-1] != parent_hidden_width:
        raise RuntimeError("parent hidden representation width differs from its readout input")
    outputs = heads.forward_all(parent_output, hidden, event_context)
    if any(parameter.requires_grad for parameter in parent.parameters()):
        raise RuntimeError("frozen parent parameter became trainable")
    optimizer.zero_grad(set_to_none=True)
    row_weight = row_weights[:, sl]
    target_chunk = target[:, sl]
    losses: list[torch.Tensor] = []
    for name in READOUT_MODES:
        loss = primary_pair_wmse(outputs[name], target_chunk, row_weight, label[:, sl])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"H233 {name} readout loss became nonfinite")
        loss.backward()
        clip_grad_norm_(heads.controls[name].parameters(), max_norm=1.0, error_if_nonfinite=True)
        losses.append(loss)
    optimizer.step()
    return {name: loss.detach() for name, loss in zip(READOUT_MODES, losses, strict=True)}


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _assert_frozen_parent_state(
    parent: torch.nn.Module, candidate_state: dict[str, torch.Tensor]
) -> None:
    frozen_state = parent.state_dict()
    if frozen_state.keys() != candidate_state.keys() or any(
        not torch.equal(frozen_state[name].detach().cpu(), candidate_state[name].detach().cpu())
        for name in frozen_state
    ):
        raise ValueError("H233 checkpoint parent tensors differ from the pinned C1 seed-3 weights")


def _checkpoint(
    parent: torch.nn.Module,
    heads: ConditionalReadoutHeads,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loader_generator: torch.Generator,
    provenance: dict[str, Any],
    recipe: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "recipe_id": recipe["recipe_id"],
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "parent_model": _cpu_state_dict(parent),
        "heads": _cpu_state_dict(heads),
        "optimizer": optimizer.state_dict(),
        "loader_generator_state": loader_generator.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "provenance": provenance,
        "recipe_sha256": file_sha256(RECIPE_PATH),
        "parent_checkpoint_sha256": CHECKPOINT_SHA256,
        "training_config": {
            "seed": SEED,
            "rank": RANK,
            "batch_size": BATCH_SIZE,
            "chunk_rows": CHUNK_ROWS,
            "epochs": EPOCHS,
            "fixed_checkpoint_epoch": CHECKPOINT_EPOCH,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "trunk_precision": "fp16 autocast on CUDA; fp32 otherwise",
            "readout_precision": "fp32",
        },
    }


def _restore_checkpoint(
    path: Path,
    parent: torch.nn.Module,
    heads: ConditionalReadoutHeads,
    optimizer: torch.optim.Optimizer,
    loader_generator: torch.Generator,
    recipe: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("recipe_id") != recipe["recipe_id"]:
        raise ValueError("resume checkpoint belongs to a different recipe")
    if checkpoint.get("parent_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("resume checkpoint has a different frozen parent")
    if checkpoint.get("recipe_sha256") != file_sha256(RECIPE_PATH):
        raise ValueError("resume checkpoint recipe hash does not match the current frozen recipe")
    _assert_frozen_parent_state(parent, checkpoint["parent_model"])
    parent.load_state_dict(checkpoint["parent_model"], strict=True)
    freeze_parent(parent)
    heads.load_state_dict(checkpoint["heads"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    loader_generator.set_state(checkpoint["loader_generator_state"])
    torch.random.set_rng_state(checkpoint["torch_rng_state"])
    if checkpoint.get("cuda_rng_state") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return int(checkpoint["next_epoch"]), checkpoint["provenance"]


def _validate_anchor_reference(split: str, row_groups: tuple[int, ...]) -> dict[str, Any] | None:
    if split != "valid":
        return None
    prediction = ROOT / "cache" / f"pred_{PARENT_RUN}_valid.npy"
    metadata_path = prediction.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("checkpoint_sha256") != PARENT_MODEL_PT_SHA256:
        raise ValueError("the existing C1 parent prediction does not bind to its verified model.pt")
    if metadata.get("prediction_sha256") != PARENT_VALID_PREDICTION_SHA256:
        raise ValueError("the existing C1 anchor prediction differs from its frozen E511 bytes")
    if tuple(map(int, metadata.get("row_groups", []))) != row_groups:
        raise ValueError("the C1 anchor prediction row order differs from the export split")
    if file_sha256(prediction) != metadata.get("prediction_sha256"):
        raise ValueError("the existing C1 anchor prediction bytes changed")
    return {
        "prediction_path": str(prediction.relative_to(ROOT)),
        "prediction_sha256": metadata["prediction_sha256"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "identity_check": "E511 verified model.pt tensors equal the pinned model_ep1.pt tensors",
    }


def _run_export(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    temporary_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or temporary_dir.exists():
        raise FileExistsError(f"refusing to overwrite H233 export destination: {output_dir}")
    recipe = _load_recipe(args.recipe)
    split, _, _, _ = _verify_static_ancestry(recipe)
    _verify_parent_alias()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("recipe_id") != recipe["recipe_id"]:
        raise ValueError("readout checkpoint does not belong to the frozen H233 recipe")
    if checkpoint.get("epoch") != CHECKPOINT_EPOCH:
        raise ValueError("export requires the fixed epoch-1 H233 checkpoint")
    if checkpoint.get("parent_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("H233 checkpoint parent hash differs from the frozen C1 seed-3 artifact")
    if checkpoint.get("recipe_sha256") != file_sha256(args.recipe.resolve()):
        raise ValueError("H233 checkpoint recipe hash differs from the export recipe")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H233 epoch-1 export requires CUDA FP16 parent inference")
    parent, _ = _load_parent(PARENT_CHECKPOINT, device)
    _assert_frozen_parent_state(parent, checkpoint["parent_model"])
    parent.load_state_dict(checkpoint["parent_model"], strict=True)
    freeze_parent(parent)
    heads = ConditionalReadoutHeads(parent.head[0].in_features, RANK).to(device)
    heads.load_state_dict(checkpoint["heads"], strict=True)
    heads.eval()

    target_index = np.load(ROOT / "cache" / f"{args.split}_targets.npz", allow_pickle=False)
    if args.split == "valid":
        row_groups = tuple(map(int, target_index["row_group"].tolist()))
        expected = set(split.valid_groups) | set(split.evaluation_groups)
        if set(row_groups) != expected:
            raise ValueError(
                "valid export must contain exactly fit-dev plus fold-0 evaluation groups"
            )
    else:
        fit_groups = checkpoint["provenance"]["fit_row_groups"]["train"]
        row_groups = tuple(map(int, fit_groups))
        if set(row_groups) != set(split.train_groups):
            raise ValueError("train export row groups do not match fixed checkpoint fit membership")
    anchor_reference = _validate_anchor_reference(args.split, row_groups)
    anchor_predictions = (
        np.load(ROOT / anchor_reference["prediction_path"], mmap_mode="r", allow_pickle=False)
        if anchor_reference is not None
        else None
    )
    dataset = trainer.Sequences(args.split, row_groups)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    temporary_dir.mkdir(parents=True)
    prediction_paths = {mode: temporary_dir / f"pred_{mode}.npy" for mode in READOUT_MODES}
    prediction_maps = {
        mode: np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(len(row_groups), SEQUENCE_LENGTH, 2),
        )
        for mode, path in prediction_paths.items()
    }
    mask_path = temporary_dir / "selected_mask.npy"
    mask_map = np.lib.format.open_memmap(
        mask_path, mode="w+", dtype=np.bool_, shape=(len(row_groups), SEQUENCE_LENGTH)
    )
    position = 0
    started = time.perf_counter()
    anchor_abs_max = 0.0
    anchor_squared_error = 0.0
    anchor_value_count = 0
    with torch.no_grad():
        for raw, _, scored in loader:
            raw = raw.to(device, non_blocking=True)
            inputs = trainer.build_inputs(raw)
            context = inputs[..., N_FEATURES : N_FEATURES + CONTEXT_WIDTH]
            batch_size = raw.shape[0]
            end_position = position + batch_size
            mask = scored.numpy().copy()
            mask[:, :WARMUP] = False
            mask_map[position:end_position] = mask
            for sl, parent_output, hidden, event_context in iter_frozen_parent_chunks(
                parent,
                inputs,
                context,
                CHUNK_ROWS,
                amp_enabled=device.type == "cuda",
            ):
                outputs = heads.forward_all(parent_output, hidden, event_context)
                if anchor_predictions is not None:
                    actual_anchor = parent_output[..., :2].float().cpu().numpy()
                    expected_anchor = anchor_predictions[position:end_position, sl]
                    difference = actual_anchor - expected_anchor
                    anchor_abs_max = max(anchor_abs_max, float(np.abs(difference).max()))
                    anchor_squared_error += float(
                        np.square(difference.astype(np.float64, copy=False)).sum()
                    )
                    anchor_value_count += difference.size
                for mode in READOUT_MODES:
                    prediction_maps[mode][position:end_position, sl] = (
                        outputs[mode].float().cpu().numpy()
                    )
            position = end_position
    for mapping in prediction_maps.values():
        mapping.flush()
    mask_map.flush()
    del prediction_maps, mask_map
    if position != len(row_groups):
        raise RuntimeError(f"export wrote {position} of {len(row_groups)} row groups")
    anchor_parity = None
    if anchor_predictions is not None:
        anchor_rmse = float(np.sqrt(anchor_squared_error / max(1, anchor_value_count)))
        anchor_parity = {
            "max_absolute_error": anchor_abs_max,
            "rmse": anchor_rmse,
            "tolerance": 1e-3,
            "passed": anchor_abs_max <= 1e-3,
            "reference": anchor_reference,
        }
        if not anchor_parity["passed"]:
            _json_write(
                temporary_dir / "export_failure.json",
                {
                    "status": "failed_parent_anchor_parity",
                    "parent_anchor_parity": anchor_parity,
                    "recipe_id": recipe["recipe_id"],
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "split": args.split,
                    "row_groups": list(row_groups),
                },
            )
            raise RuntimeError(
                f"H233 frozen-parent anchor differs from E511 prediction: {anchor_abs_max:.6g}"
            )
    metadata = {
        "schema_version": 1,
        "recipe_id": recipe["recipe_id"],
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": CHECKPOINT_EPOCH,
        "parent_checkpoint_sha256": CHECKPOINT_SHA256,
        "split": args.split,
        "row_groups": list(row_groups),
        "prediction_files": {
            mode: {
                "path": path.name,
                "sha256": file_sha256(path),
                "shape": [len(row_groups), SEQUENCE_LENGTH, 2],
                "dtype": "float32",
            }
            for mode, path in prediction_paths.items()
        },
        "selected_mask": {
            "path": mask_path.name,
            "sha256": file_sha256(mask_path),
            "warmup_rows_excluded": WARMUP,
            "source": "supplied SequenceFile scored mask AND row index >= 99",
        },
        "parent_anchor_reference": anchor_reference,
        "parent_anchor_parity": anchor_parity,
        "official_metric": "global_wp_clip2_v1; no scoring or calibration performed by export",
        "feature_order": list(CONTEXT_FEATURE_NAMES),
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
    }
    _json_write(temporary_dir / "provenance.json", metadata)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_dir, output_dir)
    print(
        json.dumps(
            {"status": "complete", "output_dir": str(output_dir), **metadata}, sort_keys=True
        )
    )


def _run_profile(args: argparse.Namespace) -> None:
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite H233 profile receipt: {output_path}")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H233 CUDA profile requested but CUDA is not available")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    recipe = _load_recipe(args.recipe)
    _verify_static_ancestry(recipe)
    _verify_parent_alias()
    device = torch.device(args.device)
    parent, _ = _load_parent(PARENT_CHECKPOINT, device)
    heads = ConditionalReadoutHeads(parent.head[0].in_features, RANK).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    _, _, _, _, propensity_model, loader, _, data = _load_fit_data(recipe, device)
    batch = next(iter(loader))
    inputs, context, label, target, row_weights = _build_batch(
        batch, device, propensity_model, data["args"]
    )
    attempted = completed = finite_checks = 0
    loss_sums = torch.zeros(len(READOUT_MODES), device=device)
    durations: list[float] = []
    torch.cuda.reset_peak_memory_stats(device)
    parent_chunks = iter_frozen_parent_chunks(parent, inputs, context, CHUNK_ROWS, amp_enabled=True)
    requested_updates = args.warmup_chunks + args.steps
    for chunk_index in range(requested_updates):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            parent_chunk = next(parent_chunks)
        except StopIteration:
            break
        attempted += 1
        losses = _update_chunk(
            parent,
            heads,
            optimizer,
            parent_chunk,
            target,
            label,
            row_weights,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        completed += 1
        finite_checks += 1
        if chunk_index >= args.warmup_chunks:
            durations.append(elapsed)
            loss_sums += torch.stack([losses[name] for name in READOUT_MODES])
    if attempted != completed or finite_checks != attempted or not durations:
        raise RuntimeError("H233 profile update evidence is incomplete")
    result = {
        "schema_version": 1,
        "recipe_id": recipe["recipe_id"],
        "stage": "profile",
        "status": "complete",
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "chunk_rows": CHUNK_ROWS,
        "warmup_chunks": args.warmup_chunks,
        "timed_chunks": args.steps,
        "optimizer_attempts": attempted,
        "optimizer_completed_updates": completed,
        "amp_skipped_updates": 0,
        "loss_checks": finite_checks * len(READOUT_MODES),
        "gradient_checks": finite_checks * len(READOUT_MODES),
        "loss_finite": True,
        "gradient_finite": True,
        "timed_median_seconds_per_chunk": float(np.median(durations)),
        "timed_mean_seconds_per_chunk": float(np.mean(durations)),
        "rows_per_second": BATCH_SIZE * CHUNK_ROWS / float(np.mean(durations)),
        "readout_mean_losses": {
            name: float(value)
            for name, value in zip(
                READOUT_MODES, (loss_sums / args.steps).cpu().tolist(), strict=True
            )
        },
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "interpretation": "one shared frozen-parent forward per 2000-row chunk; all four FP32 heads update",
        "fit_started": False,
    }
    _json_write(output_path, result)
    print(json.dumps(result, sort_keys=True))


def _run_fit(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if args.resume_from:
        if not output_dir.is_dir():
            raise FileNotFoundError("resume requires the existing output directory")
    elif output_dir.exists():
        raise FileExistsError(f"refusing to overwrite H233 run directory: {output_dir}")
    recipe = _load_recipe(args.recipe)
    _verify_static_ancestry(recipe)
    _verify_parent_alias()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("H233 frozen recipe requires CUDA FP16 trunk execution")
    torch.manual_seed(SEED)
    torch.set_num_threads(args.cpu_threads)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    parent, _ = _load_parent(PARENT_CHECKPOINT, device)
    heads = ConditionalReadoutHeads(parent.head[0].in_features, RANK).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    split, parent_provenance, _, _, propensity_model, loader, loader_generator, data = (
        _load_fit_data(recipe, device)
    )
    restored_provenance = None
    if args.resume_from:
        if not output_dir.is_dir():
            raise FileNotFoundError("resume requires the existing output directory")
        first_epoch, restored_provenance = _restore_checkpoint(
            args.resume_from.resolve(), parent, heads, optimizer, loader_generator, recipe
        )
        if first_epoch >= EPOCHS:
            raise ValueError("resume checkpoint has no remaining frozen-recipe epoch")
    else:
        output_dir.mkdir(parents=True)
        first_epoch = 0
    provenance = {
        **data["provenance"],
        "recipe_sha256": file_sha256(args.recipe.resolve()),
        "implementation_source_sha256": {
            "scripts/train_conditional_readout.py": file_sha256(Path(__file__).resolve()),
            "src/wnn/conditional_readout.py": file_sha256(ROOT / "src/wnn/conditional_readout.py"),
            "scripts/train_rnn.py": file_sha256(ROOT / "scripts/train_rnn.py"),
        },
        "parent_args_sha256": file_sha256(PARENT_PROVENANCE),
        "parent_auxiliary_outputs_retained": 21,
        "trained_outputs": "four independent rank-32 controls for primary pair 0:2 only",
        "fit_group_counts": {
            "train": len(parent_provenance["fit_row_groups"]["train"]),
            "valid": len(parent_provenance["fit_row_groups"]["valid"]),
            "evaluation_not_fit": len(split.evaluation_groups),
        },
        "random_seed": SEED,
        "run_name": args.run_name,
        "elapsed_epoch_seconds": [],
        "epoch_losses": [],
        "epoch_update_evidence": [],
        "status": "fitting",
    }
    if restored_provenance is not None:
        provenance = restored_provenance
        provenance["status"] = "fitting"
    for epoch in range(first_epoch, EPOCHS):
        started = time.perf_counter()
        heads.train()
        sums = torch.zeros(len(READOUT_MODES), device=device)
        update_count = 0
        for batch in loader:
            inputs, context, label, target, row_weights = _build_batch(
                batch, device, propensity_model, data["args"]
            )
            chunks = iter_frozen_parent_chunks(
                parent,
                inputs,
                context,
                CHUNK_ROWS,
                amp_enabled=device.type == "cuda",
            )
            for parent_chunk in chunks:
                losses = _update_chunk(
                    parent,
                    heads,
                    optimizer,
                    parent_chunk,
                    target,
                    label,
                    row_weights,
                )
                sums += torch.stack([losses[name] for name in READOUT_MODES])
                update_count += 1
        elapsed = time.perf_counter() - started
        epoch_losses = {
            name: float(value)
            for name, value in zip(
                READOUT_MODES, (sums / max(1, update_count)).cpu().tolist(), strict=True
            )
        }
        provenance["elapsed_epoch_seconds"].append(elapsed)
        provenance["epoch_losses"].append(epoch_losses)
        provenance["epoch_update_evidence"].append(
            {
                "optimizer_attempts": update_count,
                "optimizer_completed_updates": update_count,
                "amp_skipped_updates": 0,
                "loss_checks": update_count * len(READOUT_MODES),
                "gradient_checks": update_count * len(READOUT_MODES),
                "loss_finite": True,
                "gradient_finite": True,
            }
        )
        provenance["last_epoch_completed"] = epoch
        if epoch == CHECKPOINT_EPOCH:
            provenance["selected_epoch"] = CHECKPOINT_EPOCH
            provenance["status"] = "complete"
        else:
            provenance["status"] = "epoch_complete"
        payload = _checkpoint(parent, heads, optimizer, epoch, loader_generator, provenance, recipe)
        snapshot_path = output_dir / f"model_ep{epoch}.pt"
        _torch_save(snapshot_path, payload)
        provenance.setdefault("checkpoint_sha256", {})[snapshot_path.name] = file_sha256(
            snapshot_path
        )
        if epoch == CHECKPOINT_EPOCH:
            _torch_save(output_dir / "model.pt", payload)
            provenance["checkpoint_sha256"]["model.pt"] = file_sha256(output_dir / "model.pt")
        _json_write(output_dir / "provenance.json", provenance)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "updates": update_count,
                    "seconds": elapsed,
                    "losses": epoch_losses,
                    "checkpoint": str(snapshot_path),
                    "status": provenance["status"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile", help="bounded representative CUDA update profile")
    profile.add_argument("--recipe", type=Path, default=RECIPE_PATH)
    profile.add_argument("--device", default="cuda")
    profile.add_argument("--warmup-chunks", type=int, default=2)
    profile.add_argument("--steps", type=int, default=8)
    profile.add_argument("--output", type=Path, required=True)
    profile.set_defaults(handler=_run_profile)
    fit = commands.add_parser("fit", help="fit only the four independent readout heads")
    fit.add_argument("--recipe", type=Path, default=RECIPE_PATH)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--run-name", default="bd26_h233_c1s3_rank32_f0")
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--resume-from", type=Path)
    fit.add_argument("--cpu-threads", type=int, default=1)
    fit.set_defaults(handler=_run_fit)
    export = commands.add_parser("export", help="stream causal predictions from fixed epoch 1")
    export.add_argument("--recipe", type=Path, default=RECIPE_PATH)
    export.add_argument("--device", default="cuda")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--split", choices=("valid", "train"), default="valid")
    export.add_argument("--batch-size", type=int, default=8)
    export.add_argument("--output-dir", type=Path, required=True)
    export.set_defaults(handler=_run_export)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "profile" and (args.warmup_chunks < 0 or args.steps < 1):
        parser.error("profile warmup must be nonnegative and timed steps positive")
    if (
        args.command == "profile"
        and args.warmup_chunks + args.steps > SEQUENCE_LENGTH // CHUNK_ROWS
    ):
        parser.error("one representative batch has only 10 chunks at the frozen 2000-row setting")
    if args.command == "export" and args.batch_size < 1:
        parser.error("export batch size must be positive")
    if args.command == "fit" and args.cpu_threads < 1:
        parser.error("CPU thread count must be positive")
    args.handler(args)


if __name__ == "__main__":
    main()
