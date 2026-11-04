"""Build a fold-bound source-scale and four-family teacher-target cache."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from wnn.distillation import TeacherSource, combine_teacher_chunk
from wnn.distillation_cache import (
    CACHE_FORMAT,
    CACHE_VERSION,
    FAMILY_ORDER,
    SEQUENCE_ROWS,
    file_sha256,
    row_groups_sha256,
)
from wnn.final_submit_manifest import FinalFitMembership, load_final_fit_membership
from wnn.splits import (
    CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA,
    CampaignSplit,
    load_campaign_split,
)
from wnn.training.chronological_teacher import (
    validate_chronological_teacher_checkpoint,
    validate_chronological_teacher_membership,
    validate_chronological_train_index_order,
)
from wnn.training.final_submit_bridge import (
    final_fit_identity,
    validate_final_fit_artifact,
    validate_final_fit_provenance,
    validate_source_validation_receipt,
)
from wnn.training.final_submit_bridge_teachers import audit_final_fit_ancestor_closure

# Parsed JSON fields are intentionally dynamic and retain their source schema.

ROOT = Path(os.environ.get("WNN_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))

RUN_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
EPOCH_CONVENTION = {
    "source_zero_based": True,
    "friend_at_ep1": "second additional epoch",
    "student_epochs": 2,
    "student_checkpoint_policy": "fixed",
    "student_checkpoint_epoch": 1,
}
OUTPUT_TARGETS = {
    "all": ("main_source", "main_normalized", "main_homogeneous", "families"),
    "homogeneous-only": ("main_homogeneous",),
    "student-required": ("main_source", "families"),
}
TARGET_WIDTHS = {"main_source": 2, "main_normalized": 2, "main_homogeneous": 2, "families": 8}


def estimate_output_bytes(fit_rows: int, output_profile: str) -> int:
    """Estimate exact float payload bytes for one declared cache profile."""
    if fit_rows < 0:
        raise ValueError("fit row count must be nonnegative")
    try:
        width = sum(TARGET_WIDTHS[target] for target in OUTPUT_TARGETS[output_profile])
    except KeyError as error:
        raise ValueError(f"unknown distillation cache output profile: {output_profile}") from error
    return fit_rows * SEQUENCE_ROWS * np.dtype(np.float32).itemsize * width


def parse_teacher(value: str) -> TeacherSource:
    """Parse ``family=run[@epN]*coefficient`` into one source member."""
    family_and_run, separator, coefficient_text = value.partition("*")
    family, equals, run_spec = family_and_run.partition("=")
    run, epoch_separator, epoch_text = run_spec.partition("@ep")
    if (
        not separator
        or not equals
        or not family
        or not run
        or not RUN_NAME.fullmatch(run)
        or epoch_separator
        and not epoch_text.isdecimal()
    ):
        raise argparse.ArgumentTypeError("teacher must be FAMILY=RUN[@epN]*COEFFICIENT")
    try:
        coefficient = float(coefficient_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("teacher coefficient must be numeric") from error
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise argparse.ArgumentTypeError("teacher coefficient must be finite and positive")
    epoch = int(epoch_text) if epoch_separator else 3
    return TeacherSource(family, run, coefficient, epoch)


def _expected_groups(split: CampaignSplit | FinalFitMembership, name: str) -> tuple[int, ...]:
    return split.train_groups if name == "train" else split.valid_groups


def _target_order(data_root: Path, split_name: str) -> list[int]:
    path = data_root / "cache" / f"{split_name}_targets.npz"
    with np.load(path, allow_pickle=False) as targets:
        raw_groups = np.asarray(targets["row_group"])
    if raw_groups.ndim != 1 or raw_groups.dtype.kind not in "iu":
        raise ValueError(f"target index must be a one-dimensional integer array: {path}")
    groups = raw_groups.astype(np.int64, copy=False)
    if len(np.unique(groups)) != len(groups):
        raise ValueError(f"target index repeats a row group: {path}")
    return list(map(int, groups))


def _audit_ancestor_closure(
    run: str,
    split: CampaignSplit | FinalFitMembership,
    checkpoint_root: Path,
    checkpoint_file: str = "model.pt",
    expected_checkpoint_sha256: str | None = None,
    seen: set[str] | None = None,
    data_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Verify each learned ancestor stays inside the same clean outer fold."""
    visited = set() if seen is None else seen
    if run in visited:
        raise ValueError(f"checkpoint ancestry contains a cycle at {run}")
    visited.add(run)
    run_root = checkpoint_root / run
    provenance_path = run_root / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"ancestor {run} has no training provenance")
    provenance = json.loads(provenance_path.read_text())
    fit_counts = _verify_run_membership(provenance, split, run)
    args = provenance.get("args", {})
    if not isinstance(args, dict):
        raise ValueError(f"ancestor {run} has invalid training arguments")

    if isinstance(split, FinalFitMembership):
        if data_root is None:
            raise ValueError("final-fit ancestry audit requires the validated data root")
        closure = audit_final_fit_ancestor_closure(run, checkpoint_file, split, data_root)
        if (
            expected_checkpoint_sha256
            and closure[0]["checkpoint_sha256"] != expected_checkpoint_sha256
        ):
            raise ValueError(f"ancestor {run} checkpoint bytes disagree with its recorded hash")
        return closure

    checkpoint_path = run_root / checkpoint_file
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if expected_checkpoint_sha256 and checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(f"ancestor {run} checkpoint bytes disagree with its recorded hash")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    if (
        checkpoint.get("epoch") != checkpoint_args.get("checkpoint_epoch")
        or checkpoint_args.get("checkpoint_policy") != "fixed"
    ):
        raise ValueError(f"ancestor {run} is not a fixed clean-fold checkpoint")
    checkpoint_provenance = checkpoint.get("provenance", {})
    if isinstance(split, FinalFitMembership):
        validate_final_fit_artifact(
            checkpoint_args, checkpoint_provenance, split, checkpoint_epoch=checkpoint.get("epoch")
        )
        validate_final_fit_artifact(
            args, provenance, split, checkpoint_epoch=checkpoint.get("epoch")
        )
    elif checkpoint_args.get("split_manifest_sha256") != split.sha256:
        raise ValueError(f"ancestor {run} uses a different split manifest")
    elif split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
        if checkpoint_args.get("no_eval_during_fit") is not False:
            raise ValueError(f"ancestor {run} skipped required official validation evaluation")
        validate_chronological_teacher_checkpoint(
            checkpoint,
            provenance,
            split,
            checkpoint_file,
            checkpoint_sha256,
            run,
        )
    else:
        if not checkpoint_args.get("no_eval_during_fit"):
            raise ValueError(f"ancestor {run} is not a fixed clean-fold checkpoint")
        if checkpoint_provenance.get("fit_row_groups") != provenance.get("fit_row_groups"):
            raise ValueError(f"ancestor {run} checkpoint membership differs from run provenance")
        _verify_run_membership(checkpoint_provenance, split, run)

    closure = [
        {
            "run": run,
            "provenance_sha256": file_sha256(provenance_path),
            "checkpoint_file": checkpoint_file,
            "checkpoint_sha256": checkpoint_sha256,
            "split_manifest_sha256": None
            if isinstance(split, FinalFitMembership)
            else split.sha256,
            "final_fit_manifest_sha256": (
                split.manifest_sha256 if isinstance(split, FinalFitMembership) else None
            ),
            "final_fit_members_sha256": (
                split.fit_members_sha256 if isinstance(split, FinalFitMembership) else None
            ),
            "source_validation_sha256": (
                split.source_validation_sha256 if isinstance(split, FinalFitMembership) else None
            ),
            "fit_group_counts": fit_counts,
            "fit_row_group_hashes": {
                name: row_groups_sha256(list(map(int, provenance["fit_row_groups"].get(name, []))))
                for name in ("train", "valid")
            },
        }
    ]
    init = str(args.get("init") or "")
    if init:
        parent, separator, epoch_text = init.partition("@ep")
        if not RUN_NAME.fullmatch(parent) or separator and not epoch_text.isdecimal():
            raise ValueError(f"ancestor {run} has an unrecognized initializer: {init}")
        parent_path = (
            checkpoint_root
            / parent
            / (f"model_ep{int(epoch_text)}.pt" if separator else "model.pt")
        )
        parent_sha256 = file_sha256(parent_path)
        if provenance.get("initial_checkpoint_sha256") != parent_sha256:
            raise ValueError(f"ancestor {run} initializer hash is not bound to its provenance")
        closure.extend(
            _audit_ancestor_closure(
                parent,
                split,
                checkpoint_root,
                f"model_ep{int(epoch_text)}.pt" if separator else "model.pt",
                parent_sha256,
                visited.copy(),
                data_root,
            )
        )

    propensity_run = str(args.get("propensity") or "")
    if propensity_run:
        if not RUN_NAME.fullmatch(propensity_run):
            raise ValueError(f"ancestor {run} has an unrecognized propensity source")
        propensity_path = checkpoint_root / propensity_run / "model.pt"
        propensity_sha256 = file_sha256(propensity_path)
        if provenance.get("propensity_checkpoint_sha256") != propensity_sha256:
            raise ValueError(f"ancestor {run} propensity hash is not bound to its provenance")
        closure.extend(
            _audit_ancestor_closure(
                propensity_run,
                split,
                checkpoint_root,
                "model.pt",
                propensity_sha256,
                visited.copy(),
                data_root,
            )
        )
    return closure


def _verify_run_membership(
    provenance: dict[str, Any], split: CampaignSplit | FinalFitMembership, run: str
) -> dict[str, int]:
    """Require source fitting groups to stay inside the clean fold's fit pool."""
    if isinstance(split, FinalFitMembership):
        args = provenance.get("args")
        if isinstance(args, dict):
            validate_final_fit_artifact(args, provenance, split)
        else:
            validate_final_fit_provenance(provenance, split)
        return {"train": len(split.train_groups), "valid": len(split.valid_groups)}
    if provenance.get("split_manifest_sha256") != split.sha256:
        raise ValueError(f"teacher {run} uses a different split manifest")
    if split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
        validate_chronological_teacher_membership(provenance, split, run)
        return {"train": len(split.train_groups), "valid": 0}
    fit_groups = provenance.get("fit_row_groups")
    if not isinstance(fit_groups, dict):
        raise ValueError(f"teacher {run} has no recorded fit membership")
    allowed = {"train": set(split.train_groups), "valid": set(split.valid_groups)}
    actual = {name: set(map(int, fit_groups.get(name, []))) for name in ("train", "valid")}
    for name, groups in allowed.items():
        if not actual[name].issubset(groups):
            raise ValueError(f"teacher {run} fits outside the clean {name} pool")
    if not actual["train"] and not actual["valid"]:
        raise ValueError(f"teacher {run} records no fitting groups")
    split.assert_fit_groups(actual["train"], actual["valid"])
    return {name: len(groups) for name, groups in actual.items()}


def _inspect_teacher(
    run: str,
    split: CampaignSplit | FinalFitMembership,
    cache_root: Path,
    checkpoint_root: Path,
    target_orders: dict[str, list[int]],
    epoch: int,
    checkpoint_file: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Verify teacher checkpoint bytes, architecture, fit groups and prediction caches."""
    if not RUN_NAME.fullmatch(run):
        raise ValueError(f"invalid teacher run name: {run}")
    run_root = checkpoint_root / run
    provenance_path = run_root / "provenance.json"
    if checkpoint_file not in (None, "model.pt", f"model_ep{epoch}.pt"):
        raise ValueError("teacher checkpoint must be the selected fixed or explicit epoch snapshot")
    checkpoint_path = run_root / (checkpoint_file or f"model_ep{epoch}.pt")
    if not provenance_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"teacher {run} lacks provenance or model_ep{epoch}.pt")
    provenance = json.loads(provenance_path.read_text())
    fit_counts = _verify_run_membership(provenance, split, run)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    expected_checkpoint_sha256 = provenance.get("checkpoint_sha256", {}).get(checkpoint_path.name)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(f"teacher {run} checkpoint bytes disagree with provenance")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args", {})
    if checkpoint.get("epoch") != epoch or args.get("checkpoint_policy") != "fixed":
        raise ValueError(f"teacher {run} checkpoint is not the requested fixed clean-fold epoch")
    if isinstance(split, FinalFitMembership):
        validate_final_fit_artifact(
            args, checkpoint.get("provenance", {}), split, checkpoint_epoch=epoch
        )
        validate_final_fit_artifact(
            provenance.get("args", {}), provenance, split, checkpoint_epoch=epoch
        )
    elif args.get("split_manifest_sha256") != split.sha256:
        raise ValueError(f"teacher {run} checkpoint uses a different split manifest")
    elif split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
        validate_chronological_teacher_checkpoint(
            checkpoint,
            provenance,
            split,
            checkpoint_path.name,
            checkpoint_sha256,
            run,
        )
    checkpoint_provenance = checkpoint.get("provenance", {})
    _verify_run_membership(checkpoint_provenance, split, run)
    model_state = checkpoint.get("model", {})
    hidden = int(args.get("hidden", 0))
    layers = int(args.get("layers", 0))
    cell = args.get("cell", "gru")
    gate_count = {"gru": 3, "lstm": 4}.get(cell)
    recurrent = model_state.get("gru.weight_ih_l0")
    input_projection = model_state.get("inp.weight")
    output_projection = model_state.get("head.2.weight")
    if (
        hidden <= 0
        or layers <= 0
        or gate_count is None
        or recurrent is None
        or input_projection is None
        or output_projection is None
        or output_projection.ndim != 2
        or output_projection.shape[0] < 2
        or output_projection.shape[1] != hidden
        or recurrent.shape[0] != gate_count * hidden
        or input_projection.shape[0] != hidden
    ):
        raise ValueError(f"teacher {run} checkpoint dimensions disagree with model arguments")
    actual_layers = sum(1 for key in model_state if re.fullmatch(r"gru\.weight_ih_l\d+", key))
    if actual_layers != layers:
        raise ValueError(f"teacher {run} checkpoint recurrent layer count differs")

    predictions: dict[str, np.ndarray] = {}
    cache_records: dict[str, dict[str, Any]] = {}
    for split_name, expected_order in target_orders.items():
        pred_path = cache_root / f"pred_{run}_{split_name}.npy"
        sidecar_path = pred_path.with_suffix(".json")
        if not pred_path.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(f"teacher {run} needs regenerated {split_name} predictions")
        sidecar = json.loads(sidecar_path.read_text())
        if (
            sidecar.get("run") != run
            or sidecar.get("split") != split_name
            or sidecar.get("epoch") != epoch
            or sidecar.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise ValueError(
                f"teacher {run} {split_name} prediction sidecar is not bound to its checkpoint"
            )
        if (
            isinstance(split, CampaignSplit)
            and sidecar.get("split_manifest_sha256") != split.sha256
        ):
            raise ValueError(f"teacher {run} {split_name} prediction uses a different split")
        if isinstance(split, FinalFitMembership) and (
            sidecar.get("final_fit_manifest_sha256") != split.manifest_sha256
            or sidecar.get("final_fit_members_sha256") != split.fit_members_sha256
            or sidecar.get("source_validation_sha256") != split.source_validation_sha256
            or sidecar.get("campaign_export_scope") != "final-fit-source-index-feature-only"
            or sidecar.get("campaign_split_manifest_sha256") is not None
            or sidecar.get("feature_only") is not True
            or sidecar.get("prediction_role")
            != "fitting-only teacher features; no independent score"
            or sidecar.get("independent_score_claim") != "forbidden"
            or sidecar.get("checkpoint_file") != checkpoint_path.name
            or sidecar.get("run_provenance_sha256") != file_sha256(provenance_path)
            or sidecar.get("row_groups_sha256")
            != row_groups_sha256(_expected_groups(split, split_name))
        ):
            raise ValueError(
                f"teacher {run} {split_name} export is not bound to final-fit ancestry"
            )
        if (
            isinstance(split, CampaignSplit)
            and split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
            and (
                split_name != "train"
                or sidecar.get("campaign_export_scope")
                != "chronological-full-train-index-feature-only"
                or sidecar.get("campaign_split_manifest_sha256") != split.sha256
                or sidecar.get("feature_only") is not True
                or sidecar.get("checkpoint_file") != checkpoint_path.name
                or sidecar.get("run_provenance_sha256") != file_sha256(provenance_path)
            )
        ):
            raise ValueError(
                f"teacher {run} train prediction is not a provenance-bound full-index export"
            )
        row_groups = list(map(int, sidecar.get("row_groups", [])))
        if row_groups != expected_order:
            raise ValueError(
                f"teacher {run} {split_name} prediction row order differs from target index"
            )
        prediction_sha256 = file_sha256(pred_path)
        if prediction_sha256 != sidecar.get("prediction_sha256"):
            raise ValueError(f"teacher {run} {split_name} prediction bytes disagree with sidecar")
        values = np.load(pred_path, mmap_mode="r", allow_pickle=False)
        if values.shape != (len(expected_order), SEQUENCE_ROWS, 2) or values.dtype != np.float32:
            raise ValueError(f"teacher {run} {split_name} prediction shape/dtype differs")
        for start in range(0, len(values), 64):
            if not np.isfinite(values[start : start + 64]).all():
                raise ValueError(f"teacher {run} {split_name} cache contains nonfinite values")
        predictions[split_name] = values
        cache_records[split_name] = {
            "prediction_path": str(pred_path),
            "prediction_sha256": prediction_sha256,
            "sidecar_sha256": file_sha256(sidecar_path),
            "checkpoint_sha256": checkpoint_sha256,
            "epoch": epoch,
            "row_groups_sha256": row_groups_sha256(row_groups),
        }
    record = {
        "run": run,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "provenance_sha256": file_sha256(provenance_path),
        "epoch": epoch,
        "architecture": args.get("arch"),
        "cell": cell,
        "hidden": hidden,
        "layers": layers,
        "seed": args.get("seed"),
        "fit_group_counts": fit_counts,
        "fit_row_group_hashes": {
            name: row_groups_sha256(list(map(int, provenance["fit_row_groups"][name])))
            for name in ("train", "valid")
        },
        "predictions": cache_records,
    }
    if isinstance(split, FinalFitMembership):
        record.update(
            {
                "final_fit_manifest_sha256": split.manifest_sha256,
                "final_fit_members_sha256": split.fit_members_sha256,
                "source_validation_sha256": split.source_validation_sha256,
                "prediction_role": "fitting-only teacher targets; no independent score",
                "independent_score_claim": "forbidden",
            }
        )
    return record, predictions, cache_records


def _array_entry(path: Path, row_groups: list[int], array: np.ndarray) -> dict[str, Any]:
    """Return a content and row identity record for a closed output memmap."""
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": "float32",
        "row_groups": row_groups,
        "row_groups_sha256": row_groups_sha256(row_groups),
    }


def build_cache(args: argparse.Namespace) -> Path:
    """Create the declared split-bound or explicit final-fit distillation targets."""
    data_root = args.data_root.resolve()
    cache_root = args.teacher_cache_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    final_fit_manifest = getattr(args, "final_fit_manifest", None)
    final_fit_receipt = getattr(args, "final_fit_source_validation_receipt", None)
    final_fit_receipt_sha256 = getattr(args, "final_fit_source_validation_sha256", None)
    if final_fit_manifest is not None:
        if args.split_manifest is not None or final_fit_receipt is None:
            raise ValueError("final-fit cache requires only its manifest and source receipt")
        split = load_final_fit_membership(
            final_fit_manifest.resolve(),
            data_root,
            final_fit_receipt_sha256,
        )
        validate_source_validation_receipt(
            final_fit_receipt,
            final_fit_receipt_sha256,
            split,
        )
    else:
        if args.split_manifest is None:
            raise ValueError("cache build requires --split-manifest or --final-fit-manifest")
        split = load_campaign_split(args.split_manifest.resolve(), data_root)
    if any(source.epoch < 0 for source in args.teacher):
        raise ValueError("teacher epochs must be nonnegative zero-based indices")
    if abs(args.expected_coefficient_sum - 0.8125) > 1e-7:
        raise ValueError("source-scale cache must retain the declared 0.8125 coefficient sum")
    if len(args.teacher) < 4 or set(source.family for source in args.teacher) != set(FAMILY_ORDER):
        raise ValueError("teacher sources must cover the four declared families")
    if len({source.run for source in args.teacher}) != len(args.teacher):
        raise ValueError("each source run may appear only once")
    total = math.fsum(source.coefficient for source in args.teacher)
    if abs(total - args.expected_coefficient_sum) > 1e-7:
        raise ValueError(f"teacher coefficients sum to {total}, expected 0.8125")
    if args.homogeneous_family not in FAMILY_ORDER:
        raise ValueError("homogeneous family must be one of the declared teacher families")
    if args.output_profile not in OUTPUT_TARGETS:
        raise ValueError(f"unknown distillation cache output profile: {args.output_profile}")
    if args.chunk_sequences < 1 or args.chunk_sequences > 64:
        raise ValueError("chunk sequence count must be between 1 and 64")

    if isinstance(split, FinalFitMembership):
        target_orders = {name: list(_expected_groups(split, name)) for name in ("train", "valid")}
        for name in ("train", "valid"):
            if _target_order(data_root, name) != target_orders[name]:
                raise ValueError(f"{name} target index differs from final-fit source row order")
    else:
        target_orders = {
            "train": _target_order(data_root, "train"),
            "valid": (
                []
                if split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
                else _target_order(data_root, "valid")
            ),
        }
    if isinstance(split, CampaignSplit) and split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA:
        validate_chronological_train_index_order(data_root, target_orders["train"], split, "train")
    row_groups = {
        name: [
            group for group in target_orders[name] if group in set(_expected_groups(split, name))
        ]
        for name in ("train", "valid")
    }
    for name in ("train", "valid"):
        if set(row_groups[name]) != set(_expected_groups(split, name)):
            raise ValueError(f"target index omits {name} fitting groups")

    fit_rows = sum(len(groups) for groups in row_groups.values())
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to replace existing cache directory: {output_path}")
    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"refusing to reuse partial cache directory: {partial}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    estimated_bytes = estimate_output_bytes(fit_rows, args.output_profile)
    usage = shutil.disk_usage(output_path.parent)
    free_floor = int(args.min_free_gb * (1 << 30))
    output_budget = int(args.output_budget_gb * (1 << 30))
    if estimated_bytes > output_budget or usage.free - estimated_bytes < free_floor:
        raise OSError(
            f"distillation cache needs {estimated_bytes} bytes; available budget while preserving "
            f"{free_floor} free bytes is {max(0, usage.free - free_floor)}"
        )

    records: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    prediction_splits = (
        ("train",)
        if isinstance(split, CampaignSplit) and split.schema == CHRONOLOGICAL_PRIMARY_SPLIT_SCHEMA
        else ("train", "valid")
    )
    for source in args.teacher:
        if source.family not in FAMILY_ORDER:
            raise ValueError(f"unknown teacher family: {source.family}")
        checkpoint_file = (
            "model.pt"
            if getattr(args, "prediction_checkpoint", "epoch") == "selected-fixed"
            else f"model_ep{source.epoch}.pt"
        )
        ancestor_closure = _audit_ancestor_closure(
            source.run,
            split,
            checkpoint_root,
            checkpoint_file,
            data_root=data_root,
        )
        record, arrays, _ = _inspect_teacher(
            source.run,
            split,
            cache_root,
            checkpoint_root,
            {name: target_orders[name] for name in prediction_splits},
            source.epoch,
            checkpoint_file,
        )
        record["family"] = source.family
        record["coefficient"] = source.coefficient
        record["ancestor_closure"] = ancestor_closure
        records.append(record)
        predictions[source.run] = arrays

    partial.mkdir()
    file_entries: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "valid"):
        groups = row_groups[split_name]
        position = {group: index for index, group in enumerate(target_orders[split_name])}
        shapes = {
            target: (len(groups), SEQUENCE_ROWS, TARGET_WIDTHS[target])
            for target in OUTPUT_TARGETS[args.output_profile]
        }
        if not groups:
            for name, shape in shapes.items():
                path = partial / f"{name}_{split_name}.npy"
                empty = np.empty(shape, dtype=np.float32)
                np.save(path, empty, allow_pickle=False)
                file_entries[f"{name}_{split_name}"] = _array_entry(path, groups, empty)
            continue
        outputs = {
            name: np.lib.format.open_memmap(
                partial / f"{name}_{split_name}.npy", mode="w+", dtype=np.float32, shape=shape
            )
            for name, shape in shapes.items()
        }
        for start in range(0, len(groups), args.chunk_sequences):
            selected = groups[start : start + args.chunk_sequences]
            rows = np.asarray([position[group] for group in selected], dtype=np.int64)
            source_chunk = {
                run: np.asarray(arrays[split_name][rows], dtype=np.float32)
                for run, arrays in predictions.items()
            }
            main, normalized, homogeneous, families = combine_teacher_chunk(
                source_chunk, args.teacher, FAMILY_ORDER, args.homogeneous_family
            )
            stop = start + len(selected)
            target_values = (
                ("main_source", main),
                ("main_normalized", normalized),
                ("main_homogeneous", homogeneous),
                ("families", families),
            )
            for name, values in target_values:
                if name not in outputs:
                    continue
                outputs[name][start:stop] = values
        for array in outputs.values():
            array.flush()
        for name, array in outputs.items():
            key = f"{name}_{split_name}"
            path = partial / f"{key}.npy"
            file_entries[key] = _array_entry(path, groups, array)
            del array
        del outputs

    metadata: dict[str, Any] = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "fold": None if isinstance(split, FinalFitMembership) else split.fold,
        "split_manifest": None if isinstance(split, FinalFitMembership) else str(split.path),
        "split_manifest_sha256": None if isinstance(split, FinalFitMembership) else split.sha256,
        "fit_groups": {"train": list(split.train_groups), "valid": list(split.valid_groups)},
        "evaluation_valid_groups": []
        if isinstance(split, FinalFitMembership)
        else list(split.evaluation_groups),
        "family_order": list(FAMILY_ORDER),
        "source_coefficients": [
            {
                "family": source.family,
                "run": source.run,
                "epoch": source.epoch,
                "coefficient": source.coefficient,
            }
            for source in args.teacher
        ],
        "coefficient_sum": total,
        "homogeneous_family": args.homogeneous_family,
        "output_profile": args.output_profile,
        "stored_targets": list(OUTPUT_TARGETS[args.output_profile]),
        "epoch_convention": EPOCH_CONVENTION,
        "sources": records,
        "files": file_entries,
        "estimated_bytes": estimated_bytes,
    }
    if isinstance(split, FinalFitMembership):
        metadata["final_fit"] = final_fit_identity(split)
        metadata["prediction_role"] = "fitting-only-teacher-targets; no independent evaluation"
    (partial / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(partial, output_path)
    print(
        json.dumps(
            {
                "cache": str(output_path),
                "metadata_sha256": file_sha256(output_path / "metadata.json"),
                "estimated_bytes": estimated_bytes,
                "coefficient_sum": total,
            }
        )
    )
    return output_path


def main() -> None:
    """Parse the frozen cache recipe and build its audited targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-checkpoint",
        choices=("epoch", "selected-fixed"),
        default="epoch",
        help="Bind existing prediction bytes to their actual fixed selected checkpoint; never relabel hashes",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--teacher-cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "artifacts" / "runs")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--final-fit-manifest", type=Path)
    parser.add_argument("--final-fit-source-validation-receipt", type=Path)
    parser.add_argument("--final-fit-source-validation-sha256")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--teacher",
        action="append",
        required=True,
        type=parse_teacher,
        help="family source: FAMILY=RUN[@epN]*COEFFICIENT (repeat for each source)",
    )
    parser.add_argument(
        "--homogeneous-family",
        default="strong",
        choices=FAMILY_ORDER,
        help="family mean scaled by the total coefficient for C1",
    )
    parser.add_argument("--expected-coefficient-sum", type=float, default=0.8125)
    parser.add_argument("--chunk-sequences", type=int, default=8)
    parser.add_argument(
        "--output-profile",
        choices=tuple(OUTPUT_TARGETS),
        default="all",
        help="stored target set; homogeneous-only supports homogeneous-main students without family heads",
    )
    parser.add_argument("--output-budget-gb", type=float, default=30.0)
    parser.add_argument("--min-free-gb", type=float, default=25.0)
    args = parser.parse_args()
    final_fit_selected = args.final_fit_manifest is not None
    receipt_selected = (
        args.final_fit_source_validation_receipt is not None
        or args.final_fit_source_validation_sha256 is not None
    )
    if final_fit_selected != receipt_selected or (
        final_fit_selected
        and (
            args.final_fit_source_validation_receipt is None
            or args.final_fit_source_validation_sha256 is None
        )
    ):
        parser.error("--final-fit-manifest requires source-validation receipt path and SHA256")
    if args.split_manifest is not None and final_fit_selected:
        parser.error("--split-manifest and --final-fit-manifest are mutually exclusive")
    if args.split_manifest is None and not final_fit_selected:
        parser.error("cache build requires --split-manifest or --final-fit-manifest")
    if args.output_budget_gb <= 0 or args.min_free_gb < 0:
        parser.error("disk budgets must be positive (minimum free space may be zero)")
    build_cache(args)


if __name__ == "__main__":
    main()
