"""Build a fold-bound source-scale and teacher-target cache."""

from __future__ import annotations

import argparse
import hashlib
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

from wnn.cleanfold_t3_cache import (
    T3_HOMOGENEOUS_SCALE,
    T3_TEACHER_ORDER,
    combine_cleanfold_t3_chunk,
    fit_prediction_positions,
)
from wnn.distillation import TeacherSource, combine_teacher_chunk
from wnn.distillation_cache import (
    CACHE_FORMAT,
    CACHE_VERSION,
    FAMILY_ORDER,
    SEQUENCE_ROWS,
    file_sha256,
    row_groups_sha256,
)
from wnn.splits import CampaignSplit, load_campaign_split

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
}
TARGET_WIDTHS = {"main_source": 2, "main_normalized": 2, "main_homogeneous": 2, "families": 8}
CLEANFOLD_T3_PROFILE = "cleanfold_homogeneous_strong_t3_v1"
CLEANFOLD_T3_CONTRACT_SCHEMA = "fg19.teacher-cache-execution-contract.v1"
CLEANFOLD_T3_CONTRACT_LIMIT = 128 * 1024
CLEANFOLD_T3_PROVENANCE_LIMIT = 1_000_000
SHA256_TEXT = re.compile(r"[0-9a-f]{64}\Z")


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


def _expected_groups(split: CampaignSplit, name: str) -> tuple[int, ...]:
    return split.train_groups if name == "train" else split.valid_groups


def _ordered_fit_groups(
    target_groups: list[int] | tuple[int, ...], expected_fit_group_set: frozenset[int]
) -> tuple[int, ...]:
    """Keep fitting groups in target-index order using one prebuilt membership set."""
    return tuple(group for group in target_groups if group in expected_fit_group_set)


def _fit_sidecar_groups(sidecar: dict[str, Any], run: str, split_name: str) -> list[int]:
    """Read exact JSON integer row-group identities for the opt-in fit-only path."""
    row_groups = sidecar.get("row_groups")
    if not isinstance(row_groups, list) or any(type(group) is not int for group in row_groups):
        raise ValueError(f"teacher {run} {split_name} sidecar row_groups must be JSON integers")
    return row_groups


def _target_order(data_root: Path, split_name: str) -> list[int]:
    path = data_root / "cache" / f"{split_name}_targets.npz"
    with np.load(path, allow_pickle=False) as targets:
        groups = np.asarray(targets["row_group"], dtype=np.int64)
    if len(np.unique(groups)) != len(groups):
        raise ValueError(f"target index repeats a row group: {path}")
    return list(map(int, groups))


def _audit_ancestor_closure(
    run: str,
    split: CampaignSplit,
    checkpoint_root: Path,
    checkpoint_file: str = "model.pt",
    expected_checkpoint_sha256: str | None = None,
    seen: set[str] | None = None,
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

    checkpoint_path = run_root / checkpoint_file
    checkpoint_sha256 = file_sha256(checkpoint_path)
    if expected_checkpoint_sha256 and checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(f"ancestor {run} checkpoint bytes disagree with its recorded hash")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    if (
        checkpoint.get("epoch") != checkpoint_args.get("checkpoint_epoch")
        or checkpoint_args.get("checkpoint_policy") != "fixed"
        or not checkpoint_args.get("no_eval_during_fit")
        or checkpoint_args.get("split_manifest_sha256") != split.sha256
    ):
        raise ValueError(f"ancestor {run} is not a fixed clean-fold checkpoint")
    checkpoint_provenance = checkpoint.get("provenance", {})
    if checkpoint_provenance.get("fit_row_groups") != provenance.get("fit_row_groups"):
        raise ValueError(f"ancestor {run} checkpoint membership differs from run provenance")
    _verify_run_membership(checkpoint_provenance, split, run)

    closure = [
        {
            "run": run,
            "provenance_sha256": file_sha256(provenance_path),
            "checkpoint_file": checkpoint_file,
            "checkpoint_sha256": checkpoint_sha256,
            "split_manifest_sha256": split.sha256,
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
            )
        )
    return closure


def _verify_run_membership(
    provenance: dict[str, Any], split: CampaignSplit, run: str
) -> dict[str, int]:
    """Require source fitting groups to stay inside the clean fold's fit pool."""
    if provenance.get("split_manifest_sha256") != split.sha256:
        raise ValueError(f"teacher {run} uses a different split manifest")
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
    split: CampaignSplit,
    cache_root: Path,
    checkpoint_root: Path,
    target_orders: dict[str, list[int]],
    epoch: int,
    *,
    fit_only: bool = False,
    root_pin: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Verify teacher identities, fit groups and prediction caches."""
    if not RUN_NAME.fullmatch(run):
        raise ValueError(f"invalid teacher run name: {run}")
    run_root = checkpoint_root / run
    provenance_path = run_root / "provenance.json"
    checkpoint_path = run_root / f"model_ep{epoch}.pt"
    if not provenance_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"teacher {run} lacks provenance or model_ep{epoch}.pt")
    if root_pin is None:
        provenance = json.loads(provenance_path.read_text())
    else:
        provenance_size = provenance_path.stat().st_size
        if provenance_size <= 0 or provenance_size > CLEANFOLD_T3_PROVENANCE_LIMIT:
            raise ValueError(f"teacher {run} provenance exceeds its bounded read size")
        provenance_raw = provenance_path.read_bytes()
        if (
            len(provenance_raw) != provenance_size
            or hashlib.sha256(provenance_raw).hexdigest() != root_pin["provenance_sha256"]
        ):
            raise ValueError(f"teacher {run} provenance bytes changed after pin verification")
        provenance = json.loads(provenance_raw)
    fit_counts = _verify_run_membership(provenance, split, run)
    checkpoint_sha256 = (
        file_sha256(checkpoint_path) if root_pin is None else root_pin["checkpoint_sha256"]
    )
    expected_checkpoint_sha256 = provenance.get("checkpoint_sha256", {}).get(checkpoint_path.name)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(f"teacher {run} checkpoint bytes disagree with provenance")

    if root_pin is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = checkpoint.get("args", {})
        if (
            checkpoint.get("epoch") != epoch
            or args.get("checkpoint_policy") != "fixed"
            or args.get("checkpoint_epoch") != epoch
            or args.get("split_manifest_sha256") != split.sha256
        ):
            raise ValueError(
                f"teacher {run} checkpoint is not the requested fixed clean-fold epoch"
            )
        checkpoint_provenance = checkpoint.get("provenance", {})
        _verify_run_membership(checkpoint_provenance, split, run)
        model_state = checkpoint.get("model", {})
    else:
        args = provenance.get("args", {})
        if not isinstance(args, dict):
            raise ValueError(f"teacher {run} provenance has invalid training arguments")
        expected_seed = T3_TEACHER_ORDER.index(run) + 3
        if (
            int(args.get("hidden", 0)) != 384
            or int(args.get("layers", 0)) != 2
            or args.get("cell") != "gru"
            or int(args.get("seed", -1)) != expected_seed
            or args.get("split_manifest_sha256") != split.sha256
        ):
            raise ValueError(f"teacher {run} provenance differs from its root-pinned T3 recipe")
        model_hashes = provenance.get("checkpoint_sha256", {})
        if not isinstance(model_hashes, dict) or (
            model_hashes.get(checkpoint_path.name) != root_pin["checkpoint_sha256"]
        ):
            raise ValueError(f"teacher {run} provenance does not bind its selected checkpoint")
        if model_hashes.get("model.pt") != root_pin["model_sha256"]:
            raise ValueError(f"teacher {run} provenance does not bind its selected model")
        model_state = {}
    hidden = int(args.get("hidden", 0))
    layers = int(args.get("layers", 0))
    cell = args.get("cell", "gru")
    if root_pin is None:
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
        if fit_only and (
            sidecar_path.stat().st_size <= 0
            or sidecar_path.stat().st_size > CLEANFOLD_T3_CONTRACT_LIMIT
        ):
            raise ValueError(
                f"teacher {run} {split_name} prediction sidecar exceeds its read bound"
            )
        sidecar = json.loads(sidecar_path.read_text())
        if (
            sidecar.get("run") != run
            or sidecar.get("split") != split_name
            or sidecar.get("epoch") != epoch
            or sidecar.get("split_manifest_sha256") != split.sha256
            or sidecar.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise ValueError(
                f"teacher {run} {split_name} prediction sidecar is not bound to its checkpoint"
            )
        if fit_only and (
            sidecar.get("campaign_split_manifest_sha256") != split.sha256
            or sidecar.get("campaign_export_scope") != "fit"
            or sidecar.get("feature_only") is not True
        ):
            raise ValueError(f"teacher {run} {split_name} export is not explicitly fit-only")
        row_groups = (
            _fit_sidecar_groups(sidecar, run, split_name)
            if fit_only
            else list(map(int, sidecar.get("row_groups", [])))
        )
        if fit_only:
            expected_fit_groups = _expected_groups(split, split_name)
            expected_fit_group_set = frozenset(expected_fit_groups)
            canonical_fit_order, _ = fit_prediction_positions(
                row_groups, expected_order, expected_fit_groups
            )
            expected_canonical_order = _ordered_fit_groups(expected_order, expected_fit_group_set)
            if canonical_fit_order != expected_canonical_order:
                raise ValueError(f"teacher {run} {split_name} canonical fit order differs")
        elif row_groups != expected_order:
            raise ValueError(
                f"teacher {run} {split_name} prediction row order differs from target index"
            )
        prediction_sha256 = file_sha256(pred_path)
        if prediction_sha256 != sidecar.get("prediction_sha256"):
            raise ValueError(f"teacher {run} {split_name} prediction bytes disagree with sidecar")
        values = np.load(pred_path, mmap_mode="r", allow_pickle=False)
        if values.shape != (len(row_groups), SEQUENCE_ROWS, 2) or values.dtype != np.float32:
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
        if fit_only:
            cache_records[split_name]["_row_groups"] = row_groups
    record = {
        "run": run,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "provenance_sha256": (
            file_sha256(provenance_path) if root_pin is None else root_pin["provenance_sha256"]
        ),
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
    if root_pin is not None:
        record["model_sha256"] = root_pin["model_sha256"]
        record["acceptance_sha256"] = root_pin["acceptance_sha256"]
        record["tensor_fingerprint"] = root_pin["tensor_fingerprint"]
    return record, predictions, cache_records


def _cleanfold_t3_pins(
    contract_path: Path,
    expected_contract_sha256: str,
    acceptance_root: Path,
    checkpoint_root: Path,
    split: CampaignSplit,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify root-pinned teacher and acceptance identities for the T3 profile."""
    if not isinstance(expected_contract_sha256, str) or not SHA256_TEXT.fullmatch(
        expected_contract_sha256
    ):
        raise ValueError("cleanfold T3 execution-contract SHA-256 is malformed")
    contract_size = contract_path.stat().st_size
    if contract_size <= 0 or contract_size > CLEANFOLD_T3_CONTRACT_LIMIT:
        raise ValueError("cleanfold T3 execution contract exceeds its bounded read size")
    raw = contract_path.read_bytes()
    if len(raw) != contract_size or file_sha256(contract_path) != expected_contract_sha256:
        raise ValueError("cleanfold T3 execution contract bytes disagree with the root pin")
    contract = json.loads(raw)
    if not isinstance(contract, dict):
        raise ValueError("cleanfold T3 execution contract must be a JSON object")
    if (
        contract.get("schema") != CLEANFOLD_T3_CONTRACT_SCHEMA
        or contract.get("source_teacher_commit") != "af21d6d8409c46b36830f0bdddd50e67ee17025f"
        or contract.get("split_sha256") != split.sha256
        or contract.get("teacher_order") != [3, 4, 5]
    ):
        raise ValueError("cleanfold T3 execution contract is bound to a different source or split")

    arithmetic = contract.get("arithmetic", {})
    if not isinstance(arithmetic, dict):
        raise ValueError("cleanfold T3 arithmetic contract is invalid")
    if (
        arithmetic.get("mean")
        != "Unweighted arithmetic mean of S3,S4,S5 in FP32, fixed source order."
        or arithmetic.get("scale") != 0.8125
        or arithmetic.get("scaling_count") != 1
        or arithmetic.get("cache_clipping") != "none"
        or arithmetic.get("consumer_cache_scale") != 1.0
    ):
        raise ValueError("cleanfold T3 arithmetic differs from its root-bound contract")
    membership = contract.get("membership", {})
    if not isinstance(membership, dict):
        raise ValueError("cleanfold T3 membership contract is invalid")
    if (
        membership.get("train") != len(split.train_groups)
        or membership.get("valid") != len(split.valid_groups)
        or membership.get("evaluation_inferred") != 0
    ):
        raise ValueError("cleanfold T3 membership differs from its source-qualified split")

    teachers = contract.get("teachers")
    if not isinstance(teachers, list) or len(teachers) != 3:
        raise ValueError("cleanfold T3 contract must pin exactly three teachers")
    accepted: list[dict[str, Any]] = []
    for index, (teacher, run) in enumerate(zip(teachers, T3_TEACHER_ORDER, strict=True)):
        if not isinstance(teacher, dict):
            raise ValueError("cleanfold T3 teacher pin is invalid")
        expected_seed = index + 3
        if (
            teacher.get("run") != run
            or teacher.get("seed") != expected_seed
            or teacher.get("epoch") != 2
        ):
            raise ValueError("cleanfold T3 teachers must be ordered S3/S4/S5 at fixed epoch 2")
        pin_keys = (
            "acceptance_sha256",
            "model_sha256",
            "checkpoint_sha256",
            "provenance_sha256",
            "tensor_fingerprint",
        )
        if any(
            not isinstance(teacher.get(key), str) or not SHA256_TEXT.fullmatch(teacher[key])
            for key in pin_keys
        ):
            raise ValueError(f"cleanfold T3 identity pin is malformed: {run}")
        acceptance_relative = Path(str(teacher.get("acceptance_path", "")))
        if acceptance_relative.is_absolute() or ".." in acceptance_relative.parts:
            raise ValueError(f"cleanfold T3 acceptance path is unsafe: {run}")
        acceptance_path = (acceptance_root / acceptance_relative).resolve()
        if not acceptance_path.is_relative_to(acceptance_root.resolve()):
            raise ValueError(f"cleanfold T3 acceptance path escapes its root: {run}")
        if file_sha256(acceptance_path) != teacher["acceptance_sha256"]:
            raise ValueError(f"cleanfold T3 terminal acceptance bytes disagree: {run}")
        run_root = checkpoint_root / run
        actual_files = {
            "model_sha256": run_root / "model.pt",
            "checkpoint_sha256": run_root / "model_ep2.pt",
        }
        if any(file_sha256(path) != teacher[key] for key, path in actual_files.items()):
            raise ValueError(f"cleanfold T3 checkpoint or provenance bytes disagree: {run}")
        provenance_path = run_root / "provenance.json"
        provenance_size = provenance_path.stat().st_size
        if provenance_size <= 0 or provenance_size > CLEANFOLD_T3_PROVENANCE_LIMIT:
            raise ValueError(f"teacher {run} provenance exceeds its bounded read size")
        provenance_raw = provenance_path.read_bytes()
        if (
            len(provenance_raw) != provenance_size
            or hashlib.sha256(provenance_raw).hexdigest() != teacher["provenance_sha256"]
        ):
            raise ValueError(f"cleanfold T3 checkpoint or provenance bytes disagree: {run}")
        accepted.append(
            {
                "run": run,
                "acceptance_path": str(acceptance_relative),
                **{key: teacher[key] for key in pin_keys},
            }
        )
    return accepted, contract


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
    """Create the declared fold-bound distillation targets without array copies."""
    data_root = args.data_root.resolve()
    cache_root = args.teacher_cache_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    split = load_campaign_split(args.split_manifest.resolve(), data_root)
    cache_profile = getattr(args, "cache_profile", "legacy-four-family")
    cleanfold_t3 = cache_profile == CLEANFOLD_T3_PROFILE
    if cache_profile not in ("legacy-four-family", CLEANFOLD_T3_PROFILE):
        raise ValueError(f"unknown distillation cache profile: {cache_profile}")
    if any(source.epoch < 0 for source in args.teacher):
        raise ValueError("teacher epochs must be nonnegative zero-based indices")
    if len({source.run for source in args.teacher}) != len(args.teacher):
        raise ValueError("each source run may appear only once")
    total = math.fsum(source.coefficient for source in args.teacher)
    acceptance_pins: list[dict[str, Any]] = []
    execution_contract: dict[str, Any] = {}
    if cleanfold_t3:
        expected_sources = [("strong", run, 0.1, 2) for run in T3_TEACHER_ORDER]
        actual_sources = [
            (source.family, source.run, source.coefficient, source.epoch) for source in args.teacher
        ]
        if actual_sources != expected_sources:
            raise ValueError(
                "cleanfold T3 sources must be ordered strong S3/S4/S5 at 0.1 each, epoch 2"
            )
        if args.output_profile != "homogeneous-only" or args.homogeneous_family != "strong":
            raise ValueError("cleanfold T3 requires the strong homogeneous-only output profile")
        if abs(total - 0.3) > 1e-7:
            raise ValueError(
                "cleanfold T3 metadata must preserve the original 0.3 coefficient subtotal"
            )
        if not getattr(args, "cleanfold_contract", None) or not getattr(
            args, "acceptance_root", None
        ):
            raise ValueError("cleanfold T3 requires a root-pinned contract and acceptance root")
        if abs(args.expected_coefficient_sum - 0.8125) > 1e-7:
            raise ValueError("the T3 homogeneous multiplier is fixed at 0.8125")
    else:
        if abs(args.expected_coefficient_sum - 0.8125) > 1e-7:
            raise ValueError("source-scale cache must retain the declared 0.8125 coefficient sum")
        if len(args.teacher) < 4 or set(source.family for source in args.teacher) != set(
            FAMILY_ORDER
        ):
            raise ValueError("teacher sources must cover the four declared families")
        if abs(total - args.expected_coefficient_sum) > 1e-7:
            raise ValueError(f"teacher coefficients sum to {total}, expected 0.8125")
        if args.homogeneous_family not in FAMILY_ORDER:
            raise ValueError("homogeneous family must be one of the declared teacher families")
    if args.output_profile not in OUTPUT_TARGETS:
        raise ValueError(f"unknown distillation cache output profile: {args.output_profile}")
    if args.chunk_sequences < 1 or args.chunk_sequences > 64:
        raise ValueError("chunk sequence count must be between 1 and 64")

    target_orders = {name: _target_order(data_root, name) for name in ("train", "valid")}
    expected_fit_groups = {name: _expected_groups(split, name) for name in target_orders}
    expected_fit_group_sets = {
        name: frozenset(groups) for name, groups in expected_fit_groups.items()
    }
    row_groups = {
        name: list(_ordered_fit_groups(target_orders[name], expected_fit_group_sets[name]))
        for name in ("train", "valid")
    }
    for name in ("train", "valid"):
        if set(row_groups[name]) != expected_fit_group_sets[name]:
            raise ValueError(f"target index omits {name} fitting groups")

    fit_rows = sum(len(groups) for groups in row_groups.values())
    if cleanfold_t3 and os.path.lexists(args.output):
        raise FileExistsError(
            f"refusing to replace existing cache path (including symlinks): {args.output}"
        )
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
    if cleanfold_t3:
        acceptance_pins, execution_contract = _cleanfold_t3_pins(
            args.cleanfold_contract.resolve(),
            args.cleanfold_contract_sha256,
            args.acceptance_root.resolve(),
            checkpoint_root,
            split,
        )

    records: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    prediction_records: dict[str, dict[str, dict[str, Any]]] = {}
    for source in args.teacher:
        if not cleanfold_t3 and source.family not in FAMILY_ORDER:
            raise ValueError(f"unknown teacher family: {source.family}")
        pin = acceptance_pins[len(records)] if cleanfold_t3 else None
        ancestor_closure = (
            []
            if cleanfold_t3
            else _audit_ancestor_closure(
                source.run, split, checkpoint_root, f"model_ep{source.epoch}.pt"
            )
        )
        if cleanfold_t3:
            record, arrays, _ = _inspect_teacher(
                source.run,
                split,
                cache_root,
                checkpoint_root,
                target_orders,
                source.epoch,
                fit_only=True,
                root_pin=pin,
            )
        else:
            record, arrays, _ = _inspect_teacher(
                source.run, split, cache_root, checkpoint_root, target_orders, source.epoch
            )
        record["family"] = source.family
        record["coefficient"] = source.coefficient
        if not cleanfold_t3:
            record["ancestor_closure"] = ancestor_closure
        if cleanfold_t3:
            record.update(
                {
                    "acceptance_sha256": pin["acceptance_sha256"],
                    "model_sha256": pin["model_sha256"],
                    "tensor_fingerprint": pin["tensor_fingerprint"],
                }
            )
        records.append(record)
        predictions[source.run] = arrays
        if cleanfold_t3:
            prediction_records[source.run] = record["predictions"]

    prediction_positions: dict[str, dict[str, np.ndarray]] = {}
    if cleanfold_t3:
        for source in args.teacher:
            prediction_positions[source.run] = {}
            for split_name in ("train", "valid"):
                sidecar_groups = prediction_records[source.run][split_name].pop("_row_groups")
                fit_order, positions = fit_prediction_positions(
                    sidecar_groups, target_orders[split_name], expected_fit_groups[split_name]
                )
                if tuple(row_groups[split_name]) != fit_order:
                    raise ValueError(f"teacher {source.run} {split_name} cache order differs")
                prediction_positions[source.run][split_name] = positions

    partial.mkdir()
    file_entries: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "valid"):
        groups = row_groups[split_name]
        position = {group: index for index, group in enumerate(target_orders[split_name])}
        shapes = {
            target: (len(groups), SEQUENCE_ROWS, TARGET_WIDTHS[target])
            for target in OUTPUT_TARGETS[args.output_profile]
        }
        outputs = {
            name: np.lib.format.open_memmap(
                partial / f"{name}_{split_name}.npy", mode="w+", dtype=np.float32, shape=shape
            )
            for name, shape in shapes.items()
        }
        for start in range(0, len(groups), args.chunk_sequences):
            selected = groups[start : start + args.chunk_sequences]
            rows = np.asarray([position[group] for group in selected], dtype=np.int64)
            stop = start + len(selected)
            if cleanfold_t3:
                source_chunks = [
                    np.asarray(
                        predictions[source.run][split_name][
                            prediction_positions[source.run][split_name][start:stop]
                        ],
                        dtype=np.float32,
                    )
                    for source in args.teacher
                ]
                outputs["main_homogeneous"][start:stop] = combine_cleanfold_t3_chunk(source_chunks)
            else:
                source_chunk = {
                    run: np.asarray(arrays[split_name][rows], dtype=np.float32)
                    for run, arrays in predictions.items()
                }
                main, normalized, homogeneous, families = combine_teacher_chunk(
                    source_chunk, args.teacher, FAMILY_ORDER, args.homogeneous_family
                )
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
        "fold": split.fold,
        "split_manifest": str(split.path),
        "split_manifest_sha256": split.sha256,
        "fit_groups": {"train": list(split.train_groups), "valid": list(split.valid_groups)},
        "evaluation_valid_groups": list(split.evaluation_groups),
        "family_order": ["strong"] if cleanfold_t3 else list(FAMILY_ORDER),
        "source_coefficients": [
            {
                "family": source.family,
                "run": source.run,
                "epoch": source.epoch,
                "coefficient": source.coefficient,
            }
            for source in args.teacher
        ],
        "coefficient_sum": 0.3 if cleanfold_t3 else total,
        "homogeneous_family": args.homogeneous_family,
        "output_profile": args.output_profile,
        "stored_targets": list(OUTPUT_TARGETS[args.output_profile]),
        "epoch_convention": EPOCH_CONVENTION,
        "sources": records,
        "files": file_entries,
        "estimated_bytes": estimated_bytes,
    }
    if cleanfold_t3:
        metadata.update(
            {
                "cache_profile": CLEANFOLD_T3_PROFILE,
                "homogeneous_scale": float(T3_HOMOGENEOUS_SCALE),
                "homogeneous_family": "strong",
                "stored_targets": ["main_homogeneous"],
                "target_contract": {
                    "formula": "float32(0.8125 * float32_mean(three_epoch2_predictions))",
                    "teacher_epoch": 2,
                    "teacher_mean": "unweighted arithmetic mean in float32",
                    "clipping": "none during cache construction",
                    "coefficient_multiplier": 0.8125,
                },
                "student_contract": {
                    "epochs": 6,
                    "checkpoint_policy": "fixed",
                    "checkpoint_epoch": 5,
                    "hidden": 256,
                    "layers": 2,
                    "batch": 16,
                    "chunk": 2000,
                    "distill_alpha": 0.9,
                    "since_weight": 0.0,
                    "inherit_init_preprocessing": False,
                },
                "root_execution_contract": {
                    "path": str(args.cleanfold_contract.resolve()),
                    "sha256": args.cleanfold_contract_sha256,
                    "schema": execution_contract["schema"],
                },
                "root_acceptance_pins": acceptance_pins,
            }
        )
    (partial / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(partial, output_path)
    print(
        json.dumps(
            {
                "cache": str(output_path),
                "metadata_sha256": file_sha256(output_path / "metadata.json"),
                "estimated_bytes": estimated_bytes,
                "coefficient_sum": 0.3 if cleanfold_t3 else total,
            }
        )
    )
    return output_path


def main() -> None:
    """Parse the frozen cache recipe and build its audited targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--teacher-cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "artifacts" / "runs")
    parser.add_argument(
        "--cache-profile",
        choices=("legacy-four-family", CLEANFOLD_T3_PROFILE),
        default="legacy-four-family",
        help="cache semantics; the opt-in cleanfold T3 profile is source- and acceptance-pinned",
    )
    parser.add_argument("--cleanfold-contract", type=Path)
    parser.add_argument("--cleanfold-contract-sha256")
    parser.add_argument(
        "--acceptance-root",
        type=Path,
        help="root under which the pinned contract's relative records/... paths resolve",
    )
    parser.add_argument("--split-manifest", required=True, type=Path)
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
    if args.output_budget_gb <= 0 or args.min_free_gb < 0:
        parser.error("disk budgets must be positive (minimum free space may be zero)")
    build_cache(args)


if __name__ == "__main__":
    main()
