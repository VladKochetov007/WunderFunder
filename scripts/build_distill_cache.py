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
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Verify teacher checkpoint bytes, architecture, fit groups and prediction caches."""
    if not RUN_NAME.fullmatch(run):
        raise ValueError(f"invalid teacher run name: {run}")
    run_root = checkpoint_root / run
    provenance_path = run_root / "provenance.json"
    checkpoint_path = run_root / f"model_ep{epoch}.pt"
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
    if (
        checkpoint.get("epoch") != epoch
        or args.get("checkpoint_policy") != "fixed"
        or args.get("checkpoint_epoch") != epoch
        or args.get("split_manifest_sha256") != split.sha256
    ):
        raise ValueError(f"teacher {run} checkpoint is not the requested fixed clean-fold epoch")
    checkpoint_provenance = checkpoint.get("provenance", {})
    _verify_run_membership(checkpoint_provenance, split, run)
    model_state = checkpoint.get("model", {})
    hidden = int(args.get("hidden", 0))
    layers = int(args.get("layers", 0))
    cell = args.get("cell", "gru")
    gate_count = {"gru": 3, "lstm": 4}.get(cell)
    adapter = args.get("model_adapter")
    smoother_marked = "features.arm_code" in model_state
    if adapter is not None or smoother_marked:
        channels_expected = adapter.get("raw_channels") if isinstance(adapter, dict) else None
        arm = adapter.get("arm") if isinstance(adapter, dict) else None
        algorithm = adapter.get("algorithm") if isinstance(adapter, dict) else None
        code = {"fixed": 0, "learned": 1, "ema": 2, "raw": 3}.get(arm, -1)
        feature_width = 4 * len(channels_expected) if isinstance(channels_expected, list) else -1
        expected_algorithm = {
            "raw": "capacity_matched_zero_features",
            "ema": "steady_state_ema",
            "fixed": "fixed_local_trend",
            "learned": "learned_local_trend",
        }.get(arm)
        valid_channels = (
            isinstance(channels_expected, list)
            and bool(channels_expected)
            and all(
                isinstance(channel, int) and 0 <= channel < 112 for channel in channels_expected
            )
            and channels_expected == sorted(set(channels_expected))
        )
        expected_scope_channels = {
            "pilot4": [0, 3, 52, 55],
            "all112": list(range(112)),
        }
        valid_scope = (
            (
                adapter.get("channel_scope") in expected_scope_channels
                and channels_expected == expected_scope_channels.get(adapter.get("channel_scope"))
            )
            if isinstance(adapter, dict)
            else False
        )
        if (
            not isinstance(adapter, dict)
            or adapter.get("kind") != "causal_smoother_gru"
            or adapter.get("version") != 1
            or not valid_channels
            or not valid_scope
            or adapter.get("feature_width") != feature_width
            or algorithm != expected_algorithm
            or adapter.get("covariance_update")
            != ("not_applicable" if arm in ("ema", "raw") else "Joseph form")
            or adapter.get("q_r_tying")
            != (
                "not_applicable"
                if arm in ("ema", "raw")
                else "one scalar per local-trend state family shared across selected channels"
            )
            or checkpoint_provenance.get("model_adapter") != adapter
            or provenance.get("model_adapter") != adapter
            or args.get("arch") != "gru"
            or args.get("cell") != "gru"
            or args.get("enc") != "linear"
        ):
            raise ValueError(f"teacher {run} has incomplete Kalman-adapter provenance")
        arm_code = model_state.get("features.arm_code")
        channels = model_state.get("features.channel_indices")
        if (
            arm_code is None
            or int(arm_code.item()) != code
            or channels is None
            or channels.tolist() != channels_expected
        ):
            raise ValueError(f"teacher {run} causal-smoother buffers disagree with arm/channels")
        if arm == "ema":
            alpha = model_state.get("features.alpha")
            if alpha is None or alpha.numel() != 1 or not torch.isfinite(alpha).all().item():
                raise ValueError(f"teacher {run} EMA gain is missing or nonfinite")
        elif arm == "raw":
            if any(
                name.startswith("features.log_q") or name == "features.log_r"
                for name in model_state
            ):
                raise ValueError(
                    f"teacher {run} zero-feature raw control unexpectedly contains Q/R state"
                )
        elif any(
            model_state.get(f"features.{name}") is None
            or tuple(model_state[f"features.{name}"].shape) != (1,)
            or not torch.isfinite(model_state[f"features.{name}"]).all().item()
            for name in ("log_q_level", "log_q_slope", "log_r")
        ):
            raise ValueError(f"teacher {run} local-trend Q/R state is missing or nonfinite")
        recurrent = model_state.get("core.gru.weight_ih_l0")
        input_projection = model_state.get("core.inp.weight")
        output_projection = model_state.get("core.head.2.weight")
        expected_input_width = int(checkpoint.get("n_in", 0)) + feature_width
        model_mean = model_state.get("mean")
        core_mean = model_state.get("core.mean")
        actual_layers = sum(
            1 for key in model_state if re.fullmatch(r"core\.gru\.weight_ih_l\d+", key)
        )
        if (
            model_mean is None
            or tuple(model_mean.shape) != (int(checkpoint.get("n_in", 0)),)
            or core_mean is None
            or tuple(core_mean.shape) != (expected_input_width,)
            or input_projection is None
            or input_projection.shape[1] != expected_input_width
            or actual_layers != layers
        ):
            raise ValueError(f"teacher {run} causal-smoother-GRU input/core dimensions disagree")
    else:
        recurrent = model_state.get("gru.weight_ih_l0")
        input_projection = model_state.get("inp.weight")
        output_projection = model_state.get("head.2.weight")
        actual_layers = sum(1 for key in model_state if re.fullmatch(r"gru\.weight_ih_l\d+", key))
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
            or sidecar.get("split_manifest_sha256") != split.sha256
            or sidecar.get("checkpoint_sha256") != checkpoint_sha256
        ):
            raise ValueError(
                f"teacher {run} {split_name} prediction sidecar is not bound to its checkpoint"
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
        "model_adapter": adapter,
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
    """Create the declared fold-bound distillation targets without array copies."""
    data_root = args.data_root.resolve()
    cache_root = args.teacher_cache_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
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

    target_orders = {name: _target_order(data_root, name) for name in ("train", "valid")}
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
    for source in args.teacher:
        if source.family not in FAMILY_ORDER:
            raise ValueError(f"unknown teacher family: {source.family}")
        ancestor_closure = _audit_ancestor_closure(
            source.run, split, checkpoint_root, f"model_ep{source.epoch}.pt"
        )
        record, arrays, _ = _inspect_teacher(
            source.run, split, cache_root, checkpoint_root, target_orders, source.epoch
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
        "fold": split.fold,
        "split_manifest": str(split.path),
        "split_manifest_sha256": split.sha256,
        "fit_groups": {"train": list(split.train_groups), "valid": list(split.valid_groups)},
        "evaluation_valid_groups": list(split.evaluation_groups),
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
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--teacher-cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "artifacts" / "runs")
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
