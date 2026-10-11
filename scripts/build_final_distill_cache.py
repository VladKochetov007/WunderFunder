"""Build H390's two-array all-data homogeneous teacher target cache."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from wnn.final_distillation_cache import (
    EXPECTED_PROPENSITY_SHA256,
    EXPECTED_SPLIT_SHA256,
    FINAL_CACHE_FORMAT,
    FINAL_CACHE_VERSION,
    PARENTS,
    SEQUENCE_ROWS,
    TARGET_MULTIPLIER,
    TARGET_WIDTH,
    TEACHERS,
    cache_disk_requirement,
    file_sha256,
    homogeneous_teacher_target,
    row_groups_sha256,
    source_qualified_groups_sha256,
    target_row_groups,
    validate_final_distillation_cache,
    verify_h390_source_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
TARGET_CONTRACT = {
    "family": "strong",
    "formula": "float32(0.8125 * float32_mean(three_epoch2_predictions))",
    "coefficient_multiplier": TARGET_MULTIPLIER,
    "teacher_mean": "unweighted arithmetic mean in float32",
    "teacher_epoch": 2,
    "clipping": "none during cache construction",
}
TEACHER_ARGS = {
    "arch": "gru",
    "cell": "gru",
    "hidden": 384,
    "layers": 2,
    "optim": "muon",
    "muon_scope": "all",
    "lr": 0.001,
    "wd": 0.01,
    "epochs": 3,
    "checkpoint_policy": "fixed",
    "checkpoint_epoch": 2,
    "no_eval_during_fit": True,
    "batch": 32,
    "chunk": 1000,
    "sched": "onecycle",
    "precision": "fp16",
    "aux_struct": 0.3,
    "aux_next": 0.3,
    "propensity": "c20260929_prop_f1",
    "prop_power": 2.0,
    "prop_eps": 0.03,
    "valid_true_mask": 0.03,
    "since_weight": 32.0,
    "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
}
PARENT_ARGS = {
    "arch": "gru",
    "cell": "gru",
    "hidden": 256,
    "layers": 2,
    "optim": "muon",
    "muon_scope": "all",
    "lr": 0.001,
    "wd": 0.01,
    "epochs": 4,
    "checkpoint_policy": "fixed",
    "checkpoint_epoch": 3,
    "no_eval_during_fit": True,
    "train_source": "train+valid",
    "train_seqs": 10607,
    "valid_seqs": 600,
    "valid_fold": -1,
    "include_heldout_valid": True,
    "valid_train_fraction": 1.0,
    "valid_repeat": 1,
    "batch": 32,
    "chunk": 1000,
    "loss": "wmse",
    "loss_norm": "chunk",
    "sched": "onecycle",
    "precision": "fp16",
    "propensity": "c20260929_prop_f1",
    "prop_power": 0.0,
    "prop_eps": 0.03,
    "valid_true_mask": 0.03,
    "since_weight": 32.0,
    "aux_struct": 0.3,
    "aux_next": 0.3,
}


def _same_value(actual: object, expected: object) -> bool:
    """Compare JSON recipe fields without treating bool as an integer."""
    if isinstance(expected, float):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and abs(float(actual) - expected) <= 1e-12
        )
    return actual == expected and type(actual) is type(expected)


def _target_index(data_root: Path, split: str) -> list[int]:
    groups = target_row_groups(data_root, split)
    expected_count = 10_606 if split == "train" else 1_873
    if len(groups) != expected_count:
        raise ValueError(f"{split} target index has {len(groups)} rows, expected {expected_count}")
    return groups


def _teacher_provenance_contract(run: str, provenance_path: Path) -> tuple[dict[str, Any], str]:
    """Verify the pinned actual producer receipt before checkpoint/export access."""
    spec = TEACHERS[run]
    expected_sha = spec.get("provenance_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(f"teacher {run} has no reviewed producer provenance hash")
    if not provenance_path.is_file():
        raise FileNotFoundError(f"teacher {run} lacks its producer provenance receipt")
    provenance_sha = file_sha256(provenance_path)
    if provenance_sha != expected_sha:
        raise ValueError(f"teacher {run} provenance differs from the reviewed producer receipt")
    provenance = json.loads(provenance_path.read_text())
    args = provenance.get("args")
    if not isinstance(args, dict) or args.get("name") != run or args.get("seed") != spec["seed"]:
        raise ValueError(f"teacher identity differs from H390: {run}")
    for key, expected in TEACHER_ARGS.items():
        if not _same_value(args.get(key), expected):
            raise ValueError(f"teacher {run} has unexpected {key}: {args.get(key)!r}")
    if provenance.get("checkpoint_sha256", {}).get("model_ep2.pt") != spec["checkpoint_sha256"]:
        raise ValueError(f"teacher {run} receipt does not bind its reviewed epoch-2 checkpoint")
    if provenance.get("propensity_checkpoint_sha256") != EXPECTED_PROPENSITY_SHA256:
        raise ValueError(f"teacher {run} receipt binds different propensity weights")
    return provenance, provenance_sha


def _teacher_contract(
    run: str,
    checkpoint_root: Path,
    cache_root: Path,
    target_orders: dict[str, list[int]],
    split_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Verify one actual teacher checkpoint, its recipe, and both full exports."""
    spec = TEACHERS[run]
    run_root = checkpoint_root / run
    provenance_path = run_root / "provenance.json"
    checkpoint_path = run_root / "model_ep2.pt"
    provenance, provenance_sha = _teacher_provenance_contract(run, provenance_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"teacher {run} lacks model_ep2.pt")
    checkpoint_sha = file_sha256(checkpoint_path)
    if checkpoint_sha != spec["checkpoint_sha256"]:
        raise ValueError(f"teacher {run} checkpoint bytes differ from the reviewed producer")
    if provenance.get("checkpoint_sha256", {}).get(checkpoint_path.name) != checkpoint_sha:
        raise ValueError(f"teacher {run} provenance does not bind model_ep2.pt")
    if provenance.get("propensity_checkpoint_sha256") != EXPECTED_PROPENSITY_SHA256:
        raise ValueError(f"teacher {run} used different propensity weights")
    fit = provenance.get("fit_row_groups")
    if not isinstance(fit, dict):
        raise ValueError(f"teacher {run} has no fitting membership")
    expected_fit = split_manifest.get("fit", {})
    expected_eval = set(map(int, split_manifest.get("evaluation", {}).get("valid", [])))
    fitted = {name: set(map(int, fit.get(name, []))) for name in ("train", "valid")}
    expected = {
        "train": set(map(int, expected_fit.get("train", []))),
        "valid": set(map(int, expected_fit.get("valid", []))),
    }
    if fitted != expected:
        raise ValueError(f"teacher {run} membership differs from clean fold-1 fit pools")
    if fitted["valid"] & expected_eval:
        raise ValueError(f"teacher {run} fitted a fold-1 evaluation sequence")

    arrays: dict[str, np.ndarray] = {}
    prediction_records: dict[str, dict[str, Any]] = {}
    for source_split, row_groups in target_orders.items():
        prediction_path = cache_root / f"pred_{run}_{source_split}.npy"
        sidecar_path = prediction_path.with_suffix(".json")
        if not prediction_path.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(f"teacher {run} lacks its full {source_split} export")
        sidecar = json.loads(sidecar_path.read_text())
        if (
            sidecar.get("run") != run
            or sidecar.get("split") != source_split
            or sidecar.get("epoch") != 2
            or sidecar.get("checkpoint_sha256") != checkpoint_sha
            or sidecar.get("split_manifest_sha256") != EXPECTED_SPLIT_SHA256
            or list(map(int, sidecar.get("row_groups", []))) != row_groups
        ):
            raise ValueError(f"teacher {run} {source_split} sidecar has different lineage or rows")
        prediction_sha = file_sha256(prediction_path)
        if sidecar.get("prediction_sha256") != prediction_sha:
            raise ValueError(f"teacher {run} {source_split} export hash differs from its sidecar")
        array = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        if array.shape != (len(row_groups), SEQUENCE_ROWS, 2) or array.dtype != np.float32:
            raise ValueError(f"teacher {run} {source_split} export shape or dtype differs")
        arrays[source_split] = array
        prediction_records[source_split] = {
            "prediction_sha256": prediction_sha,
            "prediction_sidecar_sha256": file_sha256(sidecar_path),
            "rows": len(row_groups),
            "row_groups_sha256": row_groups_sha256(row_groups),
        }
    return (
        {
            "run": run,
            "seed": spec["seed"],
            "epoch": 2,
            "checkpoint_sha256": checkpoint_sha,
            "propensity_checkpoint_sha256": EXPECTED_PROPENSITY_SHA256,
            "provenance_sha256": file_sha256(provenance_path),
            "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
            "fit_membership": {
                name: {
                    "count": len(fitted[name]),
                    "row_groups_sha256": row_groups_sha256(sorted(fitted[name])),
                }
                for name in ("train", "valid")
            },
            "evaluation_annotation": {
                "count": len(expected_eval),
                "classification": "teacher clean-fold evaluation pool excluded from teacher fit",
            },
            "predictions": prediction_records,
        },
        arrays,
    )


def _parent_contract(
    run: str,
    checkpoint_root: Path,
    target_orders: dict[str, list[int]],
) -> dict[str, Any]:
    """Bind one all-data initialization parent and label its legacy eval rows."""
    seed = 3 if run.endswith("_s3") else 4
    spec = PARENTS[seed]
    run_root = checkpoint_root / run
    provenance_path = run_root / "provenance.json"
    checkpoint_path = run_root / "model_ep3.pt"
    if run != spec["run"] or not provenance_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"student parent {run} does not match the H390 seed map")
    provenance = json.loads(provenance_path.read_text())
    provenance_sha = file_sha256(provenance_path)
    if provenance_sha != spec["provenance_sha256"]:
        raise ValueError(
            f"student parent {run} provenance differs from the reviewed producer receipt"
        )
    args = provenance.get("args")
    if not isinstance(args, dict) or args.get("name") != run or args.get("seed") != seed:
        raise ValueError(f"student parent identity differs from H390: {run}")
    for key, expected in PARENT_ARGS.items():
        if not _same_value(args.get(key), expected):
            raise ValueError(f"student parent {run} has unexpected {key}")
    if args.get("no_eval_during_fit") is not True:
        raise ValueError(f"student parent {run} did not record no-eval fitting")
    checkpoint_sha = file_sha256(checkpoint_path)
    if checkpoint_sha != spec["checkpoint_sha256"]:
        raise ValueError(f"student parent {run} bytes differ from the reviewed producer")
    if provenance.get("checkpoint_sha256", {}).get(checkpoint_path.name) != checkpoint_sha:
        raise ValueError(f"student parent {run} provenance does not bind model_ep3.pt")
    fit = provenance.get("fit_row_groups")
    if not isinstance(fit, dict):
        raise ValueError(f"student parent {run} has no fitting membership")
    for split, groups in target_orders.items():
        actual = list(map(int, fit.get(split, [])))
        if len(actual) != len(set(actual)) or set(actual) != set(groups):
            raise ValueError(f"student parent {run} did not fit all {split} target groups")
    reporting_groups = list(map(int, provenance.get("evaluation_row_groups", [])))
    fitted_groups = set().union(*fit.values())
    if len(reporting_groups) != 600 or not set(reporting_groups) <= fitted_groups:
        raise ValueError(
            f"student parent {run} reporting rows are not its known all-data annotation"
        )
    return {
        "run": run,
        "seed": seed,
        "checkpoint_epoch": 3,
        "checkpoint_sha256": checkpoint_sha,
        "provenance_sha256": provenance_sha,
        "fit_membership": {
            name: {"count": len(groups), "row_groups_sha256": row_groups_sha256(groups)}
            for name, groups in target_orders.items()
        },
        "evaluation_annotation": {
            "count": len(reporting_groups),
            "row_groups_sha256": row_groups_sha256(reporting_groups),
            "classification": "reporting_population_not_heldout",
            "no_eval_during_fit": bool(args.get("no_eval_during_fit")),
        },
    }


def _array_entry(path: Path, row_groups: list[int], shape: tuple[int, ...]) -> dict[str, Any]:
    """Record an actual NPY header, payload byte length, hash, and row identity."""
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != shape or array.dtype != np.float32:
        raise ValueError(f"new target array has the wrong schema: {path}")
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "shape": list(shape),
        "dtype": "float32",
        "row_groups": row_groups,
        "row_groups_sha256": row_groups_sha256(row_groups),
    }


def build_cache(args: argparse.Namespace) -> Path:
    """Build exactly the two H390 homogeneous arrays from existing exports."""
    source_snapshot = verify_h390_source_snapshot(ROOT)
    data_root = args.data_root.resolve()
    cache_source_root = args.teacher_cache_root.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    split_path = args.teacher_split_manifest.resolve()
    split_sha = file_sha256(split_path)
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise ValueError("H390 only permits the frozen fold-1 teacher source manifest")
    split_manifest = json.loads(split_path.read_text())
    target_orders = {name: _target_index(data_root, name) for name in ("train", "valid")}
    teachers: list[dict[str, Any]] = []
    teacher_arrays: dict[str, dict[str, np.ndarray]] = {}
    for run in TEACHERS:
        record, arrays = _teacher_contract(
            run, checkpoint_root, cache_source_root, target_orders, split_manifest
        )
        teachers.append(record)
        teacher_arrays[run] = arrays
    parents = [
        _parent_contract(PARENTS[seed]["run"], checkpoint_root, target_orders) for seed in (3, 4)
    ]

    fit_rows = sum(map(len, target_orders.values()))
    output_bytes, disk_safety_reserve_bytes, estimated_bytes = cache_disk_requirement(fit_rows)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output.parent)
    min_free = int(args.min_free_gb * (1 << 30))
    budget = int(args.output_budget_gb * (1 << 30))
    if estimated_bytes > budget or disk.free - estimated_bytes < min_free:
        raise OSError(
            f"H390 cache needs {estimated_bytes} bytes; allowable bytes while preserving "
            f"{min_free} free are {max(0, disk.free - min_free)}"
        )
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to replace H390 cache or partial directory: {output}")
    partial.mkdir(parents=True)

    files: dict[str, dict[str, Any]] = {}
    for split, groups in target_orders.items():
        position = {group: index for index, group in enumerate(groups)}
        array_path = partial / f"main_homogeneous_{split}.npy"
        output_array = np.lib.format.open_memmap(
            array_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(groups), SEQUENCE_ROWS, TARGET_WIDTH),
        )
        for start in range(0, len(groups), args.chunk_sequences):
            selected = groups[start : start + args.chunk_sequences]
            rows = np.asarray([position[group] for group in selected], dtype=np.int64)
            source_values = []
            for run in TEACHERS:
                values = np.asarray(teacher_arrays[run][split][rows], dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"teacher {run} {split} export contains nonfinite values")
                source_values.append(values)
            output_array[start : start + len(selected)] = homogeneous_teacher_target(source_values)
        output_array.flush()
        del output_array
        files[f"main_homogeneous_{split}"] = _array_entry(
            array_path, groups, (len(groups), SEQUENCE_ROWS, TARGET_WIDTH)
        )

    metadata = {
        "format": FINAL_CACHE_FORMAT,
        "version": FINAL_CACHE_VERSION,
        "mode": "all_data_final",
        "heldout_claim": False,
        "builder_source_manifest_sha256": source_snapshot["source_manifest_sha256"],
        "packet_manifest_sha256": source_snapshot["packet_manifest_sha256"],
        "source_split_manifest_sha256": split_sha,
        "target_contract": TARGET_CONTRACT,
        "membership": {
            **{
                split: {
                    "source": f"datasets/{split}.parquet",
                    "count": len(groups),
                    "row_groups": groups,
                    "row_groups_sha256": row_groups_sha256(groups),
                    "source_qualified_sha256": source_qualified_groups_sha256(
                        f"datasets/{split}.parquet", groups
                    ),
                }
                for split, groups in target_orders.items()
            },
            "evaluation_row_groups": [],
            "source_qualified_groups_sha256": {
                split: source_qualified_groups_sha256(f"datasets/{split}.parquet", groups)
                for split, groups in target_orders.items()
            },
        },
        "sources": teachers,
        "parents": parents,
        "files": files,
        "output_bytes": output_bytes,
        "disk_safety_reserve_bytes": disk_safety_reserve_bytes,
        "estimated_bytes": estimated_bytes,
    }
    (partial / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(partial, output)
    validate_final_distillation_cache(output, data_root)
    print(
        json.dumps(
            {
                "cache": str(output),
                "metadata_sha256": file_sha256(output / "metadata.json"),
                "array_files": list(files),
                "output_bytes": output_bytes,
                "disk_safety_reserve_bytes": disk_safety_reserve_bytes,
                "estimated_bytes": estimated_bytes,
                "heldout_claim": False,
                "source_manifest_sha256": source_snapshot["source_manifest_sha256"],
            }
        ),
        flush=True,
    )
    return output


def main() -> None:
    """Parse the frozen H390 cache recipe and run its strict builder."""
    parser = argparse.ArgumentParser(
        description=f"{__doc__} Admission includes a {16 << 20}-byte safety reserve."
    )
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--teacher-cache-root", type=Path, default=ROOT / "cache")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "artifacts" / "runs")
    parser.add_argument("--teacher-split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-sequences", type=int, default=2)
    parser.add_argument("--output-budget-gb", type=float, default=2.5)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    args = parser.parse_args()
    if not 1 <= args.chunk_sequences <= 8:
        parser.error("chunk sequence count must be between 1 and 8")
    if args.output_budget_gb <= 0 or args.min_free_gb < 0:
        parser.error("disk budgets must be positive (minimum free space may be zero)")
    build_cache(args)


if __name__ == "__main__":
    main()
