"""Build H390's exact homogeneous strong-T3 target on clean fold-0 fitting rows."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from wnn.distillation_cache import (
    CACHE_FORMAT,
    CACHE_VERSION,
    CLEANFOLD_T3_CACHE_PROFILE,
    row_groups_sha256,
)
from wnn.final_distillation_cache import (
    TARGET_MULTIPLIER,
    homogeneous_teacher_target,
    source_qualified_groups_sha256,
)
from wnn.splits import CampaignSplit, file_sha256, load_campaign_split


ROOT = Path(os.environ.get("WNN_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
SPLIT_PATH = Path("research/campaign_2026-09-29/artifacts/splits/fold0.json")
SPLIT_SHA256 = "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d"
PROPENSITY_SHA256 = "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7"
F1_HOMOGENEOUS_CACHE_METADATA_SHA256 = "a2f82d8dbc7bbbe2c7cf36e18e3b7be0b41f887aa14a742b43f616b060856779"
TEACHER_RUNS = {
    3: "clean_t3_f0_muonall_h384_e3_s3_r1",
    4: "clean_t3_f0_muonall_h384_e3_s4_r1",
    5: "clean_t3_f0_muonall_h384_e3_s5_r1",
}
STUDENT_CONTRACT = {
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
}
TARGET_CONTRACT = {
    "family": "strong",
    "formula": "float32(0.8125 * float32_mean(three_epoch2_predictions))",
    "coefficient_multiplier": TARGET_MULTIPLIER,
    "teacher_mean": "unweighted arithmetic mean in float32",
    "teacher_epoch": 2,
    "clipping": "none during cache construction",
}


def _canonical_groups(split: CampaignSplit, source: str) -> list[int]:
    target_path = ROOT / "cache" / f"{source}_targets.npz"
    with np.load(target_path, allow_pickle=False) as target_index:
        ordered = list(map(int, target_index["row_group"]))
    fitting = set(split.train_groups if source == "train" else split.valid_groups)
    selected = [group for group in ordered if group in fitting]
    if len(selected) != len(fitting) or set(selected) != fitting:
        raise ValueError(f"{source} target index does not cover exact cleanfold fitting rows")
    return selected


def _recipe_args(run: str, seed: int, provenance: dict[str, Any], split: CampaignSplit) -> None:
    args = provenance.get("args", {})
    expected = {
        "name": run,
        "seed": seed,
        "arch": "gru",
        "cell": "gru",
        "hidden": 384,
        "layers": 2,
        "dropout": 0.0,
        "batch": 32,
        "chunk": 1000,
        "epochs": 3,
        "checkpoint_policy": "fixed",
        "checkpoint_epoch": 2,
        "no_eval_during_fit": True,
        "split_manifest_sha256": split.sha256,
        "train_source": "train+valid",
        "eval_source": "vfold",
        "valid_fold": 0,
        "loss": "wmse",
        "loss_norm": "chunk",
        "lr": 0.001,
        "wd": 0.01,
        "optim": "muon",
        "muon_scope": "all",
        "sched": "onecycle",
        "precision": "fp16",
        "aux_struct": 0.3,
        "aux_next": 0.3,
        "propensity": "c20260929_prop_f0",
        "prop_power": 2.0,
        "prop_eps": 0.03,
        "valid_true_mask": 0.03,
        "since_weight": 32.0,
        "since_floor": 0.25,
        "init": "",
        "inherit_init_preprocessing": False,
    }
    for key, value in expected.items():
        if args.get(key) != value:
            raise ValueError(f"teacher {run} provenance differs at {key}: {args.get(key)!r}")
    if provenance.get("propensity_checkpoint_sha256") != PROPENSITY_SHA256:
        raise ValueError(f"teacher {run} used a different F0 propensity checkpoint")
    if provenance.get("split_manifest_sha256") != split.sha256:
        raise ValueError(f"teacher {run} provenance names a different split")
    fit = provenance.get("fit_row_groups", {})
    if (set(map(int, fit.get("train", []))) != set(split.train_groups)
            or set(map(int, fit.get("valid", []))) != set(split.valid_groups)):
        raise ValueError(f"teacher {run} fit row groups differ from the frozen fold")
    if set(map(int, provenance.get("evaluation_row_groups", []))) != set(split.evaluation_groups):
        raise ValueError(f"teacher {run} evaluation row groups differ from the frozen fold")
    expected_membership = {
        "train": {
            "count": 10_535,
            "ordered_int64_le_sha256": "98a3e8da248f52e307af61efaeab845a76cf8e89e005ea5630c0af9a133508a9",
        },
        "valid": {
            "count": 1_405,
            "ordered_int64_le_sha256": "0ef790f8c27723b8f0ca1ebe045c2dcbebb4c4531a17952985802382c9568d75",
        },
    }
    identity = provenance.get("fit_membership_identity", {})
    if any(identity.get(source) != expected for source, expected in expected_membership.items()):
        raise ValueError(f"teacher {run} source-qualified fit membership identity differs")


def _teacher_source(root: Path, run: str, seed: int, split: CampaignSplit,
                    fit_groups: dict[str, list[int]]) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, dict[int, int]]]]:
    run_root = root / "artifacts" / "runs" / run
    checkpoint = run_root / "model_ep2.pt"
    provenance_path = run_root / "provenance.json"
    if not checkpoint.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f"teacher checkpoint/provenance missing: {run}")
    provenance = json.loads(provenance_path.read_text())
    _recipe_args(run, seed, provenance, split)
    checkpoint_sha = file_sha256(checkpoint)
    if provenance.get("checkpoint_sha256", {}).get(checkpoint.name) != checkpoint_sha:
        raise ValueError(f"teacher {run} checkpoint bytes differ from provenance")

    predictions: dict[str, tuple[np.ndarray, dict[int, int]]] = {}
    prediction_record: dict[str, Any] = {}
    for source in ("train", "valid"):
        prediction = root / "cache" / f"pred_{run}_{source}.npy"
        sidecar_path = prediction.with_suffix(".json")
        if not prediction.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(f"teacher {run}/{source} feature export is missing")
        sidecar = json.loads(sidecar_path.read_text())
        expected_rows = fit_groups[source]
        if (
            sidecar.get("run") != run
            or sidecar.get("split") != source
            or sidecar.get("epoch") != 2
            or sidecar.get("checkpoint_sha256") != checkpoint_sha
            or sidecar.get("split_manifest_sha256") != split.sha256
            or sidecar.get("campaign_split_manifest_sha256") != split.sha256
            or sidecar.get("campaign_fit_scope") != source
            or sidecar.get("row_groups") != expected_rows
        ):
            raise ValueError(f"teacher {run}/{source} prediction sidecar differs from cleanfold")
        prediction_sha = file_sha256(prediction)
        if sidecar.get("prediction_sha256") != prediction_sha:
            raise ValueError(f"teacher {run}/{source} prediction bytes differ from sidecar")
        values = np.load(prediction, mmap_mode="r", allow_pickle=False)
        expected_shape = (len(expected_rows), 20_000, 2)
        if values.shape != expected_shape or values.dtype != np.float32:
            raise ValueError(f"teacher {run}/{source} array shape or dtype differs")
        for first in range(0, len(values), 32):
            if not np.isfinite(values[first : first + 32]).all():
                raise ValueError(f"teacher {run}/{source} predictions contain nonfinite values")
        predictions[source] = (values, {group: index for index, group in enumerate(expected_rows)})
        prediction_record[source] = {
            "path": prediction.relative_to(root).as_posix(),
            "sha256": prediction_sha,
            "sidecar_sha256": file_sha256(sidecar_path),
            "row_groups_sha256": row_groups_sha256(expected_rows),
        }

    return (
        {
            "run": run,
            "family": "strong",
            "epoch": 2,
            "coefficient": 0.1,
            "checkpoint_path": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": checkpoint_sha,
            "provenance_path": provenance_path.relative_to(root).as_posix(),
            "provenance_sha256": file_sha256(provenance_path),
            "propensity_checkpoint_sha256": provenance["propensity_checkpoint_sha256"],
            "split_manifest_sha256": split.sha256,
            "predictions": prediction_record,
        },
        predictions,
    )


def _entry(path: Path, groups: list[int]) -> dict[str, Any]:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": path.name,
        "row_groups": groups,
        "row_groups_sha256": row_groups_sha256(groups),
        "shape": list(values.shape),
        "dtype": "float32",
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def build(output: Path, *, chunk_sequences: int = 4, min_free_gb: float = 20.0,
          output_budget_gb: float = 2.5) -> Path:
    """Build the exact fold-0 T3 target after verifying all teacher ancestry."""
    split_path = ROOT / SPLIT_PATH
    split = load_campaign_split(split_path, ROOT)
    if split.sha256 != SPLIT_SHA256 or split.fold != 0:
        raise ValueError("cleanfold T3 cache requires the frozen fold-0 split")
    split.assert_fit_groups(split.train_groups, split.valid_groups)
    evaluation = set(split.evaluation_groups)
    component_of = {(source, group): component for source, group, component in split.components}
    fit_components = {
        component_of[("train", group)] for group in split.train_groups
    } | {
        component_of[("valid", group)] for group in split.valid_groups
    }
    evaluation_components = {component_of[("valid", group)] for group in split.evaluation_groups}
    if fit_components & evaluation_components:
        raise ValueError("fold-0 teacher/student/cache fit ancestry overlaps evaluation components")
    fit_groups = {"train": list(split.train_groups), "valid": list(split.valid_groups)}
    cache_groups = {name: _canonical_groups(split, name) for name in ("train", "valid")}
    teacher_records: list[dict[str, Any]] = []
    teacher_arrays: dict[str, dict[str, tuple[np.ndarray, dict[int, int]]]] = {}
    for seed, run in TEACHER_RUNS.items():
        record, arrays = _teacher_source(ROOT, run, seed, split, fit_groups)
        teacher_records.append(record)
        teacher_arrays[run] = arrays

    rows = sum(len(value) for value in cache_groups.values())
    output_bytes = rows * 20_000 * 2 * np.dtype(np.float32).itemsize
    budget = int(output_budget_gb * (1 << 30))
    min_free = int(min_free_gb * (1 << 30))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output_bytes > budget or output.parent.stat().st_dev != ROOT.stat().st_dev:
        raise OSError("cache output exceeds its declared bound or is outside the project filesystem")
    if output.exists() or output.with_name(output.name + ".partial").exists():
        raise FileExistsError(f"refusing to replace existing cache: {output}")
    disk = os.statvfs(output.parent)
    free_bytes = disk.f_bavail * disk.f_frsize
    if free_bytes - output_bytes < min_free:
        raise OSError(
            f"cache needs {output_bytes} bytes and must preserve {min_free} free; "
            f"currently available {free_bytes}"
        )
    if not 1 <= chunk_sequences <= 8:
        raise ValueError("chunk_sequences must be between 1 and 8")

    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    files: dict[str, dict[str, Any]] = {}
    for source, groups in cache_groups.items():
        row_maps = {
            run: teacher_arrays[run][source][1] for run in TEACHER_RUNS.values()
        }
        path = partial / f"main_homogeneous_{source}.npy"
        target = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=(len(groups), 20_000, 2),
        )
        for first in range(0, len(groups), chunk_sequences):
            selected = groups[first : first + chunk_sequences]
            source_values = []
            for run in TEACHER_RUNS.values():
                values, positions = teacher_arrays[run][source]
                indices = [positions[group] for group in selected]
                source_values.append(np.asarray(values[indices], dtype=np.float32))
            target[first : first + len(selected)] = homogeneous_teacher_target(source_values)
        target.flush()
        del target
        files[f"main_homogeneous_{source}"] = _entry(path, groups)

    membership = {
        "train": {
            "source": "datasets/train.parquet",
            "count": len(cache_groups["train"]),
            "row_groups": cache_groups["train"],
            "row_groups_sha256": row_groups_sha256(cache_groups["train"]),
            "source_qualified_sha256": source_qualified_groups_sha256(
                "datasets/train.parquet", cache_groups["train"]
            ),
        },
        "valid": {
            "source": "datasets/valid.parquet",
            "count": len(cache_groups["valid"]),
            "row_groups": cache_groups["valid"],
            "row_groups_sha256": row_groups_sha256(cache_groups["valid"]),
            "source_qualified_sha256": source_qualified_groups_sha256(
                "datasets/valid.parquet", cache_groups["valid"]
            ),
        },
        "evaluation_row_groups": sorted(evaluation),
        "source_qualified_groups_sha256": {
            "train": source_qualified_groups_sha256(
                "datasets/train.parquet", cache_groups["train"]
            ),
            "valid": source_qualified_groups_sha256(
                "datasets/valid.parquet", cache_groups["valid"]
            ),
        },
    }
    coefficient = 0.1
    coefficient_sum = math.fsum([coefficient] * 3)
    metadata = {
        "format": CACHE_FORMAT,
        "version": CACHE_VERSION,
        "cache_profile": CLEANFOLD_T3_CACHE_PROFILE,
        "fold": 0,
        "split_manifest_sha256": split.sha256,
        "fit_groups": {"train": list(split.train_groups), "valid": list(split.valid_groups)},
        "evaluation_valid_groups": list(split.evaluation_groups),
        "family_order": ["strong"],
        "homogeneous_family": "strong",
        "source_coefficients": [
            {"run": run, "family": "strong", "epoch": 2, "coefficient": coefficient}
            for run in TEACHER_RUNS.values()
        ],
        "coefficient_sum": coefficient_sum,
        "homogeneous_scale": TARGET_MULTIPLIER,
        "reference_homogeneous_cache_metadata_sha256": F1_HOMOGENEOUS_CACHE_METADATA_SHA256,
        "sources": teacher_records,
        "parents": [],
        "target_contract": TARGET_CONTRACT,
        "student_contract": STUDENT_CONTRACT,
        "epoch_convention": {
            "student_epochs": 6,
            "student_checkpoint_policy": "fixed",
            "student_checkpoint_epoch": 5,
        },
        "membership": membership,
        "files": files,
        "output_bytes": output_bytes,
        "disk_safety_reserve_bytes": min_free,
        "estimated_bytes": output_bytes + min_free,
        "heldout_claim": False,
    }
    (partial / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output)
    print(json.dumps({
        "status": "complete",
        "cache": str(output),
        "split_manifest_sha256": split.sha256,
        "teacher_runs": list(TEACHER_RUNS.values()),
        "training_groups": {name: len(value) for name, value in fit_groups.items()},
        "evaluation_groups_written": 0,
        "output_bytes": output_bytes,
        "source_qualified_fit_hashes": {
            name: membership["source_qualified_groups_sha256"][name]
            for name in ("train", "valid")
        },
    }, sort_keys=True), flush=True)
    return output


def main() -> None:
    """Parse the bounded cache-build command and write one new cache directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-sequences", type=int, default=4)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--output-budget-gb", type=float, default=2.5)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    build(output, chunk_sequences=args.chunk_sequences, min_free_gb=args.min_free_gb,
          output_budget_gb=args.output_budget_gb)


if __name__ == "__main__":
    main()
