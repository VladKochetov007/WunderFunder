"""Score the fixed seed-3, seed-4, and equal-seed-bag FP32 predictions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN = ROOT / "research/campaign_2026-10-04"
SCORECARD_PATH = ROOT / "scripts/validation_scorecard.py"
OUTPUT_DEFAULT = CAMPAIGN / "checks/e450_full_valid_fixed_models_v1.json"
PREDECLARED_SCREEN = CAMPAIGN / "checks/e450_screen_seed4_bag_fp32_v1.json"
EXPECTED_GROUPS = 1873
EXPECTED_FIT_GROUPS = 1505
EXPECTED_EXCLUDED_GROUPS = 368

MODELS = {
    "COMBO_S3": (
        ("c20261004_combo_s3_final_r1_fp32", "c20261004_combo_s3_final_r1", 3),
        ("c20261004_combo_head_s3_final_fp32", "c20261004_combo_head_s3_final", 0),
    ),
    "COMBO_S4": (
        ("c20261004_combo_s4_final_fp32", "c20261004_combo_s4_final", 3),
        ("c20261004_combo_head_s4_final_fp32", "c20261004_combo_head_s4_final", 0),
    ),
    "BAG_S3_S4": (
        ("c20261004_combo_s3_final_r1_fp32", "c20261004_combo_s3_final_r1", 3),
        ("c20261004_combo_head_s3_final_fp32", "c20261004_combo_head_s3_final", 0),
        ("c20261004_combo_s4_final_fp32", "c20261004_combo_s4_final", 3),
        ("c20261004_combo_head_s4_final_fp32", "c20261004_combo_head_s4_final", 0),
    ),
}
PREDICTION_SHA256 = {
    "c20261004_combo_s3_final_r1_fp32": "1253683928bd75915e23c490c72b3b613ef73c6b478dd19813dd453e79b13c7e",
    "c20261004_combo_head_s3_final_fp32": "0847700f6493568b7e23cc153c4b68573366da728f7c3f170cdbde68992f0875",
    "c20261004_combo_s4_final_fp32": "5917763efafa5a5e3efae725b8b70ff14765696e5cd234d2500b1036604706dc",
    "c20261004_combo_head_s4_final_fp32": "4f7f90a34713eb56d11abbdb98cb23cfbe8faf464650589846f22e2f6e5ce941",
}
SIDECAR_SHA256 = {
    "c20261004_combo_s3_final_r1_fp32": "69ec57166cd2387e57bf7fd0f9fffa45bc2d159d073f16a8bfaf63fad9066b22",
    "c20261004_combo_head_s3_final_fp32": "87671b004165761c19b7d46b6cfc2964575ac92c7b9e33ce871a83bf1b78dcdc",
    "c20261004_combo_s4_final_fp32": "062e18fc2058bc2b3470c12e562a4ced177eecf82cf2bdf749b94150b9b00ae7",
    "c20261004_combo_head_s4_final_fp32": "1fcf1c5c68112c459deea3cff48e7b45dc174be7a3fb3ef81a50512f1ede4fce",
}
CHECKPOINT_SHA256 = {
    "c20261004_combo_s3_final_r1": "64c2b28651b414778d6af3c33bb9d54f653a46fb0be087de2e3c44ca88c31166",
    "c20261004_combo_head_s3_final": "f1cb88f38b6d5b5b67859346b439e69d610928bc836baac8867e84e7b1c95d85",
    "c20261004_combo_s4_final": "87bafd489d7d6e5212589154f07fef96c447777f75f949ab4dadc1a6efe4caaf",
    "c20261004_combo_head_s4_final": "f331b218b5d08626f01c56fae484b895b2b57df3ca90592ad0d71a9fd53985df",
}
FIXED_WEIGHTS = {
    "COMBO_S3": {
        "c20261004_combo_s3_final_r1_fp32": 0.5,
        "c20261004_combo_head_s3_final_fp32": 0.5,
    },
    "COMBO_S4": {
        "c20261004_combo_s4_final_fp32": 0.5,
        "c20261004_combo_head_s4_final_fp32": 0.5,
    },
    "BAG_S3_S4": {
        "c20261004_combo_s3_final_r1_fp32": 0.25,
        "c20261004_combo_head_s3_final_fp32": 0.25,
        "c20261004_combo_s4_final_fp32": 0.25,
        "c20261004_combo_head_s4_final_fp32": 0.25,
    },
}


def digest(path: Path) -> str:
    """Hash an input incrementally with bounded memory."""
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_scorecard() -> ModuleType:
    """Load the unchanged frozen scorecard module and reuse its metric/mask helpers."""
    spec = importlib.util.spec_from_file_location(
        "e450_frozen_validation_scorecard", SCORECARD_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen scorecard at {SCORECARD_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.__file__ is None or Path(module.__file__).resolve() != SCORECARD_PATH.resolve():
        raise RuntimeError("loaded scorecard path differs from the frozen source")
    return module


def validate_model_sources(scorecard: ModuleType, rowgroups: int) -> dict[str, object]:
    """Verify pinned cache, sidecar, checkpoint, epoch, and VALID row ordering."""
    import numpy as np

    expected_shape = (rowgroups, scorecard.SEQUENCE_LENGTH, 2)
    source_records = {}
    arrays = {}
    members = sorted({member for values in MODELS.values() for member in values})
    for export, run, epoch in members:
        prediction_path = ROOT / f"cache/pred_{export}_valid.npy"
        sidecar_path = prediction_path.with_suffix(".json")
        checkpoint_path = ROOT / f"artifacts/runs/{run}/model_ep{epoch}.pt"
        provenance_path = ROOT / f"artifacts/runs/{run}/provenance.json"
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        prediction_hash = digest(prediction_path)
        sidecar_hash = digest(sidecar_path)
        checkpoint_hash = digest(checkpoint_path)
        if (
            metadata.get("run") != run
            or metadata.get("split") != "valid"
            or metadata.get("epoch") != epoch
            or metadata.get("inference_precision") != "fp32"
            or metadata.get("prediction_sha256") != prediction_hash
            or metadata.get("checkpoint_sha256") != checkpoint_hash
            or metadata.get("row_groups") != list(range(rowgroups))
            or prediction_hash != PREDICTION_SHA256[export]
            or sidecar_hash != SIDECAR_SHA256[export]
            or checkpoint_hash != CHECKPOINT_SHA256[run]
        ):
            raise ValueError(
                f"FP32 predictor identity or ordered all-VALID cache differs: {export}"
            )
        value = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        if value.shape != expected_shape or value.dtype != np.float32:
            raise ValueError(f"wrong FP32 cache shape or dtype: {prediction_path}")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            provenance.get("selected_epoch") != epoch
            or provenance.get("args", {}).get("checkpoint_policy") != "fixed"
            or provenance.get("args", {}).get("checkpoint_epoch") != epoch
        ):
            raise ValueError(f"source provenance is not the prescribed fixed epoch: {run}")
        arrays[export] = value
        source_records[export] = {
            "run": run,
            "epoch": epoch,
            "coefficient_in_seed_pair": 0.5,
            "prediction": {
                "path": prediction_path.relative_to(ROOT).as_posix(),
                "sha256": prediction_hash,
                "bytes": prediction_path.stat().st_size,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "row_groups": metadata["row_groups"],
            },
            "sidecar": {
                "path": sidecar_path.relative_to(ROOT).as_posix(),
                "sha256": sidecar_hash,
            },
            "checkpoint": {
                "path": checkpoint_path.relative_to(ROOT).as_posix(),
                "sha256": checkpoint_hash,
            },
            "provenance": {
                "path": provenance_path.relative_to(ROOT).as_posix(),
                "sha256": digest(provenance_path),
                "fit_valid_groups": len(provenance.get("fit_row_groups", {}).get("valid", [])),
                "evaluation_groups": len(provenance.get("evaluation_row_groups", [])),
            },
        }
    return {"arrays": arrays, "source_records": source_records}


def validate_source_partitions(
    scorecard: ModuleType,
    sequence_ids: list[int],
    fit_rowgroups: set[int],
    evaluation_rowgroups: set[int],
) -> dict[str, list[int]]:
    """Match all four fit/evaluation provenance records to scorecard holdout groups."""
    if len(sequence_ids) != EXPECTED_GROUPS or len(set(sequence_ids)) != EXPECTED_GROUPS:
        raise ValueError("supplied VALID rows do not contain 1,873 unique sequences")
    excluded = {
        group for group, sequence_id in enumerate(sequence_ids) if scorecard.is_holdout(sequence_id)
    }
    fit = set(range(len(sequence_ids))) - excluded
    if len(excluded) != EXPECTED_EXCLUDED_GROUPS or len(fit) != EXPECTED_FIT_GROUPS:
        raise ValueError(
            f"unexpected supplied VALID partition sizes: fit={len(fit)}, excluded={len(excluded)}"
        )
    if fit_rowgroups != fit or evaluation_rowgroups != excluded:
        raise ValueError(
            "fixed model provenance groups differ from the frozen scorecard partitions"
        )
    return {
        "all_supplied_validation": list(range(len(sequence_ids))),
        "validation_in_final_fit": sorted(fit),
        "excluded_from_final_fit": sorted(excluded),
    }


def score(output: Path) -> dict[str, object]:
    """Score fixed mixtures on all VALID sequences with the frozen metric and masks."""
    import numpy as np

    if output.exists():
        raise FileExistsError(output)
    preregistered = json.loads(PREDECLARED_SCREEN.read_text(encoding="utf-8"))
    registered_arms = preregistered.get("arms", {})
    if any(registered_arms.get(name) != weights for name, weights in FIXED_WEIGHTS.items()):
        raise ValueError("fixed predictor transforms differ from preregistered seed bag")
    scorecard = load_scorecard()
    data_path = ROOT / "datasets/valid.parquet"
    parquet = scorecard.pq.ParquetFile(data_path)
    count = parquet.num_row_groups
    if count != EXPECTED_GROUPS:
        raise ValueError(f"expected {EXPECTED_GROUPS} VALID row groups, got {count}")
    sources = validate_model_sources(scorecard, count)
    arrays = sources["arrays"]
    partitions = (
        "all_supplied_validation",
        "validation_in_final_fit",
        "excluded_from_final_fit",
    )
    accumulators = {
        model_name: {partition: scorecard.GlobalAccumulator() for partition in partitions}
        for model_name in MODELS
    }
    sizes = dict.fromkeys(partitions, 0)
    sequence_ids = []
    columns = ["seq_ix", "step_in_seq", "need_prediction", "is_scored", *scorecard.TARGET_COLUMNS]
    for group in range(count):
        table = parquet.read_row_group(group, columns=columns, use_threads=False)
        sequence_id, need = scorecard.validate_sequence(table)
        sequence_ids.append(sequence_id)
        target = np.column_stack([table[name].to_numpy() for name in scorecard.TARGET_COLUMNS])
        mask = need & table["is_scored"].to_numpy().astype(bool)
        partition = (
            "excluded_from_final_fit"
            if scorecard.is_holdout(sequence_id)
            else "validation_in_final_fit"
        )
        sizes["all_supplied_validation"] += 1
        sizes[partition] += 1

        for model_name, members in MODELS.items():
            weighted = [
                np.asarray(arrays[export][group], np.float64) / len(members)
                for export, _run, _epoch in members
            ]
            prediction = sum(weighted)
            if not np.isfinite(prediction[need]).all():
                raise ValueError(f"nonfinite required-row prediction: {model_name}, group {group}")
            accumulators[model_name]["all_supplied_validation"].add(target, prediction, mask)
            accumulators[model_name][partition].add(target, prediction, mask)

    rowgroups_by_partition = {
        "validation_in_final_fit": set(),
        "excluded_from_final_fit": set(),
    }
    for group, sequence_id in enumerate(sequence_ids):
        key = (
            "excluded_from_final_fit"
            if scorecard.is_holdout(sequence_id)
            else "validation_in_final_fit"
        )
        rowgroups_by_partition[key].add(group)
    expected_fit = None
    expected_evaluation = None
    provenance_records = sources["source_records"]
    for source in provenance_records.values():
        provenance_path = ROOT / source["provenance"]["path"]
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        valid_fit = set(provenance.get("fit_row_groups", {}).get("valid", []))
        evaluation = set(provenance.get("evaluation_row_groups", []))
        if (
            len(valid_fit) != EXPECTED_FIT_GROUPS
            or len(evaluation) != EXPECTED_EXCLUDED_GROUPS
            or source["provenance"]["fit_valid_groups"] != EXPECTED_FIT_GROUPS
            or source["provenance"]["evaluation_groups"] != EXPECTED_EXCLUDED_GROUPS
        ):
            raise ValueError(f"source provenance has wrong fit/evaluation counts: {source['run']}")
        if expected_fit is None:
            expected_fit, expected_evaluation = valid_fit, evaluation
        elif valid_fit != expected_fit or evaluation != expected_evaluation:
            raise ValueError("source models do not share fixed VALID fit/evaluation memberships")
    validate_source_partitions(
        scorecard,
        sequence_ids,
        expected_fit or set(),
        expected_evaluation or set(),
    )
    if sizes != {
        "all_supplied_validation": EXPECTED_GROUPS,
        "validation_in_final_fit": EXPECTED_FIT_GROUPS,
        "excluded_from_final_fit": EXPECTED_EXCLUDED_GROUPS,
    }:
        raise ValueError(f"scorecard row counts changed: {sizes}")

    result = {
        "metric_id": "global_wp_clip2_v1",
        "dataset": data_path.relative_to(ROOT).as_posix(),
        "dataset_sha256": scorecard.digest(data_path),
        "row_groups": count,
        "rows": count * scorecard.SEQUENCE_LENGTH,
        "partition_sequence_counts": sizes,
        "partition_row_groups": {
            key: rows
            for key, rows in {
                "validation_in_final_fit": sorted(
                    rowgroups_by_partition["validation_in_final_fit"]
                ),
                "excluded_from_final_fit": sorted(
                    rowgroups_by_partition["excluded_from_final_fit"]
                ),
            }.items()
        },
        "mask": "need_prediction AND is_scored (frozen validation_scorecard mask)",
        "fixed_transforms": {
            **FIXED_WEIGHTS,
            "preregistered_screen_sha256": digest(PREDECLARED_SCREEN),
            "checkpoint_or_weight_selection": "none",
        },
        "sources": provenance_records,
        "source_code_sha256": {
            path: scorecard.digest(ROOT / path)
            for path in (
                "scripts/validation_scorecard.py",
                "utils.py",
                "src/wnn/splits.py",
                Path(__file__).resolve().relative_to(ROOT).as_posix(),
            )
        },
        "scores": {
            model_name: {
                partition: accumulator.result()
                for partition, accumulator in model_partitions.items()
            }
            for model_name, model_partitions in accumulators.items()
        },
        "scope": (
            "Fixed all-VALID FP32 cache report for the seed-3 combo, seed-4 combo, and preregistered "
            "equal 50/50 seed bag (four fixed 25% members). 1,505 supplied VALID sequences were in "
            "the final fit and 368 are historically excluded/exposed rows. Descriptive only; no "
            "weight or checkpoint selection, no untouched-test claim, no native runtime result."
        ),
        "selection": "none",
        "public_score_measured": False,
        "native_runtime_measured": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    """Run the fixed full-VALID comparison and write its immutable scorecard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    arguments = parser.parse_args()
    print(json.dumps(score(arguments.output.resolve()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
