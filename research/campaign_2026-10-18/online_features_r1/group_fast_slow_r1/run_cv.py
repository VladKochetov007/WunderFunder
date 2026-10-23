"""Fit matched component-cross-fitted ridge screens and emit label-blind OOF predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow
from group_features import extract_online_features
from ridge_model import WeightedRidge, WeightedStats
from source_utils import (
    ROOT,
    load_members_and_folds,
    require_physical_scratch,
    sha256_file,
    verify_recipe_sources,
    verify_train_footer,
    write_json,
)

from wnn.data import SEQUENCE_LENGTH, WARMUP, SequenceFile

ARMS = ("raw", "raw_plus_clock", "raw_plus_group_states")


def _sha256_bytes(data: bytes) -> str:
    """Hash in-memory serialized configuration bytes."""
    return hashlib.sha256(data).hexdigest()


def _feature_arms(sequence: np.ndarray) -> dict[str, np.ndarray]:
    """Prepare three matched representations for rows at and after warmup."""
    raw, clock, group_states = extract_online_features(sequence)
    return {
        "raw": raw,
        "raw_plus_clock": np.concatenate((raw, clock), axis=1),
        "raw_plus_group_states": np.concatenate((raw, group_states), axis=1),
    }


def _verify_prefix_invariants(sequence: np.ndarray, first_outputs: dict[str, np.ndarray]) -> None:
    """Check prefix causality against truncation and future perturbation."""
    prefix_length = 2_000
    prefix_outputs = _feature_arms(sequence[:prefix_length])
    future_perturbed = sequence.copy()
    future_perturbed[prefix_length:] += 17.0
    perturbed_outputs = _feature_arms(future_perturbed)
    prefix_rows = prefix_length - WARMUP
    for arm in ARMS:
        if not (
            np.array_equal(first_outputs[arm][:prefix_rows], prefix_outputs[arm])
            and np.array_equal(
                first_outputs[arm][:prefix_rows], perturbed_outputs[arm][:prefix_rows]
            )
        ):
            raise AssertionError(f"prefix/future-perturbation invariance failed for {arm}")


def _fit_all_folds(
    source: SequenceFile,
    members: list[dict[str, int]],
    fold_by_component: dict[int, int],
    recipe: dict[str, Any],
) -> dict[str, list[list[WeightedRidge]]]:
    """Stream component groups into the three non-held-out fold models."""
    model_config = recipe["model"]
    alpha = float(model_config["ridge_alpha"])
    stride = int(model_config["fit_row_stride"])
    fold_count = int(recipe["cv"]["fold_count"])
    widths = {arm: int(recipe["features"]["arms"][arm]["width"]) for arm in ARMS}
    stats = {
        arm: [[WeightedStats.empty(int(widths[arm])) for _ in range(2)] for _ in range(fold_count)]
        for arm in ARMS
    }
    trained_components = {fold: 0 for fold in range(fold_count)}
    clip_value = float(model_config["target_clip"])
    first_sequence: np.ndarray | None = None
    first_outputs: dict[str, np.ndarray] | None = None
    reset_checked = False

    for position, member in enumerate(members, start=1):
        component = member["component"]
        heldout_fold = fold_by_component[component]
        sequence = source.read(member["row_group"], targets=True, mask=False)
        if sequence.seq_ix != member["seq_ix"] or sequence.targets is None:
            raise ValueError(f"fit sequence identity/targets mismatch at {member['row_group']}")
        if sequence.features.shape[0] != SEQUENCE_LENGTH:
            raise ValueError("fit sequence does not have the expected complete length")

        designs = _feature_arms(sequence.features)
        if position == 1:
            _verify_prefix_invariants(sequence.features, designs)
            first_sequence = sequence.features.copy()
            first_outputs = {arm: value.copy() for arm, value in designs.items()}
        elif position == 2:
            if first_sequence is None or first_outputs is None:
                raise RuntimeError("A-B-A reset check has no stored first sequence")
            after_b = _feature_arms(first_sequence)
            if any(not np.array_equal(first_outputs[arm], after_b[arm]) for arm in ARMS):
                raise AssertionError("A-B-A sequence-local reset check failed")
            reset_checked = True
            first_sequence = None
            first_outputs = None
        sampled_targets = np.clip(sequence.targets[WARMUP::stride], -clip_value, clip_value).astype(
            np.float64, copy=False
        )
        if any(len(design) != len(sequence.targets) - WARMUP for design in designs.values()):
            raise ValueError("input feature and target rows are misaligned")

        for fold in range(fold_count):
            if fold == heldout_fold:
                continue
            trained_components[fold] += 1
            for arm in ARMS:
                sampled_design = designs[arm][::stride].astype(np.float64, copy=False)
                if not np.isfinite(sampled_design).all() or not np.isfinite(sampled_targets).all():
                    raise ValueError(f"non-finite fit values in component {component}")
                for target_index in range(2):
                    target = sampled_targets[:, target_index]
                    stats[arm][fold][target_index].update(
                        sampled_design,
                        target,
                        np.abs(target),
                    )
        print(
            json.dumps(
                {
                    "phase": "fit_group_complete",
                    "position": position,
                    "component": component,
                    "heldout_fold": heldout_fold,
                }
            ),
            flush=True,
        )
        del sequence, designs, sampled_targets

    expected_fit_components = len(members) - int(recipe["cv"]["components_per_fold"])
    if any(count != expected_fit_components for count in trained_components.values()):
        raise RuntimeError("cross-fit component counts do not match the frozen design")
    if not reset_checked:
        raise RuntimeError("two source-qualified sequences were not available for A-B-A reset")

    models: dict[str, list[list[WeightedRidge]]] = {}
    for arm in ARMS:
        models[arm] = []
        for fold in range(fold_count):
            target_models = [
                WeightedRidge.fit(stats[arm][fold][target_index], alpha)
                for target_index in range(2)
            ]
            models[arm].append(target_models)
    return models


def _predict_oof(
    source: SequenceFile,
    members: list[dict[str, int]],
    fold_by_component: dict[int, int],
    models: dict[str, list[list[WeightedRidge]]],
) -> dict[str, np.ndarray]:
    """Emit one prediction per row from models that excluded its component."""
    row_count = SEQUENCE_LENGTH - WARMUP
    predictions = {arm: np.empty((len(members), row_count, 2), dtype=np.float32) for arm in ARMS}
    for position, member in enumerate(members):
        sequence = source.read(member["row_group"], targets=False, mask=False)
        if sequence.seq_ix != member["seq_ix"]:
            raise ValueError(f"prediction sequence identity mismatch at {member['row_group']}")
        designs = _feature_arms(sequence.features)
        fold = fold_by_component[member["component"]]
        for arm in ARMS:
            for target_index in range(2):
                predictions[arm][position, :, target_index] = models[arm][fold][
                    target_index
                ].predict(designs[arm])
        print(
            json.dumps(
                {
                    "phase": "label_blind_oof_prediction_complete",
                    "position": position + 1,
                    "component": member["component"],
                    "fold": fold,
                }
            ),
            flush=True,
        )
        del sequence, designs
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise FloatingPointError("OOF prediction arrays contain non-finite values")
    return predictions


def run(root: Path, recipe_path: Path, output_path: Path) -> dict[str, Any]:
    """Run source-pinned 4-fold fitting and label-blind prediction generation."""
    started = time.perf_counter()
    output = require_physical_scratch(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifacts: {output}")
    output.mkdir(parents=True)

    recipe_bytes = recipe_path.read_bytes()
    recipe = json.loads(recipe_bytes)
    source_hashes = verify_recipe_sources(root, recipe)
    members, fold_by_component, input_hashes = load_members_and_folds(root, recipe)
    selection = recipe["selection"]
    train_path = root / selection["data_path"]
    train_identity = verify_train_footer(train_path, selection["data_identity"])

    # Parquet metadata is opened only after the measured footer identity passes.
    source = SequenceFile(train_path)
    if len(source) != int(selection["row_groups"]):
        raise ValueError("train Parquet row-group count differs from the frozen recipe")
    models = _fit_all_folds(source, members, fold_by_component, recipe)
    predictions = _predict_oof(source, members, fold_by_component, models)

    model_arrays: dict[str, np.ndarray] = {}
    for arm in ARMS:
        for fold, target_models in enumerate(models[arm]):
            for target_index, model in enumerate(target_models):
                prefix = f"{arm}_fold{fold}_target{target_index}"
                model_arrays[f"{prefix}_mean_x"] = model.mean_x
                model_arrays[f"{prefix}_scale_x"] = model.scale_x
                model_arrays[f"{prefix}_coefficients"] = model.coefficients
                model_arrays[f"{prefix}_intercept"] = np.asarray(model.intercept)
    model_path = output / "crossfit_models.npz"
    with model_path.open("xb") as destination:
        np.savez(destination, **model_arrays)

    prediction_paths: dict[str, str] = {}
    prediction_hashes: dict[str, str] = {}
    for arm, values in predictions.items():
        path = output / f"predictions_{arm}.npy"
        with path.open("xb") as destination:
            np.save(destination, values, allow_pickle=False)
        prediction_paths[arm] = str(path)
        prediction_hashes[arm] = sha256_file(path)

    member_rows = [{**member, "fold": fold_by_component[member["component"]]} for member in members]
    members_path = output / "members.json"
    write_json(members_path, {"members": member_rows})
    receipt = {
        "id": recipe["experiment_id"],
        "phase": "crossfit_fit_and_label_blind_prediction_complete",
        "status": "completed",
        "recipe_sha256": _sha256_bytes(recipe_bytes),
        "source_hashes": source_hashes,
        "sample_source_hashes": input_hashes,
        "data_identity": train_identity,
        "members": member_rows,
        "component_count": len(members),
        "fold_count": int(recipe["cv"]["fold_count"]),
        "components_per_fold": int(recipe["cv"]["components_per_fold"]),
        "fit_components_per_model": len(members) - int(recipe["cv"]["components_per_fold"]),
        "fit_rows_per_component_per_target": int(
            np.ceil((SEQUENCE_LENGTH - WARMUP) / int(recipe["model"]["fit_row_stride"]))
        ),
        "target_alignment": "same input row x[t] to target y[t], after the 99-row warmup",
        "labels_read": "fit groups only; each held-out component is excluded from its own fold model",
        "masks_read": False,
        "oof_prediction_reads_targets_or_masks": False,
        "prediction_paths": prediction_paths,
        "prediction_sha256": prediction_hashes,
        "model_archive": str(model_path),
        "model_archive_sha256": sha256_file(model_path),
        "members_path": str(members_path),
        "members_sha256": sha256_file(members_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_used": False,
        "official_metric": None,
        "result_claim": None,
    }
    receipt_path = output / "fit_predict_receipt.json"
    write_json(receipt_path, receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}), flush=True)
    return receipt


def parse_args() -> argparse.Namespace:
    """Parse bounded paths for the frozen experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=ROOT / Path(__file__).relative_to(ROOT).parent / "recipe.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Run one fixed cross-fitted ridge screen."""
    arguments = parse_args()
    run(arguments.root.resolve(), arguments.recipe.resolve(), arguments.output)


if __name__ == "__main__":
    main()
