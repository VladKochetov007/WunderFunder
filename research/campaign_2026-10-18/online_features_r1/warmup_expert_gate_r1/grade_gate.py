from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
PACKET = Path(__file__).resolve().parent
RECIPE_PATH = PACKET / "recipe.json"

SPLIT_REL = "research/campaign_2026-09-29/artifacts/splits/fold0.json"
TARGET_REL = "cache/valid_targets.npz"
GEOMETRY_GRADE_REL = "research/campaign_2026-10-18/geometry_variations_r3/geometry_r3_grade.json"
N_FOLDS = 4
FOLD_SEED = 20261018
RIDGE_ALPHA = 25.0
SOFTMAX_TEMPERATURE = 0.01
UNIFORM_SHRINKAGE = 0.5
N_BOOT = 10_000
BOOT_SEED = 20261019
WARMUP = 99


class PredictionView(Protocol):
    """Small interface for an aligned prediction array or lazy memory-map view."""

    def __getitem__(self, part: slice) -> np.ndarray: ...


def sha256_file(path: Path) -> str:
    """Hash an artifact incrementally to keep memory use bounded."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ints(values: list[int]) -> str:
    """Hash an ordered integer membership list using canonical JSON bytes."""
    payload = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    """Import a repository helper from a frozen path without running its CLI."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load helper at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_prediction_bank(
    root: Path,
    groups: list[int],
    split_sha: str,
    grade: ModuleType,
    candidate_specs: dict[str, dict[str, object]],
) -> tuple[dict[str, PredictionView], dict[str, dict[str, object]]]:
    """Bind the four source-qualified F0 prediction arrays to the fixed group order."""
    arrays: dict[str, PredictionView] = {}
    receipts: dict[str, dict[str, object]] = {}
    for name, candidate in candidate_specs.items():
        relative = str(candidate["prediction"])
        expected_hash = str(candidate["prediction_sha256"])
        expected_sidecar_hash = str(candidate["sidecar_sha256"])
        expected_checkpoint_hash = str(candidate["checkpoint_sha256"])
        path = root / relative
        sidecar_path = path.with_suffix(".json")
        if sha256_file(path) != expected_hash or sha256_file(sidecar_path) != expected_sidecar_hash:
            raise ValueError(f"{name} prediction or sidecar differs from the frozen recipe")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("split_manifest_sha256") != split_sha:
            raise ValueError(f"{name} prediction has a different split identity")
        if sidecar.get("prediction_sha256") != expected_hash:
            raise ValueError(f"{name} sidecar does not declare the frozen prediction hash")
        if sidecar.get("checkpoint_sha256") != expected_checkpoint_hash:
            raise ValueError(f"{name} sidecar does not declare the frozen checkpoint hash")
        sidecar_groups = [int(value) for value in sidecar.get("row_groups", [])]
        if sidecar_groups not in (groups, list(range(1_873))):
            raise ValueError(
                f"{name} sidecar does not declare the exact evaluation or full-validation order"
            )
        aligned = cast(Callable[..., object], grade.aligned)(
            {"prediction": relative}, groups, split_sha
        )
        if isinstance(aligned, str):
            raise ValueError(f"{name} alignment failed: {aligned}")
        array, order = cast(tuple[np.ndarray, list[int]], aligned)
        if array.dtype != np.float32 or array.shape[1:] != (20_000, 2):
            raise ValueError(f"{name} prediction shape or precision differs")
        arrays[name] = cast(PredictionView, grade.Lazy(array, order))
        for start in range(0, len(groups), 8):
            if not np.isfinite(arrays[name][slice(start, start + 8)]).all():
                raise ValueError(f"{name} has nonfinite values on the selected F0 groups")
        receipts[name] = {
            "path": relative,
            "prediction_sha256": expected_hash,
            "sidecar_sha256": expected_sidecar_hash,
            "checkpoint_sha256": sidecar.get("checkpoint_sha256"),
            "row_group_count": len(sidecar.get("row_groups", [])),
            "aligned_groups": len(groups),
            "split_manifest_sha256": sidecar.get("split_manifest_sha256"),
        }
    return arrays, receipts


def write_preflight(
    root: Path,
    groups: list[int],
    sequence_ids: list[int],
    split_sha: str,
    target_path: Path,
    features_path: Path,
    features_receipt_path: Path,
    prediction_receipts: dict[str, dict[str, object]],
    candidate_specs: dict[str, dict[str, object]],
    ancestry_receipt: dict[str, object],
    output_path: Path,
) -> dict[str, object]:
    """Freeze artifact and ancestry checks before any target or mask values are read."""
    geometry_grade_path = root / GEOMETRY_GRADE_REL
    geometry_grade = json.loads(geometry_grade_path.read_text(encoding="utf-8"))
    if geometry_grade["split_manifest_sha256"] != split_sha:
        raise ValueError("geometry endpoint grade uses a different source-qualified fold")
    if geometry_grade["matched_fit_metadata"]["pass"] is not True:
        raise ValueError("geometry endpoints failed the matched fit-membership audit")
    for name, candidate_spec in candidate_specs.items():
        candidate = geometry_grade["candidates"][name]
        if (
            candidate["prediction_sha256"] != candidate_spec["prediction_sha256"]
            or candidate["sidecar_sha256"] != candidate_spec["sidecar_sha256"]
            or candidate["producer_checkpoint_sha256"] != candidate_spec["checkpoint_sha256"]
            or candidate["split_manifest_sha256"] != split_sha
        ):
            raise ValueError(f"{name} differs from the frozen geometry grade")

    source_hashes = {
        "recipe": sha256_file(RECIPE_PATH),
        "feature_extractor": sha256_file(PACKET / "extract_warmup_features.py"),
        "gate_grader": sha256_file(Path(__file__).resolve()),
        "geometry_grade": sha256_file(geometry_grade_path),
        "target_cache_reader": sha256_file(
            root / "research/campaign_2026-09-30/audit/campaign_score.py"
        ),
        "prediction_aligner": sha256_file(root / "scripts/grade_prediction_bank.py"),
        "official_metric": sha256_file(root / "src/wnn/gmetric.py"),
        "component_evaluation": sha256_file(root / "src/wnn/evaluation.py"),
    }
    payload: dict[str, object] = {
        "schema": "wnn.warmup_expert_gate.preflight.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "pass",
        "target_numeric_values_decoded_or_used": False,
        "scoring_mask_values_decoded_or_used": False,
        "target_cache_archive_sha256_verified": True,
        "fold": 0,
        "split_manifest_sha256": split_sha,
        "evaluation_group_order_sha256": sha256_ints(groups),
        "evaluation_sequence_id_order_sha256": sha256_ints(sequence_ids),
        "evaluation_groups": len(groups),
        "target_cache": {
            "path": str(target_path.relative_to(root)),
            "size_bytes": target_path.stat().st_size,
            "sha256": sha256_file(target_path),
            "row_group_count": 1_873,
        },
        "warmup_features": {
            "path": str(features_path),
            "sha256": sha256_file(features_path),
            "receipt_path": str(features_receipt_path),
            "receipt_sha256": sha256_file(features_receipt_path),
        },
        "geometry_grade": {
            "path": GEOMETRY_GRADE_REL,
            "sha256": source_hashes["geometry_grade"],
            "matched_fit_metadata_pass": True,
        },
        "candidate_predictions": prediction_receipts,
        "verified_fit_and_teacher_ancestry": ancestry_receipt,
        "source_hashes": source_hashes,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload["fingerprint"] = fingerprint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return payload


def verify_run_membership(
    root: Path,
    split: dict[str, object],
    groups: list[int],
    split_sha: str,
    provenance_path: str,
    expected_sha256: str,
    name: str,
    expected_checkpoint: tuple[str, str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Verify one run's frozen source-qualified fit/evaluation membership and checkpoint."""
    path = root / provenance_path
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{name} provenance differs from the frozen recipe")
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if provenance.get("split_manifest_sha256") != split_sha:
        raise ValueError(f"{name} was fitted with a different split manifest")

    expected_fit = cast(dict[str, list[int]], split["fit"])
    observed_fit = cast(dict[str, list[int]], provenance["fit_row_groups"])
    if set(observed_fit) != set(expected_fit):
        raise ValueError(f"{name} fit membership sources differ from the frozen split")
    for source, expected_groups in expected_fit.items():
        observed = [int(value) for value in observed_fit[source]]
        if len(observed) != len(set(observed)) or set(observed) != set(expected_groups):
            raise ValueError(f"{name} {source} fit groups differ from the frozen split")
    evaluation_groups = [int(value) for value in provenance["evaluation_row_groups"]]
    if len(evaluation_groups) != len(set(evaluation_groups)) or set(evaluation_groups) != set(
        groups
    ):
        raise ValueError(f"{name} evaluation groups differ from the scored F0 panel")
    if set(expected_fit["valid"]) & set(groups):
        raise ValueError(f"{name} fitted on one or more scored F0 validation groups")

    valid_identity = provenance.get("fit_membership_identity", {}).get("valid", {})
    expected_identity = cast(
        dict[str, object], json.loads(RECIPE_PATH.read_text())["fit_membership_identity"]["valid"]
    )
    if valid_identity != expected_identity:
        raise ValueError(
            f"{name} validation fit-membership identity differs from the frozen recipe"
        )
    checkpoint_receipt: dict[str, object] = {}
    if expected_checkpoint is not None:
        checkpoint_name, checkpoint_sha = expected_checkpoint
        observed_sha = provenance.get("checkpoint_sha256", {}).get(checkpoint_name)
        if observed_sha != checkpoint_sha:
            raise ValueError(f"{name} checkpoint hash differs from its run provenance")
        checkpoint_receipt = {"name": checkpoint_name, "sha256": checkpoint_sha}
    return provenance, {
        "run": name,
        "provenance": provenance_path,
        "provenance_sha256": expected_sha256,
        "split_manifest_sha256": split_sha,
        "fit_group_counts": {key: len(value) for key, value in observed_fit.items()},
        "evaluation_group_count": len(evaluation_groups),
        "checkpoint": checkpoint_receipt,
    }


def verify_fit_and_teacher_ancestry(
    root: Path,
    recipe: dict[str, object],
    split: dict[str, object],
    groups: list[int],
    split_sha: str,
) -> dict[str, object]:
    """Verify clean fit membership for scored candidates, initializer and distillation teachers."""
    candidates = cast(dict[str, dict[str, object]], recipe["candidate_experts"])
    candidate_provenance: dict[str, dict[str, object]] = {}
    candidate_receipts: dict[str, dict[str, object]] = {}
    for name, candidate in candidates.items():
        provenance, receipt = verify_run_membership(
            root,
            split,
            groups,
            split_sha,
            cast(str, candidate["provenance"]),
            cast(str, candidate["provenance_sha256"]),
            name,
            (cast(str, candidate["checkpoint_name"]), cast(str, candidate["checkpoint_sha256"])),
        )
        candidate_provenance[name] = provenance
        candidate_receipts[name] = receipt

    ancestry = cast(dict[str, object], recipe["ancestry"])
    initializer = cast(dict[str, object], ancestry["warm_parent_initializer"])
    initializer_provenance, initializer_receipt = verify_run_membership(
        root,
        split,
        groups,
        split_sha,
        cast(str, initializer["provenance"]),
        cast(str, initializer["provenance_sha256"]),
        cast(str, initializer["run"]),
        (cast(str, initializer["checkpoint_name"]), cast(str, initializer["checkpoint_sha256"])),
    )
    if (
        candidate_provenance["parent"].get("initial_checkpoint_sha256")
        != initializer["checkpoint_sha256"]
    ):
        raise ValueError("warm parent initializer hash does not match the recorded source run")
    for name in ("r2_full_geometry", "r3_moments_only", "r3_dictionary_only"):
        if (
            candidate_provenance[name].get("initial_checkpoint_sha256")
            != candidates["parent"]["checkpoint_sha256"]
        ):
            raise ValueError(f"{name} does not inherit the frozen clean warm parent")

    cache_spec = cast(dict[str, object], ancestry["distillation_cache"])
    cache_path = root / cast(str, cache_spec["metadata"])
    if sha256_file(cache_path) != cache_spec["metadata_sha256"]:
        raise ValueError("warm-parent distillation metadata differs from the frozen recipe")
    if (
        candidate_provenance["parent"].get("distillation_cache", {}).get("metadata_sha256")
        != cache_spec["metadata_sha256"]
    ):
        raise ValueError("warm-parent provenance does not reference the frozen distillation cache")
    cache_metadata = json.loads(cache_path.read_text(encoding="utf-8"))
    expected_teachers = cast(list[dict[str, object]], cache_spec["teachers"])
    observed_teachers = [
        {
            "run": row["run"],
            "epoch": row["epoch"],
            "coefficient": row["coefficient"],
            "checkpoint_sha256": row["checkpoint_sha256"],
        }
        for row in cache_metadata["sources"]
    ]
    if observed_teachers != [
        {key: teacher[key] for key in ("run", "epoch", "coefficient", "checkpoint_sha256")}
        for teacher in expected_teachers
    ]:
        raise ValueError("distillation teacher identities differ from the frozen recipe")
    teacher_receipts: list[dict[str, object]] = []
    for teacher in expected_teachers:
        teacher_run = cast(str, teacher["run"])
        teacher_provenance, teacher_receipt = verify_run_membership(
            root,
            split,
            groups,
            split_sha,
            cast(str, teacher["provenance"]),
            cast(str, teacher["provenance_sha256"]),
            teacher_run,
            (f"model_ep{teacher['epoch']}.pt", cast(str, teacher["checkpoint_sha256"])),
        )
        teacher_receipts.append(teacher_receipt)

    return {
        "status": "pass",
        "fit_validation_membership_sha256": recipe["fit_membership_identity"]["valid"][
            "ordered_int64_le_sha256"
        ],
        "scored_groups_excluded_from_candidate_and_teacher_fit": True,
        "candidate_runs": candidate_receipts,
        "warm_parent_initializer": initializer_receipt,
        "teacher_runs": teacher_receipts,
        "distillation_cache_metadata_sha256": cache_spec["metadata_sha256"],
    }


def row_group_components(
    split: dict[str, object], groups: list[int]
) -> tuple[list[int], list[int]]:
    """Return source-qualified components and sequence indices in evaluation order."""
    sequence_splits = cast(dict[str, list[dict[str, int]]], split["sequences"])
    sequences = sequence_splits["valid"]
    by_group = {int(row["row_group"]): row for row in sequences}
    if any(group not in by_group for group in groups):
        raise ValueError("split is missing a sequence record for an evaluation group")
    return [int(by_group[group]["component"]) for group in groups], [
        int(by_group[group]["seq_ix"]) for group in groups
    ]


def grouped_folds(components: list[int]) -> np.ndarray:
    """Assign whole source components to four deterministic, balanced folds."""
    unique = np.array(sorted(set(components)), dtype=np.int64)
    shuffled = np.random.default_rng(FOLD_SEED).permutation(unique)
    component_fold = {
        int(component): fold
        for fold, block in enumerate(np.array_split(shuffled, N_FOLDS))
        for component in block
    }
    return np.asarray([component_fold[component] for component in components], dtype=np.int8)


def output_weights(scores: np.ndarray) -> np.ndarray:
    """Convert predicted per-sequence expert scores to a shrunken simplex gate."""
    logits = np.asarray(scores, dtype=np.float64) / SOFTMAX_TEMPERATURE
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return UNIFORM_SHRINKAGE / scores.shape[1] + (1.0 - UNIFORM_SHRINKAGE) * probabilities


def fit_ridge(
    features: np.ndarray, targets: np.ndarray, train: np.ndarray, test: np.ndarray
) -> np.ndarray:
    """Fit a standardized multi-output ridge map and predict expert utility."""
    mean = features[train].mean(axis=0)
    scale = np.maximum(features[train].std(axis=0), 1e-6)
    x_train = (features[train] - mean) / scale
    x_test = (features[test] - mean) / scale
    design_train = np.column_stack((np.ones(len(train)), x_train))
    design_test = np.column_stack((np.ones(len(test)), x_test))
    penalty = np.eye(design_train.shape[1], dtype=np.float64) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design_train.T @ design_train + penalty, design_train.T @ targets[train])
    return design_test @ beta


def get_target(cache: dict[str, np.memmap], index: int) -> tuple[np.ndarray, np.ndarray]:
    """Read one sequence's cached target and apply the frozen warmup mask."""
    target = np.asarray(cache["y"][index], dtype=np.float32)
    selection = np.asarray(cache["mask"][index], dtype=bool).copy()
    selection[:WARMUP] = False
    if target.shape != (20_000, 2) or selection.shape != (20_000,):
        raise ValueError(f"target-cache sequence {index} has an unexpected shape")
    return target, selection


def one_score(target: np.ndarray, prediction: np.ndarray, selection: np.ndarray) -> float:
    """Compute official pooled WP on one sequence using the production accumulator."""
    from wnn.gmetric import GlobalAccumulator

    accumulator = GlobalAccumulator()
    accumulator.add(target[None], prediction[None], selection[None])
    return float(accumulator.result()["wp"])


def predictions_for_group(arrays: dict[str, PredictionView], index: int) -> np.ndarray:
    """Materialize one small group from each memory-mapped expert array."""
    values = np.stack([array[slice(index, index + 1)][0] for array in arrays.values()])
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite cached predictions at F0 evaluation index {index}")
    return values


def main_grade(
    root: Path,
    features_path: Path,
    features_receipt_path: Path,
    preflight_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Cross-fit an input-only expert gate and independently score held-out components."""
    root = root.resolve(strict=True)
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    candidate_specs = cast(dict[str, dict[str, object]], recipe["candidate_experts"])
    split_path = root / SPLIT_REL
    split_sha = sha256_file(split_path)
    if split_sha != recipe["split_manifest_sha256"]:
        raise ValueError("F0 split differs from the frozen gate recipe")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    groups = [int(value) for value in split["evaluation"]["valid"]]
    components, sequence_ids = row_group_components(split, groups)
    features_receipt = json.loads(features_receipt_path.read_text(encoding="utf-8"))
    if features_receipt["input"]["targets_or_masks_read"] is not False:
        raise ValueError("warmup feature extraction is not label-blind")
    if features_receipt["membership"]["row_groups"] != groups:
        raise ValueError("warmup feature groups differ from the frozen F0 order")
    if features_receipt["membership"]["sequence_ids"] != sequence_ids:
        raise ValueError("warmup feature sequence identities differ from the frozen F0 order")
    if features_receipt["source_sha256"] != sha256_file(PACKET / "extract_warmup_features.py"):
        raise ValueError("feature receipt was produced by different extractor source")
    if features_receipt.get("recipe_sha256") != sha256_file(RECIPE_PATH):
        raise ValueError("warmup features were extracted under a different frozen recipe")
    expected_input = recipe["valid_input_source"]
    if (
        features_receipt["input"]["size_bytes"] != expected_input["size_bytes"]
        or features_receipt["input"]["footer_metadata_bytes"]
        != expected_input["footer_metadata_bytes"]
        or features_receipt["input"]["row_groups"] != expected_input["row_groups"]
        or features_receipt["input"]["warmup_slice_sha256"] != expected_input["warmup_slice_sha256"]
        or expected_input["warmup_slice_sha256"] is None
    ):
        raise ValueError("warmup input data identity differs from the frozen recipe")
    if features_receipt["features"]["sha256"] != sha256_file(features_path):
        raise ValueError("warmup feature bytes differ from their target-free receipt")
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if features.shape != (len(groups), 39) or not np.isfinite(features).all():
        raise ValueError("warmup feature artifact shape or values differ")

    sys.path.insert(0, str(root / "src"))
    grade = load_module("warmup_gate_prediction_bank", root / "scripts/grade_prediction_bank.py")
    campaign = load_module(
        "warmup_gate_campaign_score", root / "research/campaign_2026-09-30/audit/campaign_score.py"
    )
    from wnn.evaluation import paired_component_bootstrap, paired_component_moments
    from wnn.gmetric import GlobalAccumulator

    target_path = root / TARGET_REL
    if sha256_file(target_path) != recipe["target_cache_sha256"]:
        raise ValueError("target cache differs from the frozen F0 recipe")
    target_cache = cast(dict[str, np.memmap], campaign.mmap_stored_npz(target_path))
    cached_groups = [int(value) for value in target_cache["row_group"]]
    if cached_groups != list(range(1_873)):
        raise ValueError("target-cache row-group index does not match the frozen validation file")
    target_positions = {group: index for index, group in enumerate(cached_groups)}
    if any(group not in target_positions for group in groups):
        raise ValueError("target cache lacks one or more frozen F0 groups")
    target_indices = [target_positions[group] for group in groups]
    arrays, prediction_receipts = load_prediction_bank(
        root, groups, split_sha, grade, candidate_specs
    )
    ancestry_receipt = verify_fit_and_teacher_ancestry(root, recipe, split, groups, split_sha)
    preflight = write_preflight(
        root,
        groups,
        sequence_ids,
        split_sha,
        target_path,
        features_path,
        features_receipt_path,
        prediction_receipts,
        candidate_specs,
        ancestry_receipt,
        preflight_path,
    )
    folds = grouped_folds(components)
    expert_names = list(candidate_specs)
    sequence_scores = np.full((len(groups), len(expert_names)), np.nan, dtype=np.float64)
    fold_assignments: list[dict[str, object]] = []
    gate_weights = np.zeros((len(groups), len(expert_names)), dtype=np.float64)
    gate_component_moments = np.zeros((len(groups), 2, 6), dtype=np.float64)
    fixed_component_moments = np.zeros_like(gate_component_moments)
    uniform_component_moments = np.zeros_like(gate_component_moments)
    pooled_gate = GlobalAccumulator()
    pooled_fixed = GlobalAccumulator()
    pooled_uniform = GlobalAccumulator()

    for fold in range(N_FOLDS):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        training_accumulators = [GlobalAccumulator() for _ in expert_names]
        for index in train:
            target, selection = get_target(target_cache, target_indices[int(index)])
            group_predictions = predictions_for_group(arrays, int(index))
            for expert_index, prediction in enumerate(group_predictions):
                sequence_scores[index, expert_index] = one_score(target, prediction, selection)
                training_accumulators[expert_index].add(
                    target[None], prediction[None], selection[None]
                )
        training_scores = np.asarray(
            [float(accumulator.result()["wp"]) for accumulator in training_accumulators]
        )
        best_expert = int(np.argmax(training_scores))
        predicted_scores = fit_ridge(features, sequence_scores, train, test)
        gate_weights[test] = output_weights(predicted_scores)

        fold_gate = GlobalAccumulator()
        fold_fixed = GlobalAccumulator()
        fold_uniform = GlobalAccumulator()
        fold_mean_weights = gate_weights[test].mean(axis=0)
        weight_dispersion = float(np.abs(gate_weights[test] - fold_mean_weights).sum(axis=1).mean())
        for index in test:
            local_index = int(index)
            target, selection = get_target(target_cache, target_indices[local_index])
            group_predictions = predictions_for_group(arrays, local_index)
            gate_prediction = np.einsum("k,ktd->td", gate_weights[local_index], group_predictions)
            fixed_prediction = group_predictions[best_expert]
            uniform_prediction = group_predictions.mean(axis=0)
            fold_gate.add(target[None], gate_prediction[None], selection[None])
            fold_fixed.add(target[None], fixed_prediction[None], selection[None])
            fold_uniform.add(target[None], uniform_prediction[None], selection[None])
            pooled_gate.add(target[None], gate_prediction[None], selection[None])
            pooled_fixed.add(target[None], fixed_prediction[None], selection[None])
            pooled_uniform.add(target[None], uniform_prediction[None], selection[None])
            if selection.any():
                gate_moments, fixed_moments = paired_component_moments(
                    target[None], gate_prediction[None], fixed_prediction[None], selection[None]
                )
                _, uniform_moments = paired_component_moments(
                    target[None], gate_prediction[None], uniform_prediction[None], selection[None]
                )
                # Candidate moments are identical in both paired comparisons.
                gate_component_moments[local_index] = gate_moments[0]
                fixed_component_moments[local_index] = fixed_moments[0]
                uniform_component_moments[local_index] = uniform_moments[0]

        fold_assignments.append(
            {
                "fold": fold,
                "training_components": len(set(np.asarray(components)[train].tolist())),
                "heldout_components": len(set(np.asarray(components)[test].tolist())),
                "training_groups": len(train),
                "heldout_groups": len(test),
                "fixed_expert_selected_from_training_only": expert_names[best_expert],
                "training_fixed_expert_wp": float(training_scores[best_expert]),
                "heldout_scores": {
                    "gate": float(fold_gate.result()["wp"]),
                    "crossfit_fixed": float(fold_fixed.result()["wp"]),
                    "uniform_fixed": float(fold_uniform.result()["wp"]),
                },
                "mean_test_gate_weights": gate_weights[test].mean(axis=0).tolist(),
                "mean_group_weight_l1_from_fold_mean": weight_dispersion,
            }
        )

    gate_bootstrap = paired_component_bootstrap(
        gate_component_moments,
        fixed_component_moments,
        np.asarray(components, dtype=np.int64),
        n_boot=N_BOOT,
        seed=BOOT_SEED,
        centered=True,
    )
    uniform_bootstrap = paired_component_bootstrap(
        gate_component_moments,
        uniform_component_moments,
        np.asarray(components, dtype=np.int64),
        n_boot=N_BOOT,
        seed=BOOT_SEED,
        centered=True,
    )
    result: dict[str, object] = {
        "schema": "wnn.warmup_expert_gate_screen.result.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "hypothesis_id": recipe["hypothesis_id"],
        "experiment_id": recipe["experiment_id"],
        "recipe_sha256": sha256_file(RECIPE_PATH),
        "feature_receipt_sha256": sha256_file(features_receipt_path),
        "feature_sha256": sha256_file(features_path),
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "preflight_fingerprint": preflight["fingerprint"],
        "split_manifest_sha256": split_sha,
        "target_cache_sha256": sha256_file(target_path),
        "metric_id": "global_wp_clip2_v1",
        "fold": 0,
        "evaluation_group_order_sha256": sha256_ints(groups),
        "evaluation_sequence_id_order_sha256": sha256_ints(sequence_ids),
        "evaluation_component_order_sha256": sha256_ints(components),
        "exposure": "repeatedly inspected F0; component-cross-fitted gate is a screen, not fresh validation",
        "gate_target": "per-sequence official WP of each expert, used only for gate-training components",
        "candidate_ancestry": "four fold0 endpoints exclude the fixed F0 evaluation groups; TL3 and adaptive alpha aliases are excluded",
        "candidate_predictions": prediction_receipts,
        "source_hashes": {
            "gate_grader": sha256_file(Path(__file__).resolve()),
            "feature_extractor": features_receipt["source_sha256"],
            "official_metric": sha256_file(root / "src/wnn/gmetric.py"),
            "component_evaluation": sha256_file(root / "src/wnn/evaluation.py"),
            "target_cache_reader": sha256_file(
                root / "research/campaign_2026-09-30/audit/campaign_score.py"
            ),
            "prediction_aligner": sha256_file(root / "scripts/grade_prediction_bank.py"),
        },
        "split": {
            "method": "deterministic permutation of sorted source-qualified components followed by np.array_split",
            "seed": FOLD_SEED,
            "folds": N_FOLDS,
            "assignments": fold_assignments,
        },
        "gate": {
            "regression": "standardized multi-output ridge",
            "alpha": RIDGE_ALPHA,
            "temperature": SOFTMAX_TEMPERATURE,
            "uniform_shrinkage": UNIFORM_SHRINKAGE,
            "mean_weights": gate_weights.mean(axis=0).tolist(),
            "std_weights": gate_weights.std(axis=0).tolist(),
            "input_dependence": {
                "definition": "per-fold mean L1 distance of each held-out group's gate weights from that fold's mean weights",
                "fold_values": [
                    float(row["mean_group_weight_l1_from_fold_mean"]) for row in fold_assignments
                ],
                "folds_at_or_above_0p10": sum(
                    float(row["mean_group_weight_l1_from_fold_mean"]) >= 0.10
                    for row in fold_assignments
                ),
            },
        },
        "pooled_crossfit_scores": {
            "gate": pooled_gate.result(),
            "training_selected_fixed": pooled_fixed.result(),
            "uniform_fixed": pooled_uniform.result(),
        },
        "component_bootstrap": {
            "scope": "paired source-component resampling of the assembled cross-fit predictions; conditional on fitted gate folds",
            "gate_minus_training_selected_fixed": gate_bootstrap,
            "gate_minus_uniform_fixed": uniform_bootstrap,
        },
        "limitations": [
            "All F0 endpoints and labels have prior exposure; cross-fitting prevents gate train/test label reuse only within this screen.",
            "The gate learns local per-sequence WP as a surrogate target, then is judged by the exact pooled official metric.",
            "Bootstrap uncertainty conditions on the four fitted gate models and does not cover fitting-seed or unseen-regime variation.",
            "A positive result needs clean fold/seed confirmation and does not imply a public score increase.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main() -> None:
    """Run the frozen component-grouped gate grade on cached artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-receipt", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = main_grade(
        args.root,
        args.features,
        args.features_receipt,
        args.preflight_output,
        args.output,
    )
    scores = cast(dict[str, dict[str, float]], result["pooled_crossfit_scores"])
    print(
        json.dumps(
            {
                "status": "graded",
                "gate_wp": scores["gate"]["wp"],
                "training_selected_fixed_wp": scores["training_selected_fixed"]["wp"],
                "uniform_fixed_wp": scores["uniform_fixed"]["wp"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
