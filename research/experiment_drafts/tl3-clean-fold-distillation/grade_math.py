"""Canonical scoring and paired component uncertainty for the TL3 screen."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any  # Any covers canonical split, metric and moment adapters.

import numpy as np


def equal_cell_target_means(folds: dict[str, Any], cells: tuple[str, ...]) -> dict[str, float]:
    """Average per-target paired deltas with equal weight for every fixed cell."""
    return {
        target: float(
            np.mean([folds[cell]["full_mixture_kd_minus_labels"][target] for cell in cells])
        )
        for target in ("t0", "t1")
    }


def fit_pair_gate(records: dict[str, dict[str, Any]], cell_id: str) -> None:
    """Require paired students to differ only in prescribed distillation alpha."""
    labels = records[f"tl3kd_r1_{cell_id.lower()}_labels"]
    kd = records[f"tl3kd_r1_{cell_id.lower()}_kd"]
    fields = (
        "runtime_sha256",
        "initial_state_sha256",
        "normalization_identity",
        "sample_order_identities",
    )
    if any(labels[key] != kd[key] for key in fields):
        raise ValueError(f"{cell_id} label/KD source, state, normalization or order differs")
    if labels["teacher_targets"] != kd["teacher_targets"]:
        raise ValueError(f"{cell_id} label/KD do not share the exact cached teacher mean")
    left, right = dict(labels["training_args"]), dict(kd["training_args"])
    for args in (left, right):
        args.pop("name", None)
        args.pop("distill_alpha", None)
        args.pop("resume_state", None)  # Canonical full-state continuation is audited separately.
    if left != right:
        raise ValueError(f"{cell_id} label/KD differ beyond alpha")


def runtime_gate(records: dict[str, dict[str, Any]]) -> None:
    """Permit distinct B300s across cells while retaining each cell's fixed runtime."""
    for cell in ("F0", "F2"):
        runtimes = {
            record["runtime_sha256"] for record in records.values() if record["cell"] == cell
        }
        if len(runtimes) != 1:
            raise ValueError(f"{cell} changed its verified runtime/device identity")


def component_bootstrap(
    candidate_rows: np.ndarray,
    reference_rows: np.ndarray,
    components: np.ndarray,
    seed: int,
    replicates: int,
    merge_moments: Any,
    score_moments: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample whole source components using the canonical centered-WP primitives."""
    identifiers, inverse = np.unique(components, return_inverse=True)
    candidate = np.zeros((len(identifiers), 2, 6), dtype=np.float64)
    reference = np.zeros_like(candidate)
    for row, component_index in enumerate(inverse):
        for target in range(2):
            candidate[component_index, target] = merge_moments(
                candidate[component_index, target], candidate_rows[row, target]
            )
            reference[component_index, target] = merge_moments(
                reference[component_index, target], reference_rows[row, target]
            )
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(identifiers), len(identifiers))
        counts = np.bincount(selected, minlength=len(identifiers))
        deltas[index] = score_moments(candidate, counts) - score_moments(reference, counts)
    return deltas, {
        "method": "paired_whole_component_bootstrap_conditional_on_fixed_predictions",
        "interval": "percentile_95",
        "replicates": replicates,
        "seed": seed,
        "components": len(identifiers),
        "ci95": np.percentile(deltas, [2.5, 97.5]).tolist(),
        "bootstrap_sd": float(deltas.std(ddof=1)),
        "p_le_0": float(np.mean(deltas <= 0)),
        "metric_id": "global_wp_clip2_v1",
    }


def score_cell(
    cell_id: str,
    split: Any,
    records: dict[str, dict[str, Any]],
    weights: dict[str, float],
    parquet: Any,
    warmup: int,
    accumulator_type: Any,
    paired_moments: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Stream one cell's official scores and paired full-mixture moments."""
    cell = {record["role"]: record for record in records.values() if record["cell"] == cell_id}
    teachers = ("teacher_a", "teacher_b")
    predictions = {
        role: np.load(record["prediction_path"], mmap_mode="r", allow_pickle=False)
        for role, record in cell.items()
    }
    for role, array in predictions.items():
        if array.shape != (len(split.evaluation_groups), 20_000, 2) or array.dtype != np.float32:
            raise ValueError(f"{cell_id}/{role} prediction has wrong shape or dtype")
    component_map = {
        int(group): int(component)
        for source, group, component in split.components
        if source == "valid"
    }
    components = np.asarray(
        [component_map[int(g)] for g in split.evaluation_groups], dtype=np.int64
    )
    roles = (*teachers, "teacher_only", "labels", "kd", "mix_labels", "mix_kd")
    accumulators = {role: accumulator_type() for role in roles}
    candidate_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    for index, group in enumerate(split.evaluation_groups):
        if group < 0 or group >= parquet.num_row_groups:
            raise ValueError(f"{cell_id} held-out row group is outside valid.parquet")
        table = parquet.read_row_group(
            int(group),
            columns=["seq_ix", "step_in_seq", "need_prediction", "t0", "t1", "is_scored"],
            use_threads=False,
        )
        if table.num_rows != 20_000:
            raise ValueError(f"{cell_id} row group {group} is not a full sequence")
        sequence_ids = table["seq_ix"].to_numpy(zero_copy_only=False)
        if np.any(sequence_ids != sequence_ids[0]):
            raise ValueError(f"{cell_id} row group {group} contains multiple sequences")
        steps = table["step_in_seq"].to_numpy(zero_copy_only=False)
        if not np.array_equal(steps, np.arange(table.num_rows)):
            raise ValueError(f"{cell_id} row group {group} has unexpected step ordering")
        targets = np.column_stack(
            (
                table["t0"].to_numpy(zero_copy_only=False),
                table["t1"].to_numpy(zero_copy_only=False),
            )
        ).astype(np.float32, copy=False)
        need = table["need_prediction"].to_numpy(zero_copy_only=False).astype(bool, copy=False)
        if not np.array_equal(need, steps >= warmup):
            raise ValueError(f"{cell_id} row group {group} violates the official prediction mask")
        scored = table["is_scored"].to_numpy(zero_copy_only=False).astype(bool, copy=False)
        selected = need & scored
        if not selected.any():
            raise ValueError(f"{cell_id} row group {group} has no scored rows")
        row = {
            role: np.asarray(array[index], dtype=np.float32) for role, array in predictions.items()
        }
        teacher_only = np.zeros_like(row[teachers[0]], dtype=np.float32)
        for role in teachers:
            teacher_only += np.float32(1.0 / len(teachers)) * row[role]
        row["teacher_only"] = teacher_only
        for arm in ("labels", "kd"):
            mixture = np.zeros_like(row[arm], dtype=np.float32)
            mixture += np.float32(weights["student"]) * row[arm]
            for role in teachers:
                mixture += np.float32(weights[role]) * row[role]
            row[f"mix_{arm}"] = mixture
        for role, accumulator in accumulators.items():
            accumulator.add(targets, row[role], selected)
        candidate, reference = paired_moments(
            targets[None], row["mix_kd"][None], row["mix_labels"][None], selected[None]
        )
        candidate_rows.append(candidate[0])
        reference_rows.append(reference[0])
    scores = {role: accumulator.result() for role, accumulator in accumulators.items()}
    delta = {
        key: float(scores["mix_kd"][key] - scores["mix_labels"][key]) for key in ("t0", "t1", "wp")
    }
    result = {
        "fold": split.fold,
        "evaluation_groups": len(split.evaluation_groups),
        "components": int(len(np.unique(components))),
        "scores": scores,
        "standalone_student_minus_teacher_only": {
            arm: {
                key: float(scores[arm][key] - scores["teacher_only"][key])
                for key in ("t0", "t1", "wp")
            }
            for arm in ("labels", "kd")
        },
        "full_mixture_minus_teacher_only": {
            arm: {
                key: float(scores[f"mix_{arm}"][key] - scores["teacher_only"][key])
                for key in ("t0", "t1", "wp")
            }
            for arm in ("labels", "kd")
        },
        "full_mixture_kd_minus_labels": delta,
    }
    return result, np.asarray(candidate_rows), np.asarray(reference_rows), components
