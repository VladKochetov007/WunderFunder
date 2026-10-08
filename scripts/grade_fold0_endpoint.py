#!/usr/bin/env python3
"""Independently grade fixed equal blends on the exposed fold-0 component panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Prediction sidecars and provenance records have source-defined JSON schemas.
from typing import Any

import numpy as np
from audit_fold0_membership import inspect_run

ROWS = 20_000
WARMUP = 99
CLIP = 2.0
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 271_828


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_moments(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    y = np.clip(np.asarray(target, dtype=np.float64), -CLIP, CLIP)
    p = np.clip(np.asarray(prediction, dtype=np.float64), -CLIP, CLIP)
    selected = np.asarray(mask, dtype=bool)
    if not np.isfinite(y[selected]).all() or not np.isfinite(p[selected]).all():
        raise ValueError("selected target or prediction is non-finite")
    y, p = y[selected], p[selected]
    weight = np.abs(y)
    total = float(weight.sum(dtype=np.float64))
    if total < 1e-8:
        return np.zeros(6, dtype=np.float64)
    mean_y = float(np.sum(weight * y, dtype=np.float64) / total)
    mean_p = float(np.sum(weight * p, dtype=np.float64) / total)
    yc, pc = y - mean_y, p - mean_p
    return np.asarray(
        [
            total,
            mean_y,
            mean_p,
            np.sum(weight * yc * yc, dtype=np.float64),
            np.sum(weight * pc * pc, dtype=np.float64),
            np.sum(weight * yc * pc, dtype=np.float64),
        ],
        dtype=np.float64,
    )


def merge_component_moments(parts: np.ndarray, counts: np.ndarray | None = None) -> float:
    if counts is None:
        counts = np.ones(len(parts), dtype=np.float64)
    weighted = np.asarray(counts, dtype=np.float64) * parts[:, 0]
    total = float(weighted.sum())
    if total < 1e-8:
        return 0.0
    mean_y = float(np.sum(weighted * parts[:, 1]) / total)
    mean_p = float(np.sum(weighted * parts[:, 2]) / total)
    dy, dp = parts[:, 1] - mean_y, parts[:, 2] - mean_p
    variance_y = float(np.sum(counts * (parts[:, 3] + parts[:, 0] * dy * dy)))
    variance_p = float(np.sum(counts * (parts[:, 4] + parts[:, 0] * dp * dp)))
    covariance = float(np.sum(counts * (parts[:, 5] + parts[:, 0] * dy * dp)))
    if variance_y <= 1e-16 or variance_p <= 1e-16:
        return 0.0
    return float(np.clip(covariance / np.sqrt(variance_y * variance_p), -1.0, 1.0))


def endpoint_moments(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
    components: np.ndarray,
) -> list[np.ndarray]:
    output = []
    for target_index in range(2):
        blocks = []
        for component in np.unique(components):
            sequence_indices = np.flatnonzero(components == component)
            blocks.append(
                weighted_moments(
                    targets[sequence_indices, :, target_index],
                    predictions[sequence_indices, :, target_index],
                    mask[sequence_indices],
                )
            )
        output.append(np.stack(blocks))
    return output


def endpoint_score(
    parts: list[np.ndarray], counts: np.ndarray | None = None
) -> tuple[float, list[float]]:
    per_target = [merge_component_moments(value, counts) for value in parts]
    return 0.5 * (per_target[0] + per_target[1]), per_target


def checkpoint_and_prediction(
    root: Path,
    split: dict[str, Any],
    run_at: str,
    row_groups: list[int],
    split_hash: str,
    evaluation_indices: np.ndarray,
    expected_shape: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    run, marker, epoch_text = run_at.partition("@ep")
    if not marker or not epoch_text.isdigit():
        raise ValueError(f"source must be RUN@epN: {run_at}")
    run_root = root / "artifacts/runs" / run
    fit = inspect_run(root, run, split, split_hash)
    checkpoint = run_root / f"model_ep{epoch_text}.pt"
    checkpoint_hash = sha256(checkpoint)
    prediction_path = root / "cache" / f"pred_{run}_valid.npy"
    sidecar_path = prediction_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    row_groups_sha = hashlib.sha256(np.asarray(row_groups, dtype="<i8").tobytes()).hexdigest()
    if (
        sidecar.get("run") != run
        or sidecar.get("split") != "valid"
        or sidecar.get("epoch") != int(epoch_text)
        or sidecar.get("checkpoint_sha256") != checkpoint_hash
        or sidecar.get("split_manifest_sha256") != split_hash
        or sidecar.get("row_groups") != row_groups
    ):
        raise ValueError(
            f"{run_at}: prediction sidecar disagrees with frozen source/checkpoint/order"
        )
    prediction_hash = sha256(prediction_path)
    if sidecar.get("prediction_sha256") != prediction_hash:
        raise ValueError(f"{run_at}: prediction bytes differ from sidecar hash")
    prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
    if prediction.shape != expected_shape or prediction.dtype != np.float32:
        raise ValueError(f"{run_at}: prediction array shape/dtype mismatch")
    selected = np.asarray(prediction[evaluation_indices], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError(f"{run_at}: selected evaluation predictions are non-finite")
    binding = {
        "run_at": run_at,
        "checkpoint_sha256": checkpoint_hash,
        "prediction_sha256": prediction_hash,
        "prediction_sidecar_sha256": sha256(sidecar_path),
        "split_manifest_sha256": split_hash,
        "row_groups_sha256": row_groups_sha,
        "fit_membership_status": fit["status"],
        "inference_precision": sidecar.get("inference_precision"),
    }
    return selected, binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--candidate-core", required=True, help="RUN@epN")
    parser.add_argument("--candidate-head", required=True, help="RUN@epN")
    parser.add_argument("--control-parent", required=True, help="RUN@epN")
    parser.add_argument("--control-head", required=True, help="RUN@epN")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    split_path = args.split.resolve()
    split_bytes = split_path.read_bytes()
    split = json.loads(split_bytes)
    split_hash = hashlib.sha256(split_bytes).hexdigest()
    if int(split["fold"]) != 0:
        raise ValueError("expected the exposed fold-0 manifest")
    eval_groups = list(map(int, split["evaluation"]["valid"]))
    if len(eval_groups) != 468 or len(set(eval_groups)) != 468:
        raise ValueError("expected 468 unique fold-0 evaluation groups")
    component_map = {
        (source, int(node["row_group"])): int(node["component"])
        for source, nodes in split["sequences"].items()
        for node in nodes
    }
    components = np.asarray(
        [component_map[("valid", group)] for group in eval_groups], dtype=np.int64
    )
    component_ids = np.unique(components)
    if len(component_ids) != 141:
        raise ValueError("expected 141 connected evaluation components")

    target_path = root / "cache/valid_targets.npz"
    with np.load(target_path, allow_pickle=False) as cache:
        row_groups = list(map(int, cache["row_group"]))
        if len(row_groups) != len(set(row_groups)):
            raise ValueError("valid target cache repeats row-group identities")
        position = {group: index for index, group in enumerate(row_groups)}
        if not set(eval_groups).issubset(position):
            raise ValueError("target cache omits evaluation groups")
        indices = np.asarray([position[group] for group in eval_groups], dtype=np.int64)
        targets = np.asarray(cache["y"][indices], dtype=np.float32)
        is_scored = np.asarray(cache["mask"][indices], dtype=bool)
    mask = is_scored & (np.arange(ROWS) >= WARMUP)[None, :]
    if targets.shape != (468, ROWS, 2) or mask.shape != (468, ROWS):
        raise ValueError("target or score mask has unexpected fold-0 shape")

    inputs = [
        ("candidate_core", args.candidate_core),
        ("candidate_head", args.candidate_head),
        ("control_parent", args.control_parent),
        ("control_head", args.control_head),
    ]
    predictions: dict[str, np.ndarray] = {}
    bindings = {}
    for label, run_at in inputs:
        predictions[label], bindings[label] = checkpoint_and_prediction(
            root, split, run_at, row_groups, split_hash, indices, (len(row_groups), ROWS, 2)
        )
    candidate = (
        (
            predictions["candidate_core"].astype(np.float64)
            + predictions["candidate_head"].astype(np.float64)
        )
        * 0.5
    ).astype(np.float32)
    control = (
        (
            predictions["control_parent"].astype(np.float64)
            + predictions["control_head"].astype(np.float64)
        )
        * 0.5
    ).astype(np.float32)
    candidate_parts = endpoint_moments(targets, candidate, mask, components)
    control_parts = endpoint_moments(targets, control, mask, components)
    candidate_wp, candidate_by_target = endpoint_score(candidate_parts)
    control_wp, control_by_target = endpoint_score(control_parts)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(component_ids), size=len(component_ids))
        counts = np.bincount(draw, minlength=len(component_ids)).astype(np.float64)
        candidate_boot, _ = endpoint_score(candidate_parts, counts)
        control_boot, _ = endpoint_score(control_parts, counts)
        deltas[replicate] = candidate_boot - control_boot
    low, median, high = np.quantile(deltas, [0.025, 0.5, 0.975])

    report = {
        "schema": "wnn.independent-fold0-endpoint-grade.v1",
        "status": "graded_exposed_discovery_panel",
        "metric_id": "global_wp_clip2_v1",
        "split_manifest_sha256": split_hash,
        "targets_sha256": sha256(target_path),
        "evaluation_groups": len(eval_groups),
        "evaluation_components": len(component_ids),
        "selected_rows": int(mask.sum()),
        "candidate": {
            "endpoint": "0.5 * candidate core + 0.5 * head fine-tune",
            "members": [args.candidate_core, args.candidate_head],
            "member_bindings": [bindings["candidate_core"], bindings["candidate_head"]],
            "t0_wp": candidate_by_target[0],
            "t1_wp": candidate_by_target[1],
            "wp": candidate_wp,
        },
        "control": {
            "endpoint": "0.5 * matched parent + 0.5 * parent head fine-tune",
            "members": [args.control_parent, args.control_head],
            "member_bindings": [bindings["control_parent"], bindings["control_head"]],
            "t0_wp": control_by_target[0],
            "t1_wp": control_by_target[1],
            "wp": control_wp,
        },
        "paired_component_bootstrap": {
            "unit": "connected feature-only successor component",
            "components": len(component_ids),
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "delta_candidate_minus_control": candidate_wp - control_wp,
            "percentile_95_interval": [float(low), float(high)],
            "bootstrap_median": float(median),
            "probability_delta_positive_in_bootstrap": float(np.mean(deltas > 0)),
            "scope": "fixed predictions; does not include training-seed or public-transfer uncertainty",
        },
        "exposure": "fold0 evaluation groups have been repeatedly inspected; fold1 robustness remains exposed and is not pristine confirmation",
        "claim_limits": {
            "public_forecast": False,
            "heldout_claim": False,
            "public_upload": False,
        },
        "independent_implementation": {
            "equation": "pool clipped targets/predictions, use weight abs(clipped target), compute weighted Pearson per target, then average t0/t1",
            "does_not_call_score_fold0_panel": True,
            "component_resampling_uses_centered_moments": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "wp_candidate": candidate_wp,
                "wp_control": control_wp,
                "delta": candidate_wp - control_wp,
                "ci95": [float(low), float(high)],
            }
        )
    )


if __name__ == "__main__":
    main()
