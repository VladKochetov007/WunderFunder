from __future__ import annotations

import hashlib
import json
import logging
import traceback

import numpy as np
import pyarrow.parquet as pq
from continuation_common import (
    MODELS,
    OUT,
    ROOT,
    calibration_design,
    causal_design,
    read_sequence,
    sample_ids,
    sufficient_stats,
    threshold_states,
    weighted_pearson,
)

LOGGER = logging.getLogger(__name__)


def update_state(phase: str, **fields: object) -> None:
    """Persist progress so an interrupted bounded command is reviewable."""
    path = OUT / "STATE.json"
    state = json.loads(path.read_text())
    state.update(phase=phase, **fields)
    path.write_text(json.dumps(state, indent=2) + "\n")


def fit_ridge(parquet: pq.ParquetFile, groups: np.ndarray) -> dict[str, np.ndarray]:
    """Stream original train sample; fit using only small sufficient matrices."""
    gram = np.zeros((2, 244, 244))
    rhs = np.zeros((2, 244))
    totals = np.zeros(243)
    squares = np.zeros(243)
    count = 0
    provenance = []
    for index, group in enumerate(groups):
        values, targets, required, record = read_sequence(parquet, int(group), "train")
        design, _ = causal_design(values)
        keep = required & (np.arange(len(values)) % 4 == 0)
        features = design[keep].astype(np.float64)
        labels = np.clip(targets[keep].astype(np.float64), -2, 2)
        totals += features.sum(axis=0)
        squares += (features**2).sum(axis=0)
        count += len(features)
        matrix = np.column_stack([features, np.ones(len(features))])
        for target in range(2):
            weight = np.abs(labels[:, target]) + 1e-3
            gram[target] += matrix.T @ (matrix * weight[:, None])
            rhs[target] += matrix.T @ (weight * labels[:, target])
        provenance.append(record)
        if index % 8 == 0:
            LOGGER.info("ridge row group %d/%d", index + 1, len(groups))
    mean = totals / count
    scale = np.sqrt(np.maximum(squares / count - mean**2, 0)) + 1e-6
    transform = np.eye(244)
    transform[:243, :243] /= scale[None, :]
    transform[-1, :243] = -mean / scale
    penalty = np.eye(244) * (10 * count / 1000)
    penalty[-1, -1] = 0
    beta = []
    for target in range(2):
        normalized = transform.T @ gram[target] @ transform
        fitted = np.linalg.solve(normalized + penalty, transform.T @ rhs[target])
        beta.append(transform @ fitted)
    (OUT / "provenance_ridge.json").write_text(json.dumps(provenance, indent=2))
    result = {"beta": np.column_stack(beta), "mean": mean, "scale": scale}
    np.savez(OUT / "continuation_ridge.npz", **result)
    LOGGER.info("ridge fitted on %d rows; design width 243", count)
    return result


def collect_calibration(
    parquet: pq.ParquetFile, groups: np.ndarray, split: str, beta: np.ndarray
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Keep only five states, two predictions/targets, and provenance."""
    collected = {name: [] for name in ["states", "predictions", "targets", "groups"]}
    provenance = []
    for group in groups:
        values, targets, selected, record = read_sequence(parquet, int(group), split)
        design, states = causal_design(values)
        if split == "train":
            selected &= np.arange(len(values)) % 4 == 0
        predictions = design[selected].astype(np.float64) @ beta[:-1] + beta[-1]
        collected["states"].append(states[selected])
        collected["predictions"].append(predictions)
        collected["targets"].append(targets[selected])
        collected["groups"].append(np.full(selected.sum(), group, dtype=np.int32))
        provenance.append(record)
    return {name: np.concatenate(arrays) for name, arrays in collected.items()}, provenance


def fit_calibrators(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Learn source-only preprocessing and fixed-penalty matched regressions."""
    states = data["states"].astype(np.float64)
    mean = states.mean(axis=0)
    scale = states.std(axis=0) + 1e-6
    normalized = (states - mean) / scale
    thresholds = np.quantile(states, [1 / 3, 2 / 3], axis=0).T
    binned = threshold_states(states, thresholds)
    result = {"mean": mean, "scale": scale, "thresholds": thresholds}
    for model in MODELS[1:]:
        betas = []
        for target in range(2):
            features = binned if model == "threshold" else normalized
            matrix = calibration_design(data["predictions"][:, target], features, model)
            labels = np.clip(data["targets"][:, target].astype(np.float64), -2, 2)
            weight = np.abs(labels) + 1e-3
            penalty = np.eye(matrix.shape[1]) * (0.001 * len(labels))
            penalty[:2, :2] = 0
            beta = np.linalg.solve(
                matrix.T @ (matrix * weight[:, None]) + penalty,
                matrix.T @ (weight * labels),
            )
            betas.append(beta)
        result[model] = np.column_stack(betas)
    return result


def predict_calibrated(
    predictions: np.ndarray, states: np.ndarray, fitted: dict[str, np.ndarray]
) -> np.ndarray:
    """Apply fixed source statistics; no target or scoring mask inputs."""
    normalized = (states - fitted["mean"]) / fitted["scale"]
    binned = threshold_states(states, fitted["thresholds"])
    result = np.empty((len(predictions), len(MODELS), 2))
    result[:, 0] = predictions
    for index, model in enumerate(MODELS[1:], start=1):
        features = binned if model == "threshold" else normalized
        for target in range(2):
            matrix = calibration_design(predictions[:, target], features, model)
            result[:, index, target] = matrix @ fitted[model][:, target]
    return result


def main() -> None:
    """Run the frozen bounded probe, preserving every result and sample recipe."""
    frozen = json.loads((OUT / "HYPOTHESES.json").read_text())
    if frozen["status"] != "frozen_before_any_new_scores":
        raise ValueError("The original frozen specification must remain unchanged")
    train = pq.ParquetFile(ROOT / "datasets/train.parquet")
    valid = pq.ParquetFile(ROOT / "datasets/valid.parquet")
    ridge_ids = sample_ids(train.num_row_groups, 48, 2)
    dev_ids = sample_ids(valid.num_row_groups, 40, 1)
    old_train = set(ridge_ids) | set(sample_ids(train.num_row_groups, 300, 11)) | {0, 5}
    train_pool = [i for i in range(train.num_row_groups) if i not in old_train]
    calibration_ids = np.sort(np.random.default_rng(260103).choice(train_pool, 32, replace=False))
    stream_pool = [i for i in range(valid.num_row_groups) if i not in set(dev_ids)]
    old_stream = set(np.random.default_rng(7).choice(stream_pool, 600, replace=False))
    excluded = (
        set(dev_ids) | old_stream | set(sample_ids(valid.num_row_groups, 40, 0)) | {0, 5, 6, 7}
    )
    eval_pool = [i for i in range(valid.num_row_groups) if i not in excluded]
    eval_ids = np.sort(np.random.default_rng(260104).choice(eval_pool, 64, replace=False))
    sampling = {
        "ridge_train": ridge_ids.tolist(),
        "calibration_train": calibration_ids.tolist(),
        "calibration_exposed_valid": dev_ids.tolist(),
        "evaluation_valid": eval_ids.tolist(),
        "conservative_excluded_valid": sorted(int(i) for i in excluded),
        "exposure_caveat": "Disjoint from known samples and default stream600 reconstruction; undocumented actual stream invocation and campaign exposure unknown.",
        "dataset_metadata": {
            "train_row_groups": train.num_row_groups,
            "valid_row_groups": valid.num_row_groups,
        },
        "frozen_hypothesis_sha256": hashlib.sha256(
            (OUT / "HYPOTHESES.json").read_bytes()
        ).hexdigest(),
    }
    (OUT / "sampling_v1.json").write_text(json.dumps(sampling, indent=2) + "\n")
    update_state("fitting_streamed_ridge", inspected_sequence_ids=sampling)
    ridge = fit_ridge(train, ridge_ids)
    fitted = {}
    fingerprints = set()
    ridge_provenance = json.loads((OUT / "provenance_ridge.json").read_text())
    fingerprints.update(record["warmup_inputs_sha256"] for record in ridge_provenance)
    development_scores = {}
    for name, parquet, groups, split in [
        ("train", train, calibration_ids, "train"),
        ("exposed_scored_valid", valid, dev_ids, "valid"),
    ]:
        update_state(f"calibration_{name}")
        data, provenance = collect_calibration(parquet, groups, split, ridge["beta"])
        fingerprints.update(record["warmup_inputs_sha256"] for record in provenance)
        (OUT / f"provenance_calibration_{name}.json").write_text(json.dumps(provenance, indent=2))
        fitted[name] = fit_calibrators(data)
        np.savez(OUT / f"calibration_{name}_v1.npz", **fitted[name])
        development_scores[name] = [
            float(
                weighted_pearson(sufficient_stats(data["targets"][:, k], data["predictions"][:, k]))
            )
            for k in range(2)
        ]
        LOGGER.info(
            "%s calibration rows %d; base scored/sample WP %s",
            name,
            len(data["targets"]),
            development_scores[name],
        )
        del data
    update_state("evaluating_frozen_models")
    moments = np.zeros((64, 2, len(MODELS), 2, 6))
    provenance = []
    for index, group in enumerate(eval_ids):
        values, targets, selected, record = read_sequence(valid, int(group), "valid")
        record["excluded_exact_warmup_duplicate"] = record["warmup_inputs_sha256"] in fingerprints
        provenance.append(record)
        if record["excluded_exact_warmup_duplicate"]:
            LOGGER.warning("Excluded evaluation warmup duplicate group %d", group)
            continue
        design, states = causal_design(values)
        predictions = design[selected].astype(np.float64) @ ridge["beta"][:-1] + ridge["beta"][-1]
        for source, calibrator in enumerate(fitted.values()):
            calibrated = predict_calibrated(predictions, states[selected], calibrator)
            for model in range(len(MODELS)):
                for target in range(2):
                    moments[index, source, model, target] = sufficient_stats(
                        targets[selected, target], calibrated[:, model, target]
                    )
        if index % 16 == 0:
            LOGGER.info("evaluation row group %d/%d", index + 1, len(eval_ids))
    (OUT / "provenance_evaluation_v1.json").write_text(json.dumps(provenance, indent=2))
    np.savez(
        OUT / "evaluation_moments_v1.npz",
        moments=moments,
        row_groups=eval_ids,
        models=np.array(MODELS),
        sources=np.array(list(fitted)),
    )
    scores = weighted_pearson(moments.sum(axis=0))
    LOGGER.info(
        "Pooled evaluation WP (sources x models x targets):\n%s",
        np.array2string(scores, precision=6),
    )
    (OUT / "development_scores_v1.json").write_text(json.dumps(development_scores, indent=2))
    update_state(
        "probe_completed_analysis_pending", last_failure=None, evaluation_scores=scores.tolist()
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        main()
    except Exception:
        failure = traceback.format_exc()
        (OUT / "probe_offsets_failure.log").write_text(failure)
        update_state("probe_failed", last_failure=failure)
        raise
