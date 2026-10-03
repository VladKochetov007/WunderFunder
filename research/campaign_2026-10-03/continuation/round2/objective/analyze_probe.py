from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from objective_math import summaries

HERE = Path(__file__).resolve().parent


def cross_batch_r2(features: np.ndarray, target: np.ndarray) -> float:
    """Leave one original 32-sequence batch out; no row-level random CV."""
    n_batches, n_chunks = target.shape
    prediction = np.empty_like(target)
    x = features.reshape(n_batches * n_chunks, -1)
    y = target.ravel()
    group = np.repeat(np.arange(n_batches), n_chunks)
    for batch in range(n_batches):
        fit = group != batch
        center = x[fit].mean(0)
        scale = x[fit].std(0)
        scale[scale < 1e-10] = 1
        normalized = (x - center) / scale
        design = np.column_stack([np.ones(len(x)), normalized])
        penalty = np.eye(design.shape[1]) * 1.0
        penalty[0, 0] = 0
        coefficient = np.linalg.solve(design[fit].T @ design[fit] + penalty, design[fit].T @ y[fit])
        prediction[batch] = design[~fit] @ coefficient
    return float(1 - np.sum((prediction - target) ** 2) / np.sum((target - target.mean()) ** 2))


def diagnostic(
    chunk: int, permutation: np.ndarray, data: dict[str, np.ndarray], is_primary: bool
) -> dict[str, object]:
    """Aggregate original batch/chunk mass, regression controls and gradient reweighting."""
    merge = chunk // 1000
    n, old_chunks, _ = data["masses"].shape
    steps = old_chunks // merge
    batch = 32
    batches = n // batch

    def aggregate(name: str, is_mean: bool = False) -> np.ndarray:
        array = data[name][permutation]
        reshaped = array.reshape(n, steps, merge, *array.shape[2:])
        merged = reshaped.mean(2) if is_mean else reshaped.sum(2)
        grouped = merged.reshape(batches, batch, steps, *merged.shape[2:])
        return grouped.mean(1) if is_mean else grouped.sum(1)

    mass = aggregate("masses")
    W, R, N, label_only = np.moveaxis(mass, -1, 0)
    if np.any(W <= 0) or np.any(R <= 0):
        raise ValueError("zero recipe normalizer")
    relative = W.mean() / W
    output: dict[str, object] = {
        "chunk": chunk,
        "batches": batches,
        "steps": steps,
        "main_W": summaries(W.ravel()),
        "row_R": summaries(R.ravel()),
        "next_R": summaries(N.ravel()),
        "label_only_W": summaries(label_only.ravel()),
        "main_W_excluding_warmup_chunk": summaries(W[:, 1:].ravel()),
        "relative_inverse_mass": summaries(relative.ravel()),
        "within_position_cv": (W.std(0) / W.mean(0)).tolist(),
        "weighted_abs_label_sum_per_row": summaries((W / R).ravel()),
        "close_measured_claim": bool(W.std() / W.mean() < 0.05),
    }
    gradient = aggregate("gradient")
    arms = {}
    for arm, name in enumerate(("zero_prediction", "cached_teacher_prediction")):
        raw = gradient[:, :, arm].reshape(batches, steps, -1)
        chunk_gradient = raw / W[..., None]
        pooled_gradient = raw / W.mean()
        sn = chunk_gradient.mean((0, 1))
        pooled = pooled_gradient.mean((0, 1))
        cosine = float(sn @ pooled / max(np.linalg.norm(sn) * np.linalg.norm(pooled), 1e-20))
        arms[name] = {
            "chunk_mean_gradient": sn.tolist(),
            "pooled_mean_gradient": pooled.tolist(),
            "aggregate_cosine": cosine,
            "mean_gradient_relative_difference": float(
                np.linalg.norm(sn - pooled) / max(np.linalg.norm(pooled), 1e-20)
            ),
            "trace_gradient_variance_chunk": float(
                np.var(chunk_gradient.reshape(-1, 6), axis=0).sum()
            ),
            "trace_gradient_variance_pooled": float(
                np.var(pooled_gradient.reshape(-1, 6), axis=0).sum()
            ),
        }
    output["gradient_proxies"] = arms
    grouped_bins = aggregate("bins")
    pooled_bin = grouped_bins.sum((0, 1))
    sn_bin = (grouped_bins / W[:, :, None, None, None]).sum((0, 1))
    shifts = {}
    for moment, name in enumerate(
        ("label", "combo_bootstrap_target", "nonbootstrap_target"), start=1
    ):
        pooled_mean = np.divide(
            pooled_bin[:, :, moment],
            pooled_bin[:, :, 0],
            out=np.zeros((16, 2)),
            where=pooled_bin[:, :, 0] > 0,
        )
        sn_mean = np.divide(
            sn_bin[:, :, moment], sn_bin[:, :, 0], out=np.zeros((16, 2)), where=sn_bin[:, :, 0] > 0
        )
        weights = pooled_bin[:, :, 0]
        shifts[name] = {
            "pooled_conditional_bin_mean": pooled_mean.tolist(),
            "chunk_conditional_bin_mean": sn_mean.tolist(),
            "weight_mass": weights.tolist(),
            "weighted_RMS_shift": float(
                np.sqrt(np.sum(weights * (pooled_mean - sn_mean) ** 2) / weights.sum())
            ),
            "max_abs_nonempty_shift": float(
                np.max(np.abs(pooled_mean[weights > 0] - sn_mean[weights > 0]))
            ),
        }
    output["conditional_input_bins"] = shifts
    if is_primary:
        input_features = aggregate("inputs", True)
        label_features = aggregate("labels", True)
        positions = np.broadcast_to(np.arange(steps)[None, :, None], (batches, steps, 1)) / steps
        warm = np.broadcast_to((np.arange(steps) == 0)[None, :, None], (batches, steps, 1)).astype(
            float
        )
        time = np.concatenate([positions, warm], -1)
        raw_input = np.concatenate([time, input_features], -1)
        known_row = np.concatenate([raw_input, np.log(R[..., None])], -1)
        label_augmented = np.concatenate([known_row, label_features], -1)
        output["predictability_logW_LOBO_R2"] = {
            "position_only": cross_batch_r2(time, np.log(W)),
            "input_summaries": cross_batch_r2(raw_input, np.log(W)),
            "inputs_plus_exact_input_row_mass": cross_batch_r2(known_row, np.log(W)),
            "inputs_row_mass_plus_labels": cross_batch_r2(label_augmented, np.log(W)),
            "label_only": cross_batch_r2(np.concatenate([time, label_features], -1), np.log(W)),
            "caveat": "8 held-out batches; observational diagnostics, chunk input summaries available only after reading full chunk",
        }
    return output


def main() -> None:
    """Analyze frozen TRAIN sufficient statistics under preregistered order controls."""
    archive = np.load(HERE / "chunk_summaries.npz")
    data = {name: archive[name] for name in archive.files}
    permutations = [
        np.arange(len(data["groups"])),
        np.random.default_rng(173102).permutation(len(data["groups"])),
        np.random.default_rng(173103).permutation(len(data["groups"])),
    ]
    results = {
        "actual": [
            diagnostic(chunk, order, data, i == 0)
            for chunk in (1000, 2000)
            for i, order in enumerate(permutations)
        ],
        "synthetic": json.loads((HERE / "synthetic_results.json").read_text()),
        "scope": "TRAIN-only scientific-estimator diagnostic; no student/GPU/validation score",
        "prereg_sha256": hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest(),
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "actual_summary": [
                    {
                        "chunk": result["chunk"],
                        "normalizer_cv": result["main_W"]["cv"],
                        "gate_closed": result["close_measured_claim"],
                        "predictability": result.get("predictability_logW_LOBO_R2"),
                        "gradient_cosine": result["gradient_proxies"]["cached_teacher_prediction"][
                            "aggregate_cosine"
                        ],
                    }
                    for result in results["actual"]
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
