from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from objective_math import cell_fit, weighted_wp

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Run preregistered clustered ratio-estimator controls and exact population cases."""
    prereg = json.loads((HERE / "PREREG.json").read_text())
    config = prereg["synthetics"]
    outcomes = np.array([[-2, 0.2, 0.2], [-0.2, 1, 1], [-0.7, -0.7, 0.8]])
    population_x = np.repeat(np.arange(3), 3)
    population_y = outcomes.ravel()
    tilted = (outcomes * np.abs(outcomes)).sum(1) / np.abs(outcomes).sum(1)
    ordinary = outcomes.mean(1)
    exact = {
        "tilted_mean": tilted.tolist(),
        "chunk_mean": ordinary.tolist(),
        "pooled_wp": weighted_wp(population_y, tilted[population_x]),
        "chunk_wp": weighted_wp(population_y, ordinary[population_x]),
    }
    assert exact["pooled_wp"] - exact["chunk_wp"] > 0.01
    results: dict[str, object] = {"exact_positive": exact, "seeded_controls": []}
    rows = config["rows_per_cluster"]
    for seed in config["seeds"]:
        rng = np.random.default_rng(seed)
        cluster_x = np.tile(np.arange(3), config["clusters"] // 3)
        sampled_y = outcomes[cluster_x, rng.integers(0, 3, len(cluster_x))]
        x = np.repeat(cluster_x, rows)
        y = np.repeat(sampled_y, rows)
        w = np.abs(y)
        denominator = np.repeat(rows * np.abs(sampled_y), rows)
        pooled = cell_fit(x, y, w, np.ones(len(y)))
        chunk = cell_fit(x, y, w, denominator)
        deterministic_y = np.array([-2, 0.2, 1.0])[x]
        deterministic_w = np.abs(deterministic_y)
        det_pooled = cell_fit(x, deterministic_y, deterministic_w, np.ones(len(y)))
        det_chunk = cell_fit(x, deterministic_y, deterministic_w, rows * deterministic_w)
        assert np.max(np.abs(det_pooled - det_chunk)) < 1e-10
        sign_probability = np.array([0.2, 0.5, 0.8])[cluster_x]
        sign_y = np.repeat(np.where(rng.random(len(cluster_x)) < sign_probability, 1.0, -1.0), rows)
        equal_pooled = cell_fit(x, sign_y, np.ones(len(y)), np.ones(len(y)))
        equal_chunk = cell_fit(x, sign_y, np.ones(len(y)), np.full(len(y), rows))
        assert np.max(np.abs(equal_pooled - equal_chunk)) < 1e-10
        entry = {
            "seed": seed,
            "pooled_fit": pooled.tolist(),
            "chunk_fit": chunk.tolist(),
            "pooled_population_wp": weighted_wp(population_y, pooled[population_x]),
            "chunk_population_wp": weighted_wp(population_y, chunk[population_x]),
            "deterministic_high_CV_fit_max_difference": float(
                np.max(np.abs(det_pooled - det_chunk))
            ),
            "equal_mass_fit_max_difference": float(np.max(np.abs(equal_pooled - equal_chunk))),
            "mass_cv": float(denominator.std() / denominator.mean()),
        }
        results["seeded_controls"].append(entry)
    # An input-only factor cancels pointwise in a saturated model; not in a restricted constant model.
    y = np.array([-2.0, 0.5, 1.0])
    factors = np.array([1.0, 4.0, 1.0])
    results["input_weight_misspecification_counterexample"] = {
        "constant_optimum_original": float(np.average(y, weights=np.abs(y))),
        "constant_optimum_input_weighted": float(np.average(y, weights=np.abs(y) * factors)),
        "saturated_pointwise_optimum": y.tolist(),
    }
    # A teacher can reverse the ordering of metric-optimal conditional means.
    teacher = -tilted
    blend = 0.5 * tilted + 0.5 * teacher
    results["distillation_counterexample"] = {
        "teacher": teacher.tolist(),
        "alpha_half_fit": blend.tolist(),
        "wp": weighted_wp(population_y, blend[population_x]),
    }
    (HERE / "synthetic_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
