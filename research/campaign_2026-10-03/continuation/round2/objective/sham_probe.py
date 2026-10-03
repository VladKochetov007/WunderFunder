from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from estimator_control import mass_multipliers
from objective_math import cell_fit, weighted_wp

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Test finite-multiset mass alignment separately from scale-distribution effects."""
    prereg = json.loads((HERE / "PREREG_SHAM.json").read_text())
    data = np.load(HERE / "chunk_summaries.npz")
    W = data["masses"].reshape(8, 32, 20, 4).sum(1)[..., 0]
    numerator = data["gradient"].reshape(8, 32, 20, 2, 2, 3).sum(1)
    result = {}
    for arm, name in enumerate(("zero_prediction", "cached_teacher_prediction")):
        gradient = numerator[:, :, arm].reshape(8, 20, 6) / W[..., None]
        baseline = gradient.mean((0, 1))
        aligned = (gradient * mass_multipliers(W)[..., None]).mean((0, 1))
        shams = np.array(
            [
                (gradient * mass_multipliers(W, seed)[..., None]).mean((0, 1))
                for seed in prereg["seeds"]
            ]
        )
        distance = np.linalg.norm(shams - aligned, axis=1) / max(np.linalg.norm(aligned), 1e-20)
        result[name] = {
            "baseline_mean": baseline.tolist(),
            "aligned_mean": aligned.tolist(),
            "sham_average_mean": shams.mean(0).tolist(),
            "sham_minus_baseline_norm": float(np.linalg.norm(shams.mean(0) - baseline)),
            "sham_to_aligned_relative_distance_q10_q50_q90": np.quantile(
                distance, [0.1, 0.5, 0.9]
            ).tolist(),
        }
    outcomes = np.array([[-2, 0.2, 0.2], [-0.2, 1, 1], [-0.7, -0.7, 0.8]])
    rng = np.random.default_rng(101)
    cluster_x = np.tile(np.arange(3), 80)
    sampled_y = outcomes[cluster_x, rng.integers(0, 3, 240)]
    x = np.repeat(cluster_x, 16)
    y = np.repeat(sampled_y, 16)
    mass = 16 * np.abs(sampled_y)
    base_weight = np.abs(y) / np.repeat(mass, 16)
    aligned_fit = cell_fit(
        x, y, base_weight * np.repeat(mass_multipliers(mass), 16), np.ones(len(y))
    )
    ordinary_fit = cell_fit(x, y, base_weight, np.ones(len(y)))
    sham_fit = np.array(
        [
            cell_fit(
                x, y, base_weight * np.repeat(mass_multipliers(mass, seed), 16), np.ones(len(y))
            )
            for seed in prereg["seeds"]
        ]
    )
    pop_x = np.repeat(np.arange(3), 3)
    pop_y = outcomes.ravel()
    result["clustered_regression_control"] = {
        "aligned_fit": aligned_fit.tolist(),
        "ordinary_fit": ordinary_fit.tolist(),
        "sham_average_fit": sham_fit.mean(0).tolist(),
        "aligned_population_wp": weighted_wp(pop_y, aligned_fit[pop_x]),
        "ordinary_population_wp": weighted_wp(pop_y, ordinary_fit[pop_x]),
        "sham_population_wp_mean": float(
            np.mean([weighted_wp(pop_y, fit[pop_x]) for fit in sham_fit])
        ),
        "equal_mass_sham_exact": bool(
            np.array_equal(mass_multipliers(np.ones(20)), mass_multipliers(np.ones(20), 9000))
        ),
    }
    (HERE / "sham_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
