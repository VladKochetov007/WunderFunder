from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from wnn.gmetric import global_wp  # noqa: E402


def correlation_direction(y: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """Differentiate clipped weighted correlation away from clipping boundaries."""
    y = np.clip(y.astype(np.float64), -2, 2)
    raw = p.astype(np.float64)
    p = np.clip(raw, -2, 2)
    d = (q.astype(np.float64) - raw) * (np.abs(raw) < 2)
    w = np.abs(y)
    w /= w.sum()
    yc, pc, dc = (a - np.sum(w * a) for a in (y, p, d))
    vy, vp = np.sum(w * yc**2), np.sum(w * pc**2)
    cyp, cpd, cyd = (np.sum(w * a * b) for a, b in ((yc, pc), (pc, dc), (yc, dc)))
    return float((cyd - cyp * cpd / vp) / np.sqrt(vy * vp))


def main() -> None:
    rng = np.random.default_rng(734)
    y = rng.normal(size=(2000, 2)).astype(np.float32)
    p = (0.6 * y + rng.normal(size=y.shape)).astype(np.float32)
    # A gap around the clipping boundary keeps the central difference in one cell.
    close = np.abs(np.abs(p) - 2) < 0.1
    p[close] = np.sign(p[close]) * 2.2
    q = (0.5 * y + rng.normal(size=y.shape)).astype(np.float32)
    mask = np.ones(len(y), dtype=bool)
    analytic = float(np.mean([correlation_direction(y[:, k], p[:, k], q[:, k]) for k in range(2)]))
    epsilon = 0.001
    plus = global_wp(y, p + epsilon * (q - p), mask)["wp"]
    minus = global_wp(y, p - epsilon * (q - p), mask)["wp"]
    numerical = (plus - minus) / (2 * epsilon)
    assert abs(analytic - numerical) < 0.0001

    # Same input-defined coefficients in both trajectories: one layer only.
    retention = rng.uniform(0.998, 0.9999, size=(20000, 8))
    candidate = rng.normal(size=retention.shape)
    state = np.zeros(8)
    perturbed = np.full(8, 0.1)
    product = np.ones(8)
    max_error = 0.0
    for a, v in zip(retention, candidate, strict=True):
        state = a * state + (1 - a) * v
        perturbed = a * perturbed + (1 - a) * v
        product *= a
        max_error = max(max_error, float(np.max(np.abs(perturbed - state - 0.1 * product))))
    assert max_error < 1e-12
    result = {
        "scope": "synthetic algebra checks only; no competition inputs, fit, quality or timing claim",
        "seed": 734,
        "weighted_correlation_direction": {
            "analytic": analytic,
            "official_float32_central_difference": numerical,
            "epsilon": epsilon,
            "absolute_error": abs(analytic - numerical),
            "saturated_prediction_coordinates": int((np.abs(p) > 2).sum()),
        },
        "single_layer_input_conditioned_state_difference": {
            "rows": 20000,
            "hidden": 8,
            "maximum_float64_product_formula_error": max_error,
        },
        "constant_retention_e_folding_time": {
            str(tau): {
                "retention": float(np.exp(-1 / tau)),
                "update_gate_logit": float(np.log(np.expm1(1 / tau))),
            }
            for tau in (100, 1000, 10000)
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
