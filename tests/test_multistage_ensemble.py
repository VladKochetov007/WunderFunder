from __future__ import annotations

import numpy as np

from wnn.multistage_search_ensemble_math import solve_simplex


def test_simplex_solver_obeys_bounds_mass_and_convex_solution() -> None:
    """Recover the known interior minimizer of a positive-definite convex quadratic."""
    quadratic = np.eye(3, dtype=np.float64)
    linear = np.array([0.2, 0.5, 0.3], dtype=np.float64)

    result = solve_simplex(quadratic, linear)

    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    assert np.isclose(result.sum(), 1.0, atol=1e-12)
    assert np.allclose(result, linear, atol=1e-8)
