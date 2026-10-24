"""Check the audit's loss and measure formulas on a hand-computable fixture."""

from __future__ import annotations

import numpy as np
from audit_batch_mass import (
    BATCH_DENOMINATOR_EPSILON,
    batch_statistics,
    component_rows,
)


def main() -> int:
    """Assert clipping, raw-parent error, mass distortion, tails, and components."""
    batch_a = np.asarray(
        [
            [1.0, -2.0, 1.0, 2.0, 0.0],
            [0.5, 0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 100.0, -100.0],
        ],
        dtype="<f4",
    )
    batch_b = np.asarray([[4.0, -1.0, 0.5, 3.0, 0.0]], dtype="<f4")
    zero_batch = np.asarray([[0.0, 0.0, 1.0, 2.0, -2.0]], dtype="<f4")
    mass_a, numerator_a, components_a = batch_statistics(batch_a, np.asarray([0, 0, 0]), 3)
    mass_b, numerator_b, components_b = batch_statistics(batch_b, np.asarray([1]), 3)
    mass_zero, numerator_zero, components_zero = batch_statistics(zero_batch, np.asarray([2]), 3)
    assert np.allclose(mass_a, [2.5, 2.0])
    assert np.allclose(numerator_a, [1.375, 8.0])
    assert np.allclose(mass_b, [1.0, 0.5])
    assert np.allclose(numerator_b, [1.0, 0.5])
    assert np.allclose(mass_zero, [0.0, 0.0])
    assert np.allclose(numerator_zero, [0.0, 0.0])
    tiny_mass, tiny_numerator, _ = batch_statistics(
        np.asarray([[1e-10, 0.0, 1.0, 1.0, 0.0]], dtype="<f4"), np.asarray([0]), 3
    )
    assert 0.0 < float(tiny_mass.sum()) < BATCH_DENOMINATOR_EPSILON
    assert np.isclose(
        float(tiny_numerator.sum()) / max(float(tiny_mass.sum()), BATCH_DENOMINATOR_EPSILON),
        0.01,
    )
    total_mass_a = float(mass_a.sum())
    total_mass_b = float(mass_b.sum())
    total_mass = total_mass_a + total_mass_b
    pooled = float((numerator_a.sum() + numerator_b.sum()) / total_mass)
    equal_batch = float((numerator_a.sum() / total_mass_a + numerator_b.sum() / total_mass_b) / 3)
    assert np.isclose(pooled, 10.875 / 6.0)
    assert np.isclose(equal_batch, (9.375 / 4.5 + 1.5 / 1.5 + 0.0) / 3)
    average_mass = total_mass / 3
    assert np.isclose(average_mass / total_mass_a, 4.0 / 9.0)
    assert np.isclose(average_mass / total_mass_b, 4.0 / 3.0)
    pooled_components = components_a + components_b + components_zero
    equal_components = components_a / total_mass_a + components_b / total_mass_b
    pooled_shares = pooled_components / total_mass
    equal_shares = equal_components / 2.0  # Exclude the zero-mass batch from the data measure.
    assert np.allclose(pooled_shares, [0.75, 0.25, 0.0])
    assert np.allclose(equal_shares, [0.5, 0.5, 0.0])
    assert component_rows(19_998, 4, [0, 1, 2]).tolist() == [0, 0, 1, 1]
    print("PASS: synthetic clipping, raw-parent loss, batch mass, and component measures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
