from __future__ import annotations

import numpy as np


def mass_multipliers(mass: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Return aligned or permuted equal-multiset estimator multipliers.

    Args:
        mass: Frozen detached positive chunk masses. Dividing each by a single
            mean gives exact mean-one multipliers over the supplied chunk slots.
        seed: None uses own mass; otherwise permute its finite multiset with a
            private RNG. This matches scalar distribution, not update norms.

    Returns:
        Multipliers with the same shape as mass. Mean multiplier is one.
    """
    mass = np.asarray(mass, float)
    if not mass.size or not np.isfinite(mass).all() or np.any(mass <= 0):
        raise ValueError("positive finite chunk masses required")
    scale = mass / mass.mean()
    if seed is None:
        return scale
    return np.random.default_rng(seed).permutation(scale.ravel()).reshape(scale.shape)
