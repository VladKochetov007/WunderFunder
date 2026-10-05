"""Train-only codebook fitting and deterministic target assignments."""

from __future__ import annotations

import numpy as np


def _farthest_centers(values: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Fit compact vector-quantization centers using training rows only."""
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or not len(x) or count < 1:
        raise ValueError("values must be a nonempty matrix and count positive")
    unique = np.unique(x, axis=0)
    count = min(count, len(unique))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, len(unique)))
    centers = [unique[first]]
    distance = np.square(unique - centers[0]).sum(axis=1)
    while len(centers) < count:
        next_index = int(np.argmax(distance))
        centers.append(unique[next_index])
        distance = np.minimum(distance, np.square(unique - unique[next_index]).sum(axis=1))
    result = np.stack(centers).astype(np.float32)
    for _ in range(20):
        assignment = ((x[:, None, :] - result[None, :, :]) ** 2).sum(axis=-1).argmin(axis=1)
        updated = result.copy()
        for index in range(len(result)):
            member = x[assignment == index]
            if len(member):
                updated[index] = member.mean(axis=0)
        if np.array_equal(updated, result):
            break
        result = updated
    return np.clip(result, -2, 2).astype(np.float32)


def fit_codebooks(
    training_targets: np.ndarray,
    joint_atoms: int = 32,
    marginal_atoms: int = 16,
    max_fit_rows: int = 65_536,
    seed: int = 271_828,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Fit paired and marginal atoms from a deterministic train-only sample.

    The sampling cap affects only codebook fitting. Every target row remains in
    the fixed head-training pass. ``n_codebook_rows`` is returned for provenance.
    """
    targets = np.asarray(training_targets)
    if targets.ndim != 2 or targets.shape[1] != 2 or not np.isfinite(targets).all():
        raise ValueError("training targets must be finite with shape (N, 2)")
    stride = max(1, int(np.ceil(len(targets) / max_fit_rows)))
    sample = np.clip(targets[::stride], -2, 2).astype(np.float32, copy=False)
    paired = _farthest_centers(sample, joint_atoms, seed)
    independent = np.stack(
        [_farthest_centers(sample[:, k : k + 1], marginal_atoms, seed + k)[:, 0] for k in range(2)]
    )
    # A fixed cyclic derangement preserves each learned marginal atom set while
    # breaking which t1 atom is paired with each t0 atom.
    permutation = np.roll(np.arange(len(paired)), 1)
    shuffled = paired.copy()
    shuffled[:, 1] = paired[permutation, 1]
    return paired, independent, shuffled, len(sample)


def nearest_labels(targets: np.ndarray, atoms: np.ndarray, block: int = 16_384) -> np.ndarray:
    """Assign rows to nearest atoms with bounded temporary memory."""
    y = np.asarray(targets, dtype=np.float32)
    c = np.asarray(atoms, dtype=np.float32)
    if y.ndim != 2 or c.ndim != 2 or y.shape[1] != c.shape[1]:
        raise ValueError("target and atom dimensions differ")
    labels = np.empty(len(y), dtype=np.int64)
    for start in range(0, len(y), block):
        target = y[start : start + block]
        distance = np.square(target[:, None, :] - c[None, :, :]).sum(axis=-1)
        labels[start : start + block] = distance.argmin(axis=1)
    return labels


def nearest_clipped_labels(
    targets: np.ndarray, atoms: np.ndarray, block: int = 16_384
) -> np.ndarray:
    """Assign metric-clipped targets to atoms fitted on the clipped support."""
    values = np.asarray(targets)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("target labels require a finite matrix")
    clipped = np.clip(values, -2, 2)
    return nearest_labels(clipped, atoms, block=block)
