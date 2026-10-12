"""Check that per-sequence moment accumulation matches the existing batch API."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MAIN = Path("/home/vlad/development/wunder/wnn_connectome_starterpack")
sys.path.insert(0, str(MAIN / "src"))

from wnn.evaluation import paired_component_bootstrap, paired_component_moments  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(41)
    targets = rng.normal(size=(7, 100, 2)).astype(np.float32)
    candidate = rng.normal(size=(7, 100, 2)).astype(np.float32)
    reference = rng.normal(size=(7, 100, 2)).astype(np.float32)
    mask = rng.random((7, 100)) > 0.23
    mask[:, ::3] = True

    batched_candidate, batched_reference = paired_component_moments(
        targets, candidate, reference, mask
    )
    row_candidate = np.empty_like(batched_candidate)
    row_reference = np.empty_like(batched_reference)
    for index in range(len(targets)):
        candidate_moment, reference_moment = paired_component_moments(
            targets[index : index + 1],
            candidate[index : index + 1],
            reference[index : index + 1],
            mask[index : index + 1],
        )
        row_candidate[index] = candidate_moment[0]
        row_reference[index] = reference_moment[0]

    if not np.array_equal(batched_candidate, row_candidate):
        raise AssertionError("candidate moments differ between batched and rowwise calls")
    if not np.array_equal(batched_reference, row_reference):
        raise AssertionError("reference moments differ between batched and rowwise calls")

    components = np.arange(len(targets), dtype=np.int64) % 4
    batched = paired_component_bootstrap(
        batched_candidate,
        batched_reference,
        components,
        n_boot=16,
        seed=271828,
        centered=True,
    )
    rowwise = paired_component_bootstrap(
        row_candidate,
        row_reference,
        components,
        n_boot=16,
        seed=271828,
        centered=True,
    )
    if batched != rowwise:
        raise AssertionError("bootstrap results differ between batched and rowwise moments")

    print("rowwise moment and bootstrap outputs exactly equal batched helper")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
