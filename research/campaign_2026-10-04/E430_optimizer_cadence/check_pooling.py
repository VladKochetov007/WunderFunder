"""Tiny NumPy demonstration of macro-mass pooling versus chunk-mean averaging."""

from __future__ import annotations

import json

import numpy as np


def main() -> None:
    """Show that unequal supervised mass makes the two reductions differ."""
    rows = np.arange(2_000)
    valid = rows >= 99  # the trainer's first 1,000-row window has 901 scored rows
    weights = np.where(rows < 1_000, 1.0, 4.0)
    row_loss = np.where(rows < 1_000, 0.1, 0.8)
    numerator = weights * valid * row_loss
    mass = weights * valid
    pooled = float(numerator.sum() / mass.sum())
    chunk_means = [
        float(numerator[start : start + 1_000].sum() / mass[start : start + 1_000].sum())
        for start in (0, 1_000)
    ]
    unweighted_chunk_average = float(np.mean(chunk_means))
    expected = float((901 * 0.1 + 4_000 * 0.8) / (901 + 4_000))
    if not np.isclose(pooled, expected, rtol=0.0, atol=1e-15):
        raise AssertionError("global numerator/denominator pooling changed")
    if not np.isclose(unweighted_chunk_average, 0.45, rtol=0.0, atol=1e-15):
        raise AssertionError("chunk-mean reference changed")
    if abs(pooled - unweighted_chunk_average) < 0.2:
        raise AssertionError("the example no longer discriminates the reductions")
    print(
        json.dumps(
            {
                "first_window_valid_rows": int(valid[:1_000].sum()),
                "second_window_valid_rows": int(valid[1_000:].sum()),
                "window_masses": [901.0, 4_000.0],
                "window_losses": chunk_means,
                "macro_pooled_loss": pooled,
                "mean_of_window_losses": unweighted_chunk_average,
                "difference": pooled - unweighted_chunk_average,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
