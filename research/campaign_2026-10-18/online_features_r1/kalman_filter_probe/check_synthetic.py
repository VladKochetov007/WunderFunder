"""Bounded causal, reset, covariance, EMA, and throughput checks."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from kalman_features import (
    FEATURE_WIDTH,
    RAW_COLUMNS,
    OnlineKalmanFeatures,
    identity_unscaled_prior,
    steady_state_random_walk_gain,
)


def main() -> None:
    """Run the fixed synthetic checks and write a small execution receipt."""
    rng = np.random.default_rng(20261018)
    sequence = rng.normal(size=(20_000, 4)).astype(np.float32)
    sequence[:, 1] += np.linspace(-1.0, 1.0, len(sequence), dtype=np.float32)
    model = OnlineKalmanFeatures(identity_unscaled_prior())
    started = time.perf_counter()
    features = model.transform_selected_sequence(sequence)
    elapsed = time.perf_counter() - started
    assert features.shape == (20_000, FEATURE_WIDTH)
    assert np.isfinite(features).all()

    posterior_matches = True
    for model_name in ("random_walk", "local_trend"):
        batch_filter = OnlineKalmanFeatures()
        batch_levels = batch_filter.filter_selected_sequence(sequence[:1_000], model_name)
        stream_filter = OnlineKalmanFeatures()
        stream_levels = np.stack(
            [stream_filter.step_filtered_selected(row, model_name) for row in sequence[:1_000]]
        )
        posterior_matches &= bool(np.array_equal(batch_levels, stream_levels))
    assert posterior_matches

    changed = sequence.copy()
    changed[1_000:] = changed[1_000:] * -17.0 + 123.0
    changed_features = OnlineKalmanFeatures().transform_selected_sequence(changed)
    assert np.array_equal(features[:1_000], changed_features[:1_000])

    a = OnlineKalmanFeatures().transform_selected_sequence(sequence[:2_000])
    OnlineKalmanFeatures().transform_selected_sequence(sequence[:2_000] * 3.0 - 4.0)
    reused = OnlineKalmanFeatures()
    assert np.array_equal(a, reused.transform_selected_sequence(sequence[:2_000]))
    reused.transform_selected_sequence(sequence[:2_000] * 3.0 - 4.0)
    assert np.array_equal(a, reused.transform_selected_sequence(sequence[:2_000]))

    constant = np.full((2_000, 4), 7.25, dtype=np.float64)
    extreme = np.tile(np.array((1e6, -1e6, 1e6, -1e6)), (2_000, 1))
    for values in (constant, extreme):
        transformed = OnlineKalmanFeatures().transform_selected_sequence(values)
        assert np.isfinite(transformed).all()

    q, r = 0.02, 0.5
    gain = steady_state_random_walk_gain(q, r)
    scalar = sequence[:1_000, 0].astype(np.float64)
    scalar_features = OnlineKalmanFeatures().transform_selected_sequence(
        np.column_stack((scalar, scalar, scalar, scalar))
    )[:, 0]
    ema = scalar[0]
    ema_values = np.empty_like(scalar)
    ema_values[0] = ema
    for index in range(1, len(scalar)):
        ema += gain * (scalar[index] - ema)
        ema_values[index] = ema
    recovered = scalar + scalar_features
    assert np.allclose(recovered, ema_values, rtol=2e-6, atol=2e-6)

    raw112 = np.zeros((8, 112), dtype=np.float32)
    raw112[:, RAW_COLUMNS] = sequence[:8]
    augmented = OnlineKalmanFeatures().augment_sequence(raw112)
    assert augmented.shape == (8, 112 + FEATURE_WIDTH)
    assert np.array_equal(augmented[:, :112], raw112)
    min_cov = model.covariance_min_eigenvalue()
    assert np.isfinite(min_cov) and min_cov >= 0.0

    receipt = {
        "schema": "kalman_online_synthetic_checks.v1",
        "seed": 20261018,
        "rows_profiled": 20_000,
        "channels_profiled": 4,
        "features_per_row": FEATURE_WIDTH,
        "fixed_config": {
            "rw_q": q,
            "rw_r": r,
            "trend_q_level": 0.02,
            "trend_q_velocity": 0.0002,
            "trend_r": 0.5,
        },
        "checks": {
            "finite_20k": True,
            "future_perturbation_prefix_equal": True,
            "a_b_a_exact_reset": True,
            "constant_and_extreme_finite": True,
            "covariance_min_eigenvalue_nonnegative": True,
            "rw_matches_steady_constant_gain_ema": True,
            "streaming_posterior_matches_batch": posterior_matches,
            "raw_columns_retained": True,
        },
        "minimum_localtrend_covariance_eigenvalue": min_cov,
        "steady_state_random_walk_gain": gain,
        "wall_seconds_20k_rows": elapsed,
        "cpu_seconds": time.process_time(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "module_sha256": hashlib.sha256(
            Path(__file__).with_name("kalman_features.py").read_bytes()
        ).hexdigest(),
        "no_torch_import": "torch" not in __import__("sys").modules,
        "data_or_label_access": False,
    }
    Path(__file__).with_name("synthetic_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
