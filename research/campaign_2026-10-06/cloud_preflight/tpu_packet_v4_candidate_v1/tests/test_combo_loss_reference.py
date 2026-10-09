from __future__ import annotations

import numpy as np
from combo_loss_reference import aggregate_shards, component_sums, loss_from_sums


def make_batch() -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Create deterministic unequal-mass supervision and corresponding logits."""
    rng = np.random.default_rng(46320261007)
    batch_size, time_rows = 8, 19
    prediction = rng.normal(size=(batch_size, time_rows, 23)).astype(np.float32)
    ema_prediction = rng.normal(size=(batch_size, time_rows, 2)).astype(np.float32)
    labels = rng.normal(size=(batch_size, time_rows, 2)).astype(np.float32)
    fixed_target = rng.normal(size=(batch_size, time_rows, 2)).astype(np.float32)
    shard_scale = np.asarray([0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 3.0, 8.0], dtype=np.float32)
    main_weights = rng.uniform(0.1, 1.0, size=(batch_size, time_rows)).astype(np.float32)
    main_weights *= shard_scale[:, None]
    main_weights[:, ::7] = 0.0
    next_weights = rng.uniform(0.1, 1.0, size=(batch_size, time_rows)).astype(np.float32)
    next_weights *= shard_scale[:, None]
    next_weights[0] = 0.0
    batch = {
        "labels": labels,
        "fixed_target": fixed_target,
        "main_weights": main_weights,
        "structural_labels": np.stack(
            (
                rng.integers(0, 12, size=(batch_size, time_rows)),
                rng.integers(0, 3, size=(batch_size, time_rows)),
                rng.integers(0, 4, size=(batch_size, time_rows)),
            ),
            axis=-1,
        ).astype(np.int32),
        "next_weights": next_weights,
        "next_rows": rng.integers(0, 8, size=(batch_size, time_rows)).astype(np.float32),
        "next_up": rng.integers(0, 2, size=(batch_size, time_rows)).astype(np.float32),
    }
    return prediction, batch, ema_prediction


def slice_batch(batch: dict[str, np.ndarray], rows: slice) -> dict[str, np.ndarray]:
    """Slice the batch dimension for one deliberately unequal-mass shard."""
    return {name: value[rows] for name, value in batch.items()}


def test_summed_sufficient_statistics_match_concatenated_objective() -> None:
    """Global loss is invariant to sharding when raw masses are summed first."""
    prediction, batch, ema_prediction = make_batch()
    global_sums = component_sums(prediction, batch, ema_prediction)
    shard_sums = [
        component_sums(
            prediction[index : index + 1],
            slice_batch(batch, slice(index, index + 1)),
            ema_prediction[index : index + 1],
        )
        for index in range(prediction.shape[0])
    ]

    combined = aggregate_shards(shard_sums)

    for name, expected in global_sums.items():
        assert np.isclose(combined[name], expected, rtol=1e-12, atol=1e-12)
    assert np.isclose(loss_from_sums(combined), loss_from_sums(global_sums), rtol=1e-12, atol=1e-12)


def test_mean_of_shard_normalized_losses_is_a_detectable_negative_control() -> None:
    """Averaging per-shard normalized losses fails under unequal shard masses."""
    prediction, batch, ema_prediction = make_batch()
    global_loss = loss_from_sums(component_sums(prediction, batch, ema_prediction))
    shard_losses = [
        loss_from_sums(
            component_sums(
                prediction[index : index + 1],
                slice_batch(batch, slice(index, index + 1)),
                ema_prediction[index : index + 1],
            )
        )
        for index in range(prediction.shape[0])
    ]

    assert batch["next_weights"][0].sum() == 0.0
    assert batch["next_weights"][1:].sum() > 0.0
    assert abs(float(np.mean(shard_losses)) - global_loss) > 1e-3
