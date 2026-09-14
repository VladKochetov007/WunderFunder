"""Vectorized metric must agree with the organizers' reference scorer in utils.py."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import utils as reference
from wnn.metric import block_scores, score, weighted_pearson_rows

T = reference.SEQUENCE_LENGTH


def _reference_block(targets, predictions, mask):
    acc = reference.BlockAccumulator()
    acc.add(targets, predictions, mask)
    return acc.scores[0] if acc.scores else None


def _random_block(rng, *, scored_frac=0.1, quantum=0.12543, pred_scale=1.0):
    k = rng.integers(-30, 31, size=(T, 2))
    targets = (k * quantum).astype(np.float32)
    predictions = (rng.normal(size=(T, 2)) * pred_scale + 0.3 * targets).astype(np.float32)
    predictions[: reference.WARMUP] = np.nan
    mask = rng.random(T) < scored_frac
    mask[: reference.WARMUP] = False
    return targets, predictions, mask


@given(
    seed=st.integers(0, 2**32 - 1),
    frac=st.sampled_from([0.001, 0.05, 0.5]),
    scale=st.sampled_from([0.01, 1.0, 5.0]),
)
@settings(max_examples=40, deadline=None)
def test_rows_match_reference_weighted_pearson(seed, frac, scale):
    rng = np.random.default_rng(seed)
    targets, predictions, mask = _random_block(rng, scored_frac=frac, pred_scale=scale)
    got = weighted_pearson_rows(targets[None, :, 0], predictions[None, :, 0], mask[None])
    want = reference.weighted_pearson(targets[mask, 0], predictions[mask, 0])
    assert got.shape == (1,)
    assert got[0] == pytest.approx(want, abs=1e-9)


def test_batch_matches_reference_accumulator_including_ineligible_blocks():
    rng = np.random.default_rng(7)
    blocks = [_random_block(rng, scored_frac=f) for f in (0.2, 0.01, 0.3)]
    # Ineligible: every selected t1 is non-negative.
    t, p, m = _random_block(rng)
    t[:, 1] = np.abs(t[:, 1])
    blocks.append((t, p, m))
    # Ineligible: empty mask.
    t, p, m = _random_block(rng)
    m[:] = False
    blocks.append((t, p, m))

    targets = np.stack([b[0] for b in blocks])
    predictions = np.stack([b[1] for b in blocks])
    mask = np.stack([b[2] for b in blocks])

    per_block = block_scores(targets, predictions, mask)
    for i, (t, p, m) in enumerate(blocks):
        want = _reference_block(t, p, m)
        if want is None:
            assert np.isnan(per_block[i, 0]) and np.isnan(per_block[i, 1])
        else:
            np.testing.assert_allclose(per_block[i], want, atol=1e-9)

    acc = reference.BlockAccumulator()
    for t, p, m in blocks:
        acc.add(t, p, m)
    want = acc.result()
    got = score(targets, predictions, mask)
    assert got["eligible_blocks"] == want["eligible_blocks"] == 3
    assert got["weighted_pearson"] == pytest.approx(want["weighted_pearson"], abs=1e-9)
    assert got["t0"] == pytest.approx(want["t0"], abs=1e-9)
    assert got["t1"] == pytest.approx(want["t1"], abs=1e-9)


def test_degenerate_cases_contribute_zero():
    rng = np.random.default_rng(3)
    targets, predictions, mask = _random_block(rng)
    constant = np.full_like(predictions, 0.5)
    assert weighted_pearson_rows(targets[None, :, 0], constant[None, :, 0], mask[None])[0] == 0.0
    zeros = np.zeros_like(targets)
    assert weighted_pearson_rows(zeros[None, :, 0], predictions[None, :, 0], mask[None])[0] == 0.0


def test_prediction_clipping_is_applied():
    # Predictions beyond the clip must be clipped exactly like the reference.
    rng = np.random.default_rng(11)
    targets, predictions, mask = _random_block(rng, scored_frac=0.4, pred_scale=10.0)
    got = weighted_pearson_rows(targets[None, :, 1], predictions[None, :, 1], mask[None])[0]
    want = reference.weighted_pearson(targets[mask, 1], predictions[mask, 1])
    assert got == pytest.approx(want, abs=1e-9)
    unclipped = np.corrcoef(targets[mask, 1], predictions[mask, 1])[0, 1]
    assert abs(got - unclipped) > 1e-4


def test_score_rejects_selected_warmup_rows():
    rng = np.random.default_rng(5)
    targets, predictions, mask = _random_block(rng)
    mask[10] = True
    with pytest.raises(ValueError, match="warm-up"):
        score(targets[None], predictions[None], mask[None])
