from __future__ import annotations

import numpy as np

from wnn.markov import N_MARKOV_FEATURES, markov_features


def _book_path(states, rows_per_event=3, seed=0):
    """Rows where the i0 price block takes the given state values; volumes vary between events."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in states:
        for r in range(rows_per_event):
            x = rng.normal(size=112).astype(np.float32)
            x[0:22] = s  # bid p0 and ask p11 both equal the state level
            rows.append(x)
    return np.stack(rows)


def test_shape_and_prefix_invariance():
    x = _book_path([1, 2, 1, 2, 3, 2, 1, 2, 1, 2] * 5)
    f = markov_features(x)
    assert f.shape == (len(x), N_MARKOV_FEATURES)
    for cut in (1, 7, 31, 100):
        np.testing.assert_array_equal(markov_features(x[:cut]), f[:cut])


def test_learns_alternation_without_labels():
    # Bounce 1,2,1,2,...: from state 2 the next move is always down.
    x = _book_path([1, 2] * 30, rows_per_event=2)
    f = markov_features(x)
    last_row_in_state2 = len(x) - 1
    bid_up, bid_down = f[last_row_in_state2, 1], f[last_row_in_state2, 2]
    assert bid_down > 0.4 and bid_up < -0.4


def test_first_visit_has_no_information():
    x = _book_path([5, 6, 7])
    f = markov_features(x)
    assert np.all(f[:3] == 0)
