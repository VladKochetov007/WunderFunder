"""Synthetic only: never opens a dataset, cache, model, or empirical result."""
import json
import time
import numpy as np
from core import (DECAY, audit_clock, audit_marks, events, fit_clock,
                  map_age, mark_states, miss_counts, pair_masks, to_json)


def fixture(seed, n=2400, false_clock=False, nonlinear=False):
    rng = np.random.default_rng(seed)
    # Reset channels have independent renewal schedules; elapsed increments are shared.
    dt = rng.uniform(.2, 2, size=n)
    dt6 = rng.lognormal(mean=-.2, sigma=1.2, size=n) if false_clock else dt
    reset5, reset6 = rng.random(n) < .07, rng.random(n) < .09
    age = np.zeros((n, 2))
    for t in range(1, n):
        # Make every synthetic renewal observable as a downward step. Repeated
        # renewals while age remains exactly zero are not identifiable by this recipe.
        reset5[t] &= age[t - 1, 0] > 0
        reset6[t] &= age[t - 1, 1] > 0
        age[t, 0] = 0 if reset5[t] else age[t - 1, 0] + dt[t]
        age[t, 1] = 0 if reset6[t] else age[t - 1, 1] + dt6[t]
    true_age = age.copy()
    if nonlinear:
        age[:, 0] = np.sqrt(age[:, 0] + 1) - 1
        age[:, 1] = np.log1p(age[:, 1])
    else:
        age[:, 0] = age[:, 0] * .12 - 5.199
        age[:, 1] = age[:, 1] * .07 - 5.199
    dv = np.zeros((n, 4)); mark = np.array([.6, -.4, .2, -.1])
    for t in range(n):
        if reset5[t] and rng.random() < .3:
            mark = rng.uniform(-1, 1, size=4)
        dv[t] = mark
    dp = np.tile(np.array([.2, .5, -.3, .8]), (n, 1))
    for t in range(1, n):
        dp[t] = dp[t - 1]
        if reset5[t] and rng.random() < .5:
            dp[t] = rng.uniform(-1, 1, 4)
    return {'chain': seed, 'class': seed % 4, 'half': seed % 2,
            'age': age, 'dv': dv, 'dp': dp, 'true_age': true_age}


def test_counterexample():
    # Equal observed payload histories and old detector states; distinct reset histories.
    dv = np.tile([1., -.5, .2, -.1], (9, 1))
    a_many = np.array([0., 1., 0., 1., 0., 1., 0., 1., 0.])
    a_once = np.array([0., 1., 2., 3., 4., 5., 6., 7., 0.])
    sr_many, sd_many = mark_states(dv, a_many, np.zeros(4), np.ones(4))
    sr_once, sd_once = mark_states(dv, a_once, np.zeros(4), np.ones(4))
    np.testing.assert_array_equal(sd_many, sd_once)
    np.testing.assert_allclose(sr_many[-1], (1 - DECAY ** 4) * dv[0], atol=1e-15)
    np.testing.assert_allclose(sr_once[-1], (1 - DECAY) * dv[0], atol=1e-15)
    assert np.linalg.norm(sr_many[-1] - sr_once[-1]) > .1


def test_prefix_and_validity():
    r = fixture(123, n=90)
    r['dv'][8] = np.nan; r['dv'][15] = 5.199; r['dv'][23] = 0
    r['age'][7, 0] = np.nan
    m, q = np.zeros(4), np.ones(4)
    full = mark_states(r['dv'], r['age'][:, 0], m, q)
    for n in range(1, len(r['dv']) + 1):
        short = mark_states(r['dv'][:n], r['age'][:n, 0], m, q)
        for a, b in zip(full, short):
            np.testing.assert_array_equal(a[:n], b)
    # A new competition sequence must not inherit its predecessor's mark or age.
    twice = mark_states(r['dv'], r['age'][:, 0], m, q)
    for a, b in zip(full, twice):
        np.testing.assert_array_equal(a, b)
    assert all(np.isfinite(x).all() for x in full)
    reset, marked, old, _ = events(r['dv'], r['age'][:, 0])
    assert not reset[0] and not marked[0] and not old[0]
    assert not marked[[8, 15, 23]].any()
    assert not old[[8, 9, 15, 16]].any()
    # Sentinel ages remain admissible, small negative steps are tracked, ties retained.
    ages = np.array([[0., 0.], [0., 0.], [-.5e-6, 0.], [-5.199, 1.], [np.nan, 1.]])
    masks = pair_masks(ages)
    assert masks['all'].sum() == 4 and masks['finite_nonreset'].sum() == 2
    assert masks['microdrop_nonreset'].sum() == 1 and masks['both_raw_ties'].sum() == 1
    assert masks['reset5'].sum() == 1 and masks['invalid'].sum() == 1
    # Floating-point subtraction order at the exact 1e-6 reset boundary.
    p = .1; q = p - 1e-6
    boundary = np.array([[p, 0.], [q, 1.]])
    assert events(np.tile([1., 0., 0., 0.], (2, 1)), boundary[:, 0])[0][1] == pair_masks(boundary)['reset5'][0]
    # Invalid previous marks count separately; never borrow a prior or future mark.
    dv = np.array([[np.nan] * 4, [1., -.5, .2, -.1], [1., -.5, .2, -.1], [1., -.5, .2, -.1]])
    rr = {'dv': dv, 'age': np.array([[3., 0.], [0., 1.], [3., 2.], [0., 3.]]), 'dp': np.ones((4, 4))}
    count = miss_counts(rr)
    assert count['miss_previous_dv_invalid'] == 1 and count['intended_equal_dv'] == 1


def test_clock(false_clock=False, nonlinear=False):
    fit_records = [fixture(i, false_clock=false_clock, nonlinear=nonlinear) for i in range(1, 13)]
    audit_records = [fixture(i, false_clock=false_clock, nonlinear=nonlinear) for i in range(101, 113)]
    fitted = fit_clock(fit_records)
    report = audit_clock(audit_records, fitted)
    assert abs(fitted['range_before_normalization'] - 1) < 1e-7
    assert all((x >= 0).all() for x in fitted['increments'])
    # True inverse oracle uses generating ages, never inferred order or target values.
    errors = []
    for r in audit_records:
        mask = pair_masks(r['age'])['monotone']
        d = np.diff(r['true_age'], axis=0)[mask]
        errors.extend((np.abs(d[:, 0] - d[:, 1]) / (np.abs(d).sum(1) + 1e-8)).tolist())
    oracle_median = float(np.median(errors))
    if false_clock:
        assert oracle_median > .2
        assert not report['gates']['increment_error'], report
    else:
        assert max(errors) < 1e-12
        assert all(report['gates'].values()), report
    return {'false_clock': false_clock, 'nonlinear': nonlinear, 'gates': report['gates'],
            'error': report['error'], 'coverage': report['coverage'],
            'origin_fraction': report['reset']['near_origin_fraction'], 'oracle_median_error': oracle_median,
            'optimizer': fitted['optimizer']}


def test_degenerate_rejected():
    r = fixture(19, 100)
    r['age'][:] = 0
    try:
        fit_clock([r])
    except ValueError as e:
        assert 'support' in str(e)
    else:
        raise AssertionError('Constant ages cannot identify a nonzero-range clock')


def test_footer_accounting():
    # Synthetic bytes only: previous QMARK footer_bytes includes its 8-byte trailer.
    import hashlib
    import tempfile
    from pathlib import Path
    from run_inputs import fingerprint
    with tempfile.TemporaryDirectory() as directory:
        payload = b'synthetic footer'
        trailer = len(payload).to_bytes(4, 'little') + b'PAR1'
        path = Path(directory) / 'synthetic-footer.bin'
        path.write_bytes(b'PAR1' + payload + trailer)
        result = fingerprint(path)
        assert result['footer_bytes'] == len(payload) + 8
        assert result['footer_sha256'] == hashlib.sha256(payload + trailer).hexdigest()


def test_mark_audit():
    fitting = [fixture(i, 600) for i in range(201, 225)]
    held = [fixture(i, 600) for i in range(301, 325)]
    result = audit_marks(fitting, held)
    assert result['gates']['missing_repeated_marks']
    assert result['gates']['state_divergence']
    assert result['counts']['equal_dv_changed_dp'] > 0
    assert result['counts']['equal_full_dp_dv'] > 0
    assert result['counts']['unclassified_miss'] == 0
    return to_json({k: v for k, v in result.items() if k != 'per_chain'})


def main():
    started = time.monotonic()
    test_counterexample(); test_prefix_and_validity(); test_degenerate_rejected(); test_footer_accounting()
    outcomes = [test_clock(), test_clock(nonlinear=True), test_clock(false_clock=True)]
    mark_outcome = test_mark_audit()
    print(json.dumps(to_json({'status': 'PASS', 'synthetic_only': True,
                             'tests': ['repeated_payload_counterexample', 'all_prefix_parity', 'sequence_reset',
                                       'mark_validity', 'all_row_denominators', 'degenerate_rejection',
                                       'perfect_linear_clock', 'perfect_nonlinear_clock', 'false_clock_oracle',
                                       'mark_audit_and_dp_strata', 'reset_threshold_boundary', 'footer_length_convention'],
                             'clock_fixtures': outcomes, 'mark_audit_fixture': mark_outcome,
                             'seconds': time.monotonic() - started}), indent=2))


if __name__ == '__main__':
    main()
