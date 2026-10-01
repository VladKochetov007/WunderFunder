"""QRENEWAL-01: input-only diagnostics. No data I/O or target-stage code."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

TOL = 1e-6
EPS = 1e-12
DECAY = 2.0 ** (-1.0 / 16.0)
GRID_START, GRID_STEP = 99, 64
MAX_PAIRS = 1024
N_KNOTS, RIDGE = 32, 0.001


def spaced(indices, limit=MAX_PAIRS):
    """No random sampling or post-outcome sample replacement."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) <= limit:
        return indices
    return indices[np.floor(np.linspace(0, len(indices) - 1, limit)).astype(int)]


def events(dv, a5):
    """At row t only current and immediately preceding observations are used."""
    dv, a5 = np.asarray(dv), np.asarray(a5)
    n = len(dv)
    valid = np.isfinite(dv).all(axis=1) & (np.abs(dv) < 5.19).all(axis=1)
    nonneutral = np.ptp(dv, axis=1) > TOL
    reset = np.zeros(n, dtype=bool)
    old = np.zeros(n, dtype=bool)
    reset[1:] = np.isfinite(a5[1:]) & np.isfinite(a5[:-1]) & (a5[1:] < a5[:-1] - TOL)
    old[1:] = valid[1:] & valid[:-1] & np.any(dv[1:] != dv[:-1], axis=1) & nonneutral[1:]
    return reset, reset & valid & nonneutral, old, valid


def mark_states(dv, a5, median, iqr):
    """Independent zero state at each sequence boundary; missing marks hold."""
    reset, marked, old, valid = events(dv, a5)
    u = np.clip((dv - median) / np.maximum(iqr, 1e-3), -4, 4)
    sr, sd = np.zeros((len(dv), 4)), np.zeros((len(dv), 4))
    r, d = np.zeros(4), np.zeros(4)
    for t in range(len(dv)):
        if marked[t]:
            r = DECAY * r + (1 - DECAY) * u[t]
        if old[t]:
            d = DECAY * d + (1 - DECAY) * u[t]
        sr[t], sd[t] = r, d
    return sr, sd


def miss_counts(record):
    dv, a5, dp = record['dv'], record['age'][:, 0], record['dp']
    reset, marked, old, valid = events(dv, a5)
    missed = marked & ~old
    prev_valid = np.r_[False, valid[:-1]]
    equal_dv = np.r_[False, np.all(dv[1:] == dv[:-1], axis=1)]
    intended = missed & prev_valid & equal_dv
    dp_pair_valid = np.zeros(len(dv), dtype=bool)
    dp_equal = np.zeros(len(dv), dtype=bool)
    dp_valid = np.isfinite(dp).all(axis=1) & (np.abs(dp) < 5.19).all(axis=1)
    dp_pair_valid[1:] = dp_valid[1:] & dp_valid[:-1]
    dp_equal[1:] = np.all(dp[1:] == dp[:-1], axis=1)
    out = {
        'rows': len(dv), 'resets': int(reset.sum()), 'marked_resets': int(marked.sum()),
        'reset_without_admitted_mark': int((reset & ~marked).sum()), 'old_events': int(old.sum()),
        'marked_resets_not_old': int(missed.sum()),
        'miss_previous_dv_invalid': int((missed & ~prev_valid).sum()),
        'intended_equal_dv': int(intended.sum()),
        'equal_dv_changed_dp': int((intended & dp_pair_valid & ~dp_equal).sum()),
        'equal_full_dp_dv': int((intended & dp_pair_valid & dp_equal).sum()),
        'equal_dv_dp_invalid': int((intended & ~dp_pair_valid).sum()),
        'unclassified_miss': int((missed & ~(intended | ~prev_valid)).sum()),
        'a5_extreme_rows': int((np.isfinite(a5) & (np.abs(a5) >= 5.19)).sum()),
        'a5_flat_pairs': int((np.isfinite(a5[1:]) & np.isfinite(a5[:-1]) & (a5[1:] == a5[:-1])).sum()),
    }
    assert out['marked_resets_not_old'] == out['intended_equal_dv'] + out['miss_previous_dv_invalid'] + out['unclassified_miss']
    return out


def pair_masks(age, knots=None):
    p, q = age[:-1], age[1:]
    delta = q - p
    finite = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    # Exactly the same subtraction order as the serving/reset feature rule.
    reset5 = finite & (q[:, 0] < p[:, 0] - TOL)
    reset6 = finite & (q[:, 1] < p[:, 1] - TOL)
    nonreset = finite & ~(reset5 | reset6)
    monotone = finite & (delta >= 0).all(axis=1)
    inside = np.ones(len(p), dtype=bool)
    if knots is not None:
        for j in range(2):
            inside &= (p[:, j] >= knots[j][0]) & (q[:, j] <= knots[j][-1])
            inside &= (q[:, j] >= knots[j][0]) & (p[:, j] <= knots[j][-1])
    return {
        'all': np.ones(len(p), dtype=bool), 'finite': finite,
        'invalid': ~finite, 'reset5': reset5, 'reset6': reset6,
        'any_reset': reset5 | reset6, 'finite_nonreset': nonreset,
        'microdrop_nonreset': nonreset & ~monotone,
        'monotone': monotone, 'both_raw_ties': monotone & (delta == 0).all(axis=1),
        'one_raw_tie': monotone & ((delta == 0).sum(axis=1) == 1),
        'outside_support_nonreset': nonreset & ~inside,
        'eligible': monotone & inside,
    }


def basis(x, knots):
    """Coefficients for nonnegative per-bin output increments, left origin zero."""
    return np.clip((np.asarray(x)[:, None] - knots[:-1]) / np.diff(knots), 0, 1)


def map_age(x, knots, increments):
    return np.interp(x, knots, np.r_[0.0, np.cumsum(increments)])


def map_pairs(age, fit):
    mapped = np.column_stack([map_age(age[:, j], fit['knots'][j], fit['increments'][j]) for j in range(2)])
    return np.diff(mapped, axis=0)


def summary(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {'n': len(x), 'median': float(np.median(x)) if len(x) else None,
            'p90': float(np.quantile(x, .9)) if len(x) else None,
            'min': float(np.min(x)) if len(x) else None,
            'max': float(np.max(x)) if len(x) else None}


def fit_clock(records):
    """Convex quadratic with nonnegative bin increments and one exact range constraint."""
    knots, iqr, quantiles = [], [], []
    for j in range(2):
        raw = np.concatenate([r['age'][:, j][np.isfinite(r['age'][:, j])] for r in records])
        if not len(raw):
            raise ValueError('No finite fitting ages')
        k = np.unique(np.quantile(raw, np.linspace(0, 1, N_KNOTS), method='linear'))
        qs = np.quantile(raw, [.05, .25, .5, .75, .95], method='linear')
        if len(k) < 4 or qs[-1] - qs[0] <= EPS or qs[3] - qs[1] <= EPS:
            raise ValueError('Insufficient fitting age support')
        knots.append(k); iqr.append(float(qs[3] - qs[1])); quantiles.append(qs)
    sizes = [len(k) - 1 for k in knots]
    total = sum(sizes)
    pair_samples, positive_raw5 = [], []
    for r in records:
        mask = pair_masks(r['age'], knots)
        idx = spaced(np.flatnonzero(mask['eligible']))
        if len(idx):
            p, q = r['age'][idx], r['age'][idx + 1]
            a = np.column_stack([basis(q[:, 0], knots[0]) - basis(p[:, 0], knots[0]),
                                 -(basis(q[:, 1], knots[1]) - basis(p[:, 1], knots[1]))])
            pair_samples.append((r['chain'], a, len(idx)))
            d = (q[:, 0] - p[:, 0]) / iqr[0]
            positive_raw5.append(d[d > EPS])
        else:
            raise ValueError('A fitting chain has zero eligible increment pairs')
    positive_raw5 = np.concatenate(positive_raw5)
    if not len(positive_raw5):
        raise ValueError('Fitting a5 has no positive sampled increments')
    residual_scale = float(np.median(positive_raw5))
    h_data = np.zeros((total, total))
    for _, a, n in pair_samples:
        h_data += a.T @ a / (len(records) * n * residual_scale ** 2)
    h_smooth = np.zeros_like(h_data)
    offset = 0
    for k, scale, size in zip(knots, iqr, sizes):
        # Adjacent first-slope differences are the declared second-difference ridge.
        inv_width = scale / np.diff(k)
        l = np.zeros((size - 1, total))
        for b in range(size - 1):
            l[b, offset + b] = -inv_width[b]
            l[b, offset + b + 1] = inv_width[b + 1]
        h_smooth += (l.T @ l) / (2 * (size - 1))
        offset += size
    h = h_data + RIDGE * h_smooth
    # Divide the whole objective by one positive scalar, leaving its minimizer unchanged.
    objective_scale = max(float(np.linalg.norm(h, ord=2)), 1e-12)
    h_opt = h / objective_scale
    c = np.zeros(total)
    c[:sizes[0]] = (basis(np.array([quantiles[0][-1]]), knots[0]) - basis(np.array([quantiles[0][0]]), knots[0]))[0]
    x0 = np.r_[np.diff(knots[0]) / (quantiles[0][-1] - quantiles[0][0]),
               np.diff(knots[1]) / (quantiles[1][-1] - quantiles[1][0])]
    sol = minimize(lambda x: .5 * x @ h_opt @ x, x0, jac=lambda x: h_opt @ x,
                   method='SLSQP', bounds=[(0, None)] * total,
                   constraints=[{'type': 'eq', 'fun': lambda x: c @ x - 1,
                                 'jac': lambda x: c}],
                   options={'maxiter': 1000, 'ftol': 1e-13, 'disp': False})
    x = sol.x
    if not sol.success or not np.isfinite(x).all() or x.min() < -1e-10 or abs(c @ x - 1) > 1e-7:
        raise ValueError(f'Clock optimization failed: {sol.message}; min={x.min()}; range={c @ x}')
    gradient = h_opt @ x
    free = x > 1e-8
    if not np.any(free & (c > 0)):
        raise ValueError('No free positive-range coefficient for KKT multiplier')
    multiplier = -float(c[free] @ gradient[free] / (c[free] @ c[free]))
    lagrangian_gradient = gradient + multiplier * c
    kkt_scale = max(1.0, float(np.abs(gradient).max()), abs(multiplier) * float(np.abs(c).max()))
    kkt = {'active_coefficient_threshold': 1e-8, 'equality_multiplier': multiplier,
           'stationarity_free': float(np.abs(lagrangian_gradient[free]).max() / kkt_scale),
           'dual_violation_active': float(max(0.0, -float(lagrangian_gradient[~free].min())) / kkt_scale) if (~free).any() else 0.0,
           'complementarity': float(np.abs(x * lagrangian_gradient).max() / kkt_scale),
           'normalization': kkt_scale, 'tolerance': 1e-5}
    if max(kkt['stationarity_free'], kkt['dual_violation_active'], kkt['complementarity']) > 1e-5:
        raise ValueError(f'Clock optimizer fails fixed KKT residual bound: {kkt}')
    # A lower bound for strict convexity on the range-constraint tangent space.
    from scipy.linalg import null_space
    tangent = null_space(c[None, :])
    tangent_eigen = np.linalg.eigvalsh(tangent.T @ h_opt @ tangent)
    if tangent_eigen[0] <= 1e-12:
        raise ValueError('Regularized clock objective fails fixed tangent-curvature stability bound')
    data_eigen = np.linalg.eigvalsh(tangent.T @ h_data @ tangent)
    data_tol = max(float(data_eigen[-1]), 0.0) * 1e-10
    data_positive = data_eigen[data_eigen > data_tol]
    result = {
        'knots': knots, 'increments': [x[:sizes[0]].copy(), x[sizes[0]:].copy()],
        'quantiles_05_25_50_75_95': quantiles, 'input_iqr': iqr,
        'residual_scale': residual_scale, 'range_before_normalization': float(c @ x),
        'optimizer': {'success': bool(sol.success), 'iterations': int(sol.nit), 'message': str(sol.message),
                      'kkt': kkt,
                      'data_objective': float(x @ h_data @ x),
                      'ridge_objective': float(RIDGE * x @ h_smooth @ x),
                      'tangent_min_eigenvalue_scaled': float(tangent_eigen[0]),
                      'tangent_max_eigenvalue_scaled': float(tangent_eigen[-1]),
                      'unregularized_design': {
                          'tangent_dimension': len(data_eigen), 'rank_relative_eigen_tolerance': 1e-10,
                          'rank': len(data_positive), 'gram_eigenvalues': data_eigen.tolist(),
                          'condition_on_retained_subspace': float(np.sqrt(data_positive[-1] / data_positive[0])) if len(data_positive) else None,
                          'full_tangent_condition': float(np.sqrt(data_eigen[-1] / data_eigen[0])) if len(data_positive) == len(data_eigen) else None,
                          'note': 'Diagnostic only. Positive regularized curvature does not establish data-only clock identification.'},
                      'objective_scale': objective_scale},
        'fit_sample_pairs_per_chain': [{'chain': chain, 'pairs': n} for chain, _, n in pair_samples],
    }
    positives = []
    for r in records:
        delta = map_pairs(r['age'], result)
        ok = pair_masks(r['age'], knots)['eligible'] & np.isfinite(delta).all(1) & (delta >= -EPS).all(1)
        common = delta.mean(1)
        positives.append(common[ok & (common > EPS)])
    positives = np.concatenate(positives)
    if not len(positives):
        raise ValueError('Fitted map has no positive common fitting increments')
    normalizer = float(np.median(positives))
    result['positive_increment_normalizer'] = normalizer
    result['increments'] = [d / normalizer for d in result['increments']]
    origins, excursions = [], []
    complete_total, complete_rejected = 0, 0
    for r in records:
        age = r['age'][:, 0]
        resets = np.flatnonzero(events(r['dv'], age)[0])
        mapped = map_age(age, knots[0], result['increments'][0])
        supported = np.isfinite(age) & (age >= knots[0][0]) & (age <= knots[0][-1])
        origins.extend(mapped[resets[supported[resets]]].tolist())
        for start, end in zip(resets[:-1], resets[1:]):
            complete_total += 1
            excursion = mapped[end - 1] - mapped[start]
            if supported[start:end].all() and np.all(np.diff(age[start:end]) >= 0) and excursion > EPS:
                excursions.append(excursion)
            else:
                complete_rejected += 1
    if not origins or not excursions:
        raise ValueError('No supported fitting reset origins or complete positive runs')
    result['reset_origin'] = float(np.median(origins))
    result['complete_run_excursion_median'] = float(np.median(excursions))
    result['complete_runs'] = {'total': complete_total, 'admitted': len(excursions), 'rejected': complete_rejected}
    return result


def audit_clock(records, fit):
    counts, all_errors, per_chain = {}, [], []
    reset_total = reset_inside = reset_near = 0
    reset_strata = {key: {'total': 0, 'near_origin': 0} for key in ['extreme', 'nonextreme', 'invalid_current_mark']}
    for r in records:
        masks = pair_masks(r['age'], fit['knots'])
        d = map_pairs(r['age'], fit)
        common = d.mean(1)
        usable = masks['eligible'] & np.isfinite(d).all(1) & (d >= -EPS).all(1) & (common > EPS)
        masks['mapped_zero_common'] = masks['eligible'] & np.isfinite(d).all(1) & (np.abs(common) <= EPS)
        masks['usable'] = usable
        local = {k: int(v.sum()) for k, v in masks.items()}
        for key, val in local.items():
            counts[key] = counts.get(key, 0) + val
        e = np.abs(d[usable, 0] - d[usable, 1]) / (np.abs(d[usable]).sum(1) + 1e-8)
        all_errors.append(e)
        reset, marked, _, _ = events(r['dv'], r['age'][:, 0])
        value = map_age(r['age'][:, 0], fit['knots'][0], fit['increments'][0])
        inside = np.isfinite(r['age'][:, 0]) & (r['age'][:, 0] >= fit['knots'][0][0]) & (r['age'][:, 0] <= fit['knots'][0][-1])
        near = inside & (np.abs(value - fit['reset_origin']) <= .05 * fit['complete_run_excursion_median'])
        reset_total += int(reset.sum()); reset_inside += int((reset & inside).sum()); reset_near += int((reset & near).sum())
        strata = {'extreme': np.abs(r['age'][:, 0]) >= 5.19,
                  'nonextreme': np.abs(r['age'][:, 0]) < 5.19,
                  'invalid_current_mark': ~marked}
        for key, m in strata.items():
            reset_strata[key]['total'] += int((reset & m).sum())
            reset_strata[key]['near_origin'] += int((reset & m & near).sum())
        per_chain.append({'chain': r['chain'], 'class': r['class'], 'half': r['half'],
                          'counts': local, 'error': summary(e),
                          'coverage': float(usable.sum() / masks['finite_nonreset'].sum()) if masks['finite_nonreset'].any() else 0.0,
                          'reset_total': int(reset.sum()), 'reset_near_origin': int((reset & near).sum())})
    err = summary(np.concatenate(all_errors))
    coverage = counts['usable'] / max(counts['finite_nonreset'], 1)
    origin_fraction = reset_near / max(reset_total, 1)
    return {'counts': counts, 'error': err, 'coverage': coverage,
            'per_chain': per_chain, 'per_chain_error_medians': summary([x['error']['median'] for x in per_chain if x['error']['median'] is not None]),
            'reset': {'total': reset_total, 'inside_support': reset_inside, 'near_origin': reset_near,
                      'near_origin_fraction': origin_fraction, 'strata': reset_strata},
            'gates': {'coverage': coverage >= .5,
                      'increment_error': err['median'] is not None and err['median'] <= .05 and err['p90'] <= .20,
                      'reset_origin': reset_total > 0 and origin_fraction >= .90}}


def audit_marks(fit_records, audit_records):
    admitted = []
    for r in fit_records:
        _, marked, old, _ = events(r['dv'], r['age'][:, 0])
        admitted.append(r['dv'][marked | old])
    marks = np.concatenate(admitted)
    if not len(marks):
        raise ValueError('No fitting admitted marks')
    qs = np.quantile(marks, [.25, .5, .75], axis=0, method='linear')
    median, iqr = qs[1], np.maximum(qs[2] - qs[0], 1e-3)
    fit_old = []
    for r in fit_records:
        _, sd = mark_states(r['dv'], r['age'][:, 0], median, iqr)
        fit_old.append(sd[GRID_START::GRID_STEP])
    old_grid = np.concatenate(fit_old)
    old_q = np.quantile(old_grid, [.25, .75], axis=0, method='linear')
    state_scale = np.maximum(old_q[1] - old_q[0], 1e-3)
    squares = []
    per_chain, totals = [], {}
    for r in audit_records:
        sr, sd = mark_states(r['dv'], r['age'][:, 0], median, iqr)
        z = (sr[GRID_START::GRID_STEP] - sd[GRID_START::GRID_STEP]) / state_scale
        squares.append(np.mean(z ** 2, axis=0))
        counts = miss_counts(r)
        for key, val in counts.items():
            totals[key] = totals.get(key, 0) + val
        per_chain.append({'chain': r['chain'], 'half': r['half'], 'class': r['class'],
                          'counts': counts, 'rms_in_fit_old_state_iqr': np.sqrt(np.mean(z ** 2, axis=0)).tolist()})
    rms = np.sqrt(np.mean(squares, axis=0))
    # Cases (a)+(b) from the proposal. Unclassifiable price payloads and previous
    # invalid volume endpoints are reported but cannot make the gate pass.
    qualified_misses = totals['equal_dv_changed_dp'] + totals['equal_full_dp_dv']
    intended_fraction = qualified_misses / max(totals['marked_resets'], 1)
    contributing = sum(x['counts']['equal_dv_changed_dp'] + x['counts']['equal_full_dp_dv'] > 0 for x in per_chain)
    return {'fit_mark_median': median, 'fit_mark_iqr': iqr, 'fit_old_state_iqr': state_scale,
            'counts': totals, 'qualified_repeated_misses': qualified_misses,
            'intended_miss_fraction': intended_fraction, 'contributing_chains': contributing,
            'rms_in_fit_old_state_iqr': rms, 'per_chain': per_chain,
            'gates': {'missing_repeated_marks': intended_fraction >= .1 and contributing >= 20,
                      'state_divergence': int((rms >= .1).sum()) >= 2}}


def to_json(value):
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
