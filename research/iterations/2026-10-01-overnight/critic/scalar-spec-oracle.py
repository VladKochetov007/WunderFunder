#!/usr/bin/env python3
"""Source-blind SCALARPLE01 numeric oracle. Generated values only; stdlib only.

Written from scalar/spec.md without reading scalar_model.py or native code.
Float32 operations are rounded separately through struct, never NumPy/Torch.
The backward oracle implements the specified first-order surrogate; it is not
the derivative of IEEE rounding and supplies no raw-input/knot derivatives.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import struct


def f32(x):
    try:
        return struct.unpack('<f', struct.pack('<f', float(x)))[0]
    except OverflowError:
        return math.copysign(math.inf, float(x))


def bits(x):
    return struct.unpack('<I', struct.pack('<f', f32(x)))[0]


def from_bits(u):
    return struct.unpack('<f', struct.pack('<I', u))[0]


SCALE = f32(1.0 / math.sqrt(44))
FMAX = from_bits(0x7f7fffff)
SMIN = from_bits(1)


def canonical_knots(ordered_values):
    """Only post-quantile canonicalization. Does not estimate any quantile."""
    out = []
    for value in ordered_values:
        value = f32(value)
        if not math.isfinite(value):
            raise ValueError('nonfinite float32 knot')
        value = 0.0 if value == 0 else value
        if out and value < out[-1]:
            raise ValueError('unordered knot input')
        if not out or value != out[-1]:
            out.append(value)
    if not 1 <= len(out) <= 16:
        raise ValueError('knot count outside 1..16')
    return out + [out[-1]] * (16 - len(out)), len(out)


def validate_knots(knots, count):
    if type(count) is not int or not 1 <= count <= 16 or len(knots) != 16:
        raise ValueError('invalid knot count/shape')
    if any(not math.isfinite(k) or f32(k) != k for k in knots):
        raise ValueError('knots must be finite float32')
    if any(k == 0 and bits(k) != 0 for k in knots):
        raise ValueError('knots must canonicalize signed zero')
    if any(float(knots[j+1]) - float(knots[j]) <= 0 for j in range(count-1)):
        raise ValueError('valid knots must be strictly increasing in float64')
    if any(bits(k) != bits(knots[count-1]) for k in knots[count:]):
        raise ValueError('padding must repeat last valid knot')


def locate(x, knots, count):
    validate_knots(knots, count)
    x = f32(x)
    if not math.isfinite(x):
        raise ValueError('nonfinite raw input')
    clamped = min(max(x, knots[0]), knots[count-1])
    lower = 0
    # Last valid knot <= clamped is searchsorted(side='right') - 1.
    for j in range(count):
        if knots[j] <= clamped:
            lower = j
    upper = min(lower + 1, count - 1)
    if lower == upper:
        fraction = 0.0
    else:
        fraction = f32(min(1.0, max(0.0,
            (float(clamped) - float(knots[lower])) /
            (float(knots[upper]) - float(knots[lower])))))
    return lower, upper, fraction


def interpolate(a, b, fraction, direct=False):
    a, b, fraction = f32(a), f32(b), f32(fraction)
    if direct:
        return a
    difference = f32(b - a)
    product = f32(difference * fraction)
    return f32(a + product)


def encode(raw, knots, counts, tables):
    """Single row of 44 selected raw values -> 16-vector plus lookup tuples."""
    if not (len(raw) == len(knots) == len(counts) == len(tables) == 44):
        raise ValueError('expected 44 features')
    accum = [0.0] * 16
    locations = []
    for feature in range(44):
        if len(tables[feature]) != 16 or any(len(v) != 16 for v in tables[feature]):
            raise ValueError('expected 16x16 table per feature')
        lower, upper, frac = locate(raw[feature], knots[feature], counts[feature])
        locations.append((lower, upper, frac))
        for coord in range(16):
            value = interpolate(tables[feature][lower][coord],
                                tables[feature][upper][coord], frac, lower == upper)
            accum[coord] = f32(accum[coord] + value)
    return [f32(value * SCALE) for value in accum], locations


def table_backward(locations_batch, upstream_batch, counts):
    """Literal ordered CPU scatter; GPU parallel scatter need not be bitexact."""
    result = [[[0.0] * 16 for _ in range(16)] for _ in range(44)]
    if len(locations_batch) != len(upstream_batch):
        raise ValueError('batch mismatch')
    for locations, upstream in zip(locations_batch, upstream_batch):
        if len(locations) != 44 or len(upstream) != 16:
            raise ValueError('shape mismatch')
        for feature, (lower, upper, frac) in enumerate(locations):
            if upper != min(lower + 1, counts[feature] - 1):
                raise ValueError('lookup/count state mismatch')
            for coord in range(16):
                scaled = f32(f32(upstream[coord]) * SCALE)
                right = f32(scaled * frac)
                left = f32(scaled - right)
                result[feature][lower][coord] = f32(result[feature][lower][coord] + left)
                result[feature][upper][coord] = f32(result[feature][upper][coord] + right)
    return result


def run_generated_checks():
    checks, observations = [], {}

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    knots, count = canonical_knots([-0.0, 0.0, 1.0, 1 + 2**-25,
                                    1 + 2**-24, 1 + 3 * 2**-25, 1 + 2**-23])
    check('collapse_after_float32_and_positive_zero', count == 3 and
          [bits(k) for k in knots[:count]] == [0, 0x3f800000, 0x3f800001])
    validate_knots(knots, count)
    constant, one = canonical_knots([3.0] * 16)
    check('count_one_all_finite_clamps', all(locate(x, constant, one) == (0, 0, 0.0)
          for x in [-FMAX, -0.0, SMIN, 3.0, FMAX]))
    check('count_one_direct_endpoint_avoids_unused_padding',
          bits(interpolate(-0.0, math.inf, 0, True)) == 0x80000000)

    ordinary, n = canonical_knots([-2.0, 0.0, 1.0, 4.0])
    expected = [(0, 1, 0.0), (0, 1, 0.0), (1, 2, 0.0),
                (2, 3, 0.0), (3, 3, 0.0), (3, 3, 0.0)]
    check('exact_knots_right_ties_and_clamps',
          [locate(x, ordinary, n) for x in [-FMAX, -2.0, 0.0, 1.0, 4.0, FMAX]] == expected)
    adjacent, n = canonical_knots([1.0, from_bits(0x3f800001), from_bits(0x3f800002)])
    check('adjacent_float32_intervals_do_not_collapse', n == 3 and
          locate(from_bits(0x3f800001), adjacent, n) == (1, 2, 0.0))
    subnormal, n = canonical_knots([0.0, SMIN, 2 * SMIN])
    check('subnormal_distinct_gaps_float64', n == 3 and
          locate(SMIN, subnormal, n) == (1, 2, 0.0))
    symmetric, n = canonical_knots([-FMAX, FMAX])
    check('float64_gap_avoids_float32_overflow',
          locate(0.0, symmetric, n) == (0, 1, 0.5) and math.isinf(f32(FMAX - (-FMAX))))
    huge, n = canonical_knots([-FMAX, 0.0])
    check('strict_interior_may_round_fraction_to_one', locate(-SMIN, huge, n) == (0, 1, 1.0))
    tiny, n = canonical_knots([0.0, FMAX])
    check('strict_interior_may_round_fraction_to_zero', locate(SMIN, tiny, n) == (0, 1, 0.0))

    for bad in [math.inf, -math.inf, math.nan]:
        try:
            locate(bad, ordinary, 4)
        except ValueError:
            pass
        else:
            raise AssertionError('nonfinite raw accepted')
        try:
            canonical_knots([bad])
        except ValueError:
            pass
        else:
            raise AssertionError('nonfinite knot accepted')
    checks.append('reject_nonfinite_inputs_and_knots')

    # Counts, valid knots, and padding are one immutable state, not independent hints.
    bad_cases = [(ordinary, 3), (ordinary, 5), (ordinary, 4.0),
                 ([-0.0] + [0.0] * 15, 1), ([0.0, 0.0] + [1.0] * 14, 3)]
    for bad_k, bad_count in bad_cases:
        try:
            validate_knots(bad_k, bad_count)
        except ValueError:
            pass
        else:
            raise AssertionError('inconsistent immutable knot/count accepted')
    checks.append('reject_inconsistent_knots_counts_padding')

    # Deliberately sensitive accumulation order: left fold yields 1, real sum 2.
    sequence = [2.0**24, 1.0, -2.0**24, 1.0] + [0.0] * 40
    knot_rows = [constant[:] for _ in range(44)]
    counts = [1] * 44
    tables = [[[v] * 16 for _ in range(16)] for v in sequence]
    value, locations = encode([3.0] * 44, knot_rows, counts, tables)
    check('ordered_44_feature_sum_not_tree_or_double_sum',
          all(bits(v) == bits(SCALE) for v in value) and f32(math.fsum(sequence)) == 2.0)

    # Benign dyadic gradients isolate the declared table derivative and repeated scatter.
    base, count = canonical_knots([0.0, 1.0, 2.0])
    knot_rows = [base[:] for _ in range(44)]
    counts = [count] * 44
    tables = [[[f32(feature + k / 4 + c / 32) for c in range(16)]
               for k in range(16)] for feature in range(44)]
    inputs = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, 0.5]
    upstream = [[float(sample + 1)] * 16 for sample in range(len(inputs))]
    locs = [encode([x] * 44, knot_rows, counts, tables)[1] for x in inputs]
    gradients = table_backward(locs, upstream, counts)
    check('unused_padding_has_no_gradient',
          all(g == 0 for feature in gradients for row in feature[count:] for g in row))
    # Independently assemble real-number analytic coefficient sums then allow FP32 scatter rounding.
    expected_g = [0.0, 0.0, 0.0]
    for x, u in zip(inputs, upstream):
        if x < 1:
            expected_g[0] += u[0] * (1-x) * SCALE
            expected_g[1] += u[0] * x * SCALE
        elif x < 2:
            expected_g[1] += u[0] * (2-x) * SCALE
            expected_g[2] += u[0] * (x-1) * SCALE
        else:
            expected_g[2] += u[0] * SCALE
    check('repeated_bin_gradient_matches_analytic_coefficients',
          all(abs(gradients[j][k][c] - expected_g[k]) <= 2e-6
              for j in range(44) for k in range(3) for c in range(16)))
    check('per_feature_table_gradient_mass',
          all(abs(sum(row[c] for row in gradients[j]) - sum(u[c] * SCALE for u in upstream)) <= 3e-6
              for j in range(44) for c in range(16)))
    one_locs = [[(0, 0, 0.0)] * 44 for _ in range(3)]
    one_grad = table_backward(one_locs, [[1.0] * 16] * 3, [1] * 44)
    one_expected = f32(f32(SCALE + SCALE) + SCALE)
    check('count_one_scatter_does_not_double_gradient',
          all(bits(feature[0][c]) == bits(one_expected) for feature in one_grad for c in range(16)))

    # Concrete examples: algebraic rewrites do not preserve the FP32 contract.
    rng = random.Random(2026100103)
    rewrite_witness = None
    fma_witness = None
    for _ in range(10000):
        g = f32(rng.uniform(-100, 100))
        frac = f32(rng.random())
        declared = f32(g - f32(g * frac))
        rewritten = f32(g * f32(1.0 - frac))
        if rewrite_witness is None and bits(declared) != bits(rewritten):
            rewrite_witness = dict(g=g, fraction=frac, left_declared=declared,
                                   left_rewritten=rewritten)
        a, b = f32(rng.uniform(-100, 100)), f32(rng.uniform(-100, 100))
        difference = f32(b - a)
        separated = interpolate(a, b, frac)
        # Binary64 exactly holds the F32 product (48 significant bits); this
        # small-magnitude witness also has enough precision for product+add.
        contracted = f32(a + difference * frac)
        if fma_witness is None and bits(separated) != bits(contracted):
            fma_witness = dict(a=a, b=b, fraction=frac,
                               separated=separated, product_add_contracted=contracted)
        if rewrite_witness is not None and fma_witness is not None:
            break
    check('gradient_algebra_rewrite_witness_found', rewrite_witness is not None)
    check('product_add_contraction_witness_found', fma_witness is not None)
    observations['gradient_rewrite_witness'] = rewrite_witness
    observations['fma_witness'] = fma_witness

    signed_zero = f32(-0.0 + 0.0)
    check('zero_residual_numeric_equality_is_not_universal_bitidentity',
          signed_zero == -0.0 and bits(signed_zero) != bits(-0.0))
    observations['signed_zero'] = {'baseline_bits': bits(-0.0), 'plus_zero_bits': bits(signed_zero),
                                   'numeric_equality': signed_zero == -0.0}
    extreme = interpolate(-FMAX, FMAX, 0.5)
    check('finite_table_values_do_not_guarantee_finite_interpolation', math.isinf(extreme))
    observations['finite_table_overflow'] = {'a': -FMAX, 'b': FMAX, 'fraction': 0.5,
                                            'result': 'positive infinity'}
    observations['projection_scope'] = 'Not tested; this oracle covers lookup/interpolation/ordered sum/table backward only.'
    return {'status': 'PASS', 'checks': checks, 'n_checks': len(checks),
            'observations': observations, 'scale_float32': SCALE, 'scale_bits': bits(SCALE)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve()
    root = here.parent.parent
    bound_paths = [root / 'scalar-input-proposal.json', root / 'scalar/spec.md',
                   root / 'amendment-02.json', here]
    report = run_generated_checks()
    report.update({'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'implementation_read': False, 'empirical_inputs_read': False,
                   'gpu_used': False, 'source_sha256': {
                       str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in bound_paths}})
    payload = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(payload)
    print(payload, end='')


if __name__ == '__main__':
    main()
