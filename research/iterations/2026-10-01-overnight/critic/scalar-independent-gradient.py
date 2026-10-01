"""Generated CPU derivative oracle independent of actor interpolation helpers."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
P = HERE.parent / 'scalar'
ROOT = HERE.parents[3]
sys.path[:0] = [str(P), str(ROOT / 'src')]
import numpy as np
import torch
from scalar_model import ScalarAdapter
torch.set_num_threads(1)
started = time.monotonic()
rng = np.random.default_rng(991741)
scale = float(np.float32(1 / np.sqrt(np.float64(44))))
knots = np.empty((44, 16), np.float32)
counts = np.array([1, 2, 3, 16] * 11, dtype=np.int64)
for j, count in enumerate(counts):
    values = np.linspace(-2, 2, count, dtype=np.float64).astype(np.float32)
    if count == 1:
        values[:] = 0
    knots[j, :count] = values
    knots[j, count:] = values[-1]
# Positive adjacent/subnormal intervals are retained rather than epsilon-clamped.
knots[2, :3] = np.array([0, np.nextafter(np.float32(0), np.float32(1)), np.nextafter(np.nextafter(np.float32(0), np.float32(1)), np.float32(1))])
knots[2, 3:] = knots[2, 2]
indices = list(range(22)) + list(range(52, 74))
records = []
for name, n in [('random', 37), ('all_left_clamped', 2048), ('all_right_clamped', 2048), ('exact_knots', 61), ('adjacent_boundaries', 61)]:
    x = rng.normal(size=(n, 211)).astype(np.float32)
    for j, index in enumerate(indices):
        if name == 'all_left_clamped':
            x[:, index] = -np.finfo(np.float32).max
        elif name == 'all_right_clamped':
            x[:, index] = np.finfo(np.float32).max
        elif name in ['exact_knots', 'adjacent_boundaries']:
            v = knots[j, np.arange(n) % counts[j]].copy()
            if name == 'adjacent_boundaries':
                direction = np.where(np.arange(n) % 2, np.float32(np.inf), np.float32(-np.inf))
                v = np.nextafter(v, direction)
            x[:, index] = v
    # Build an explicit linear map from table values to outputs, in float64.
    # Interval search is a literal loop, not torch.searchsorted or actor code.
    basis = np.zeros((44, n, 16), dtype=np.float64)
    for j, index in enumerate(indices):
        count = int(counts[j])
        for row in range(n):
            value = min(max(float(x[row, index]), float(knots[j, 0])), float(knots[j, count - 1]))
            lo = 0
            for upper in range(1, count):
                if value >= float(knots[j, upper]):
                    lo = upper
            hi = min(lo + 1, count - 1)
            alpha = float(np.float32(0 if hi == lo else (value - float(knots[j, lo])) / (float(knots[j, hi]) - float(knots[j, lo]))))
            basis[j, row, lo] += 1 - alpha
            basis[j, row, hi] += alpha
    model = ScalarAdapter(knots, counts)
    table = rng.normal(size=(44, 16, 16)).astype(np.float32)
    with torch.no_grad():
        model.table.copy_(torch.from_numpy(table))
    upstream = rng.normal(size=(n, 16)).astype(np.float32)
    output = model.aggregated(torch.from_numpy(x))
    (output * torch.from_numpy(upstream)).sum().backward()
    expected = np.einsum('jnk,nr->jkr', basis, upstream.astype(np.float64), optimize=False) * scale
    actual = model.table.grad.detach().numpy().astype(np.float64)
    difference = actual - expected
    maximum = float(np.abs(difference).max())
    relative_rms = float(np.sqrt(np.mean(difference**2)) / max(np.sqrt(np.mean(expected**2)), 1e-12))
    assert maximum < 2e-4 and relative_rms < 2e-5, (name, maximum, relative_rms)
    assert np.isfinite(actual).all()
    for j, count in enumerate(counts):
        assert np.count_nonzero(actual[j, count:]) == 0
    # Sum of knot derivatives equals derivative of a shared featurewise offset.
    expected_offset = upstream.astype(np.float64).sum(axis=0) * scale
    conservation = float(np.abs(actual.sum(axis=1) - expected_offset).max())
    assert conservation < 2e-4
    records.append({'case': name, 'rows': n, 'gradient_vs_dense_float64_max_abs': maximum, 'gradient_relative_RMS': relative_rms, 'featurewise_offset_derivative_max_abs': conservation, 'unused_padding_gradient_exact_zero': True})
assert not torch.cuda.is_initialized()
result = {'status': 'PASS', 'scalar_source_sha256': hashlib.sha256((P / 'scalar_model.py').read_bytes()).hexdigest(), 'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'oracle': 'Independently built literal-search triangular basis in float64; no actor interpolation helper used for reference.', 'cases': records, 'scope': 'Mathematical table VJP accuracy, not bitidentity of distinct floating accumulation orders. Generated CPU only.', 'empirical_data_reads': 0, 'GPU_initialized': False, 'forecast_fits': 0, 'WP_queries': 0, 'seconds': time.monotonic() - started}
out = HERE / 'scalar-independent-gradient-result.json'
assert not out.exists()
out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
