"""Generated CPU correctness only; no repository datasets or trained checkpoints."""
import hashlib
import json
from pathlib import Path
import resource
import time
import torch
from minimal_rnn import MinimalRNN, blocked_scan, streaming_states

torch.set_num_threads(1)
torch.manual_seed(2041)
HERE = Path(__file__).resolve().parent
BEGIN = time.perf_counter()
checks = []


def close(name, actual, expected, atol=2e-5, rtol=2e-5):
    assert torch.isfinite(actual).all(), name
    torch.testing.assert_close(actual.double(), expected.double(), atol=atol, rtol=rtol)
    checks.append({'name': name, 'max_abs': float((actual.double()-expected.double()).abs().max()),
                   'atol': atol, 'rtol': rtol})


for initial_kind in ('zero', 'positive'):
    k = torch.randn(2, 2000, 9)
    v = torch.randn_like(k)
    h0 = torch.zeros(2, 9) if initial_kind == 'zero' else torch.rand(2, 9) + .1
    actual, last = blocked_scan(k, v, h0)
    want, end = streaming_states(k.double(), v.double(), h0.double())
    close('cell_long_double_oracle_' + initial_kind, actual, want)
    pieces, carry = [], h0
    starts = [0, 1, 63, 64, 129, 511, 999, 2000]
    for lo, hi in zip(starts[:-1], starts[1:]):
        part, carry = blocked_scan(k[:, lo:hi], v[:, lo:hi], carry)
        pieces.append(part)
    close('cell_split_' + initial_kind, torch.cat(pieces, 1), actual)
    close('cell_end_' + initial_kind, carry, last)
    for stop in (1, 63, 64, 65, 999):
        prefix, _ = blocked_scan(k[:, :stop], v[:, :stop], h0)
        close(f'prefix_{initial_kind}_{stop}', prefix, actual[:, :stop])
    changed = v.clone(); changed[:, 1001:] += 10
    future, _ = blocked_scan(k, changed, h0)
    assert torch.equal(future[:, :1001], actual[:, :1001]), 'future changed past'

# Literal FP32 recurrence has saturated sigmoid gates; the double oracle tests
# the mathematical recurrence too. Absolute tolerance is explicit at extremes.
k = torch.tensor([-80., -20., 0., 20., 80.]).repeat(2, 2000, 1)
v = torch.tensor([-80., -8., 0., 8., 80.]).repeat(2, 2000, 1)
h0 = torch.tensor([0., .2, .5, 2., 20.]).repeat(2, 1)
extreme, _ = blocked_scan(k, v, h0)
literal, _ = streaming_states(k, v, h0)
double, _ = streaming_states(k.double(), v.double(), h0.double())
close('extreme_FP32_literal', extreme, literal, atol=2e-4)
close('extreme_FP64_oracle', extreme, double, atol=2e-4)

for initial_kind in ('zero', 'positive'):
    k = (.6*torch.randn(2, 131, 7)).requires_grad_()
    v = (.6*torch.randn(2, 131, 7)+.07).requires_grad_()
    h0 = (torch.zeros(2, 7) if initial_kind=='zero' else torch.rand(2, 7)+.2).requires_grad_()
    weight = torch.randn_like(k)
    y, _ = blocked_scan(k, v, h0)
    grad = torch.autograd.grad((y*weight).sum(), (k, v, h0))
    kd, vd, hd = [x.detach().double().requires_grad_() for x in (k, v, h0)]
    yd, _ = streaming_states(kd, vd, hd)
    expected = torch.autograd.grad((yd*weight.double()).sum(), (kd, vd, hd))
    for label, a, b in zip(('gate', 'candidate', 'h0'), grad, expected):
        close(f'VJP_{initial_kind}_{label}', a, b, atol=1e-5, rtol=1e-4)

model = MinimalRNN().eval()
assert sum(p.numel() for p in model.parameters()) == 449282
x = torch.randn(1, 2000, 211)
h0 = torch.rand(2, 1, 256) + .1
with torch.no_grad():
    t0 = time.perf_counter(); full, last = model(x, h0); model_scan_seconds = time.perf_counter()-t0
    pieces, carry = [], h0
    for lo, hi in ((0, 37), (37, 999), (999, 2000)):
        part, carry = model(x[:, lo:hi], carry); pieces.append(part)
    close('model_split', torch.cat(pieces, 1), full)
    close('model_split_state', carry, last)
    rows, carry = [], h0
    t0 = time.perf_counter()
    for row in x.unbind(1):
        out, carry = model.step(row, carry); rows.append(out)
    streaming_seconds = time.perf_counter()-t0
    close('model_stream', torch.stack(rows, 1), full)
    close('model_stream_state', carry, last)
    prefix, _ = model(x[:, :511], h0); close('model_prefix', prefix, full[:, :511])
    perturbed = x.clone(); perturbed[:, 512:] += 50
    future, _ = model(perturbed, h0)
    assert torch.equal(future[:, :512], full[:, :512]), 'model future changed past'
    reset, reset_state = model(x)
    explicit, explicit_state = model(x, torch.zeros_like(h0))
    assert torch.equal(reset, explicit) and torch.equal(reset_state, explicit_state)

# All model parameters and inputs, without truncation, through a short chain.
small = MinimalRNN(n_in=11, hidden=8, layers=2, n_out=2)
inp = torch.randn(2, 23, 11, requires_grad=True); initial = torch.rand(2, 2, 8, requires_grad=True)
coef = torch.randn(2, 23, 2)
y, _ = small(inp, initial)
params = list(small.parameters())
grads = torch.autograd.grad((y*coef).sum(), [inp, initial, *params])
rows, state = [], initial
for row in inp.unbind(1):
    value, state = small.step(row, state); rows.append(value)
ref = torch.stack(rows, 1)
ref_grads = torch.autograd.grad((ref*coef).sum(), [inp, initial, *params])
for i, (a, b) in enumerate(zip(grads, ref_grads)):
    close(f'model_VJP_{i}', a, b, atol=2e-5, rtol=2e-4)

for bad in (-torch.ones(1, 2), torch.full((1, 2), float('nan'))):
    try:
        blocked_scan(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), bad)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid initial state accepted')

result = {'status': 'PASS_GENERATED_CPU_ONLY', 'torch_version': torch.__version__,
          'source_sha256': hashlib.sha256((HERE/'minimal_rnn.py').read_bytes()).hexdigest(),
          'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'seed': 2041, 'scan_block': 64, 'state_dtype': 'float32', 'checks': checks,
          'serving_model_parameters': 449282, 'serving_dense_MACs_per_row': 447744,
          'sequence_length': 2000, 'model_scan_CPU_seconds': model_scan_seconds,
          'model_stream_CPU_seconds': streaming_seconds, 'total_CPU_check_seconds': time.perf_counter()-BEGIN,
          'peak_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'GPU_used': False, 'real_data_or_trained_checkpoints_read': False,
          'forecast_fit_or_metric': False, 'qualification': 'CPU mathematical/numerical prototype only; not GPU memory/speed, native serving speed, or forecast quality.'}
print(json.dumps(result), flush=True)
