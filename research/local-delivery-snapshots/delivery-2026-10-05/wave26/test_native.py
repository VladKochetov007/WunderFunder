import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
import json
import numpy as np

R = Path.home()/'WunderFunder-creative'
D = R/'research/iterations/2026-10-05-wave26'
level = sys.argv[1]
sys.path.insert(0, str(D/'native-class/build'/level))
import wnn_cpp
assert wnn_cpp.CLASS_HEAD_SUPPORTED
probe = np.load(D/'fixtures-probe.npz')
results = []
for variant in ('uniform', 'soft', 'saturated', 'zero'):
    w = {k: np.ascontiguousarray(v, np.float32) for k, v in np.load(D/f'fixture-{variant}.npz').items()}
    fp = wnn_cpp.Ensemble([w], [1.], dynamic_int8=False)
    got = np.stack([fp.run(x) for x in probe['x']])
    err = float(abs(got-probe[variant]).max())
    assert err < .001, (variant, level, err)
    first = None
    for backend in ('scalar', 'avx2', 'vnni'):
        model = wnn_cpp.Ensemble([w], [1.], dynamic_int8=True, quant_backend=backend)
        got = np.stack([model.run(x) for x in probe['x']])
        assert np.isfinite(got).all()
        assert np.array_equal(got[0], model.run(probe['x'][0]))
        ref = probe[variant].reshape(-1, 2)
        corr = [float(np.corrcoef(ref[:, k], got.reshape(-1, 2)[:, k])[0, 1]) for k in (0, 1)]
        assert min(corr) > .999, (variant, level, corr)
        if first is None:
            first = got
        else:
            assert np.array_equal(first, got)
        if variant == 'zero':
            plain = {k: v for k, v in w.items() if not k.startswith(('meta.class_head', 'class_gate.', 'class_res.'))}
            old_path = wnn_cpp.Ensemble([plain], [1.], dynamic_int8=True, quant_backend=backend)
            assert np.array_equal(got, np.stack([old_path.run(x) for x in probe['x']]))
        if variant == 'uniform':
            folded = {k: v.copy() for k, v in w.items() if not k.startswith(('meta.class_head', 'class_gate.', 'class_res.'))}
            folded['head.2.weight'] += .5*w['class_res.weight'].reshape(4, 2, -1).sum(0)
            folded['head.2.bias'] += .5*w['class_res.bias'].reshape(4, 2).sum(0)
            equivalent = wnn_cpp.Ensemble([folded], [1.], dynamic_int8=True, quant_backend=backend)
            folded_pred = np.stack([equivalent.run(x) for x in probe['x']])
            assert abs(got-folded_pred).max() < 5e-5
        results.append({'variant': variant, 'backend': backend, 'fp32_max_abs': err,
                        'int8_correlation': corr, 'reset_and_backend_parity': True})
    # Reject incomplete, malformed or nonfinite new metadata instead of silently ignoring it.
    invalid = []
    for key in ('meta.class_head_mode', 'class_res.weight', 'class_res.bias', 'class_gate.weight', 'class_gate.bias'):
        item = dict(w)
        del item[key]
        invalid.append(item)
    for value in (np.array([2.], np.float32), np.array([np.nan], np.float32)):
        invalid.append(dict(w, **{'meta.class_head_mode': value}))
    invalid.append(dict(w, **{'class_gate.weight': w['class_gate.weight'][:3]}))
    invalid.append(dict(w, **{'class_res.bias': np.full(8, np.nan, np.float32)}))
    invalid.append(dict(w, **{'meta.linear_activations': np.array([0., 1.], np.float32)}))
    for bad in invalid:
        try:
            wnn_cpp.Ensemble([bad], [1.], dynamic_int8=True)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed class metadata accepted')
(D/f'parity-{level}.json').write_text(json.dumps(results, indent=2))
print(level, 'PASS', len(results), 'cases', flush=True)
