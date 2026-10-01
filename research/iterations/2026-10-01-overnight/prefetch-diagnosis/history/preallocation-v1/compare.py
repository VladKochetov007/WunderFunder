"""Exact generated trace comparison and the single frozen maintenance-usefulness gate."""
from pathlib import Path
import hashlib, json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(control_path, reuse_path):
    import numpy as np
    control, reuse = [json.loads(Path(p).read_text()) for p in (control_path, reuse_path)]
    assert control['status'] == reuse['status'] == 'COMPLETE'
    assert control['arm'] == 'per_epoch' and reuse['arm'] == 'reuse'
    for key in ['source_sha256', 'allocation_sha256', 'plan_sha256', 'trainer_sha256',
                'extracted_class_sha256', 'fixture', 'fixture_sha256', 'fixture_total_bytes',
                'initial_rng', 'torch_version', 'cuda_version', 'device_name', 'allocator_backend',
                'allow_tf32', 'deterministic_algorithms', 'GPU_allocator_fraction', 'GPU_boundary_reserved_limit_bytes']:
        assert control[key] == reuse[key], key
    assert control['generated_only'] and reuse['generated_only']
    assert not control['competition_inputs_read'] and not reuse['competition_inputs_read']
    assert control['forecast_fits'] == reuse['forecast_fits'] == control['WP_queries'] == reuse['WP_queries'] == 0
    assert len(control['epochs']) == len(reuse['epochs']) == 4
    assert len({r['stream_id'] for r in control['epochs']}) == 4, 'per-epoch arm did not obtain four distinct streams'
    assert len({r['stream_id'] for r in reuse['epochs']}) == 1, 'reuse arm changed stream'
    for epoch, (left, right) in enumerate(zip(control['epochs'], reuse['epochs'])):
        assert left['epoch'] == right['epoch'] == epoch
        assert left['batch_ids'] == right['batch_ids'] == [i % 2 for i in range(32)]
        assert left['rng'] == right['rng']
        assert left['trace'] == right['trace']
    traces = [Path(p).with_suffix('.npz') for p in (control_path, reuse_path)]
    array_checks = []
    for result, path in zip((control, reuse), traces):
        assert sha(path) == result['trace_file_sha256']
    with np.load(traces[0], allow_pickle=False) as lhs, np.load(traces[1], allow_pickle=False) as rhs:
        assert lhs.files == rhs.files
        for key in lhs.files:
            a, b = lhs[key], rhs[key]
            assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(), key
            assert np.isfinite(a).all()
            array_checks.append({'name': key, 'shape': list(a.shape), 'dtype': str(a.dtype),
                                 'sha256': hashlib.sha256(a.tobytes()).hexdigest()})
    reduction = control['peak_reserved_bytes'] - reuse['peak_reserved_bytes']
    wall_ratio = reuse['loop_wall_seconds'] / control['loop_wall_seconds']
    useful = reduction >= 512 * (1 << 20) and wall_ratio <= 1.03
    return {'status': 'PASS_USEFUL_MAINTENANCE_DIAGNOSTIC' if useful else 'CLOSED_NO_USEFUL_GAIN',
            'exact_trace_parity': True, 'trace_arrays': array_checks,
            'peak_reserved_reduction_bytes': reduction, 'reuse_to_per_epoch_loop_wall_ratio': wall_ratio,
            'peak_reserved_gate_bytes': 512 * (1 << 20), 'maximum_wall_ratio': 1.03,
            'usefulness_gate_PASS': useful, 'no_production_change_authorized': True,
            'control_receipt_sha256': sha(control_path), 'reuse_receipt_sha256': sha(reuse_path),
            'limits': 'one generated workload and one fixed ordered pair; no inference about forecast quality or production throughput'}
