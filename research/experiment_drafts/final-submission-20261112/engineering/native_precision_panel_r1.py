"""Compare existing dynamic-INT8 kernels using the frozen final model weights."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path


def main() -> None:
    """Measure identical causal streams, keeping scientific choices fixed."""
    os.sched_setaffinity(0, {8})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    root = Path('/data/epanov_s/tabdata-final-submission-20261112-r1')
    package = root / 'profiles/precision-r2'
    old = Path('/data/epanov_s/tabdata/runtime/final-submit-all-labeled-20261031-r1')
    sys.path[:0] = [str(package / 'v3'), str(old / 'source/src')]
    import numpy as np
    import wnn_native
    from wnn.data import SequenceFile

    manifest = json.loads((package / 'members.json').read_bytes())
    weights = []
    for item in manifest['members']:
        payload = (package / item['weights']).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == item['weights_sha256']
        with np.load(package / item['weights'], allow_pickle=False) as archive:
            weights.append({key: np.ascontiguousarray(archive[key], dtype=np.float32)
                            for key in archive.files})
    data = SequenceFile('/data/epanov_s/tabdata/data/datasets/valid.parquet')
    sequence = data.read(0, targets=False, mask=False)
    points = [types.SimpleNamespace(seq_ix=sequence.seq_ix, step_in_seq=index,
                                   need_prediction=index >= 99, state=row)
              for index, row in enumerate(sequence.features)]
    baseline = None
    baseline_members = None
    modes = ['q8vnni', 'q8vnni256', 'q8vnni_blocked', 'q8vnni_fused_gru']
    results = []
    for mode in modes:
        try:
            model = wnn_native.FinalTl3Predictor(
                weights, precision=mode, fast_activations=True, serving_policy='full_mix')
        except (ValueError, TypeError) as error:
            results.append({'mode': mode, 'unsupported': str(error)})
            continue
        times = []
        for _ in range(2):
            model.reset()
            output = np.empty((len(points) - 99, 2), dtype=np.float32)
            started = time.perf_counter()
            for index, point in enumerate(points):
                value = model.predict(point)
                if index >= 99:
                    output[index - 99] = value
            times.append(time.perf_counter() - started)
        model.reset()
        members = np.empty((len(points), 4, 2), dtype=np.float32)
        for index, point in enumerate(points):
            members[index] = model.step_members(point)
        assert np.isfinite(output).all() and np.isfinite(members).all()
        if baseline is None:
            baseline = output.copy()
            baseline_members = members.copy()
        record = {
            'mode': mode,
            'effective_precisions': model.effective_precisions,
            'seconds_20000_rows': times,
            'callback_bitwise_identical_to_q8vnni': bool(np.array_equal(output.view(np.uint32), baseline.view(np.uint32))),
            'member_outputs_bitwise_identical_to_q8vnni': bool(np.array_equal(members.view(np.uint32), baseline_members.view(np.uint32))),
            'maximum_callback_difference': float(np.max(np.abs(output - baseline))),
            'maximum_member_difference': float(np.max(np.abs(members - baseline_members))),
        }
        results.append(record)
    result = {
        'schema': 'wnn.frozen-final-kernel-panel.v1',
        'sequence_id': sequence.seq_ix,
        'sequence_rows': len(points),
        'labels_read': False,
        'weights_or_blend_modified': False,
        'native_library_sha256': hashlib.sha256((package / 'v3/wnn_native.abi3.so').read_bytes()).hexdigest(),
        'results': results,
    }
    target = root / 'profiles/kernel-panel-r1.json'
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
