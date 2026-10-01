"""CPU-only load/packing check of the fixed fold-0 endpoint; no inference."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CHECKPOINT = ROOT / 'artifacts/runs/oct02_mingru_s3h2_f0/model_ep1.pt'
CHECKPOINT_SHA = '46f2a12e69038670ddd3d80817e03d5822e9759ee4793e087eb32d5a761488c6'
ADAPTER_SHA = '3cd6766977822b2c4f544f4cf9843f6cc9be68d85cdae7e6eba95ee52a123777'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def worker(out):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert set(os.sched_getaffinity(0)) == {35}
    assert sha(HERE / 'native_adapter.py') == ADAPTER_SHA
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    import numpy as np
    import torch
    from native_adapter import load_checkpoint, NativeMinimal
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_flush_denormal(False)
    flat = load_checkpoint(CHECKPOINT, CHECKPOINT_SHA)
    # Independent explicit ABI order from the frozen native constructor.
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)['model']
    names = ['mean', 'std', 'inp.weight', 'inp.bias', 'head.0.weight',
             'head.0.bias', 'head.2.weight', 'head.2.bias']
    for layer in range(2):
        names += [f'cells.{layer}.{part}.{field}'
                  for part in ('gate', 'candidate') for field in ('weight', 'bias')]
    values = []
    for name in names:
        a = state[name].detach().numpy()
        values.append((a[:2] if name.startswith('head.2.') else a).reshape(-1))
    np.testing.assert_array_equal(flat, np.concatenate(values))
    with NativeMinimal(flat) as model:
        np.testing.assert_array_equal(model.state(), np.zeros((2, 1, 256), np.float32))
        model.reset()
        np.testing.assert_array_equal(model.state(), np.zeros((2, 1, 256), np.float32))
    write(out / 'result.json', {
        'status': 'PASS_TRAINED_CHECKPOINT_LOAD_AND_PACK_ONLY',
        'checkpoint_sha256': CHECKPOINT_SHA, 'adapter_sha256': ADAPTER_SHA,
        'packed_float32_count': int(flat.size),
        'packed_sha256': hashlib.sha256(flat.tobytes()).hexdigest(),
        'checks': ['strict_trained_config_load', 'explicit_ABI_pack_bitexact',
                   'native_constructor_accepts_trained_weights', 'initial_and_reset_state_zero'],
        'inference_rows': 0, 'forecast_metric_queries': 0, 'fits': 0, 'GPU': False,
        'quality_claim': False, 'runtime_qualification': False,
        'limitation': 'No inputs or predictions were evaluated; trained drift and serving qualification remain pending.'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worker', action='store_true')
    args = ap.parse_args()
    out = HERE / 'trained-load-01'
    if args.worker:
        worker(out)
        return
    out.mkdir()
    unit = 'mingru-trained-load-' + uuid.uuid4().hex
    helper = ROOT / 'research/iterations/2026-10-01-context-replay/run.py'
    spec = importlib.util.spec_from_file_location('scope_helpers', helper)
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(ROOT / 'src'))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMBA_NUM_THREADS', 'ARROW_NUM_THREADS', 'WNN_THREADS'):
        env[key] = '1'
    argv = ['systemd-run', '--user', '--scope', '--quiet', '--unit', unit,
            '-p', 'CPUQuota=100%', '-p', f'MemoryMax={4 * 2**30}', '-p', 'MemorySwapMax=0',
            '--', 'taskset', '-c', '35', 'nice', '-n', '10', 'timeout', '--signal=TERM',
            '--kill-after=5s', '20s', '.venv/bin/python', str(Path(__file__).relative_to(ROOT)), '--worker']
    sources = {str(p.relative_to(ROOT)): sha(p)
               for p in (Path(__file__), HERE / 'native_adapter.py', helper, CHECKPOINT)}
    receipt = dict(status='RUNNING_LOAD_ONLY', unit=unit, argv=argv, source_sha256=sources,
                   inference_rows=0, forecast_metric_queries=0, fits=0)
    write(out / 'started.json', receipt)
    start = time.monotonic()
    try:
        with (out / 'stdout.log').open('x') as stdout, (out / 'stderr.log').open('x') as stderr:
            rc = subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr).returncode
        receipt['exit_code'] = rc
        assert rc == 0, rc
        for p, expected in sources.items():
            assert sha(ROOT / p) == expected, p
        receipt.update(status='COMPLETE_LOAD_ONLY', result_sha256=sha(out / 'result.json'))
    except BaseException as exc:
        receipt.update(status='FAILED_NO_RETRY', failure=repr(exc))
        raise
    finally:
        fields, pids = H.scope(unit)
        if pids:
            H.command(['systemctl', '--user', 'stop', unit + '.scope'])
            fields, pids = H.scope(unit)
        receipt.update(scope_state=fields, remaining_scope_pids=sorted(pids),
                       seconds=time.monotonic() - start)
        write(out / 'execution.json', receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
