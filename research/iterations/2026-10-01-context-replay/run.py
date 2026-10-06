"""Run the single planned replay in an owned, resource-limited Linux scope."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def command(argv, **kwargs):
    return subprocess.run(argv, text=True, capture_output=True, timeout=10, **kwargs)


def gpu(gpu_id):
    result = command(['nvidia-smi', f'--id={gpu_id}',
                      '--query-gpu=uuid,memory.total,memory.used,memory.free',
                      '--format=csv,noheader,nounits'], check=True)
    uid, total, used, free = (v.strip() for v in result.stdout.strip().split(','))
    return dict(uuid=uid, total_MiB=int(total), used_MiB=int(used), free_MiB=int(free))


def scope(unit):
    result = command(['systemctl', '--user', 'show', unit + '.scope',
                      '-p', 'ControlGroup', '-p', 'ActiveState', '-p', 'SubState'])
    values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    pids = set()
    if values.get('ControlGroup'):
        directory = Path('/sys/fs/cgroup') / values['ControlGroup'].lstrip('/')
        for path in directory.rglob('cgroup.procs'):
            try:
                pids.update(int(v) for v in path.read_text().split())
            except FileNotFoundError:
                pass
    return values, pids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan_hash = sha(args.plan)
    plan = json.loads(args.plan.read_text())
    assert plan['execution_authorized'] is True
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    resource = plan['resources']
    before = gpu(resource['gpu_id'])
    assert before['uuid'] == resource['gpu_uuid']
    assert before['free_MiB'] >= resource['minimum_initial_free_MiB'], before
    assert resource['allocator_fraction'] * before['total_MiB'] <= resource['allocator_max_MiB']
    output = HERE / 'execution-01'
    output.mkdir()
    unit = 'wunder-context-replay-' + uuid.uuid4().hex
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / 'src'), PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES=str(resource['gpu_id']),
               WNN_GPU_FRACTION=str(resource['allocator_fraction']), WNN_GPU_DUTY='1.0',
               WNN_THREADS='4', WNN_LOADER_WORKERS='2', WNN_LOADER_RAM_GB='2')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMBA_NUM_THREADS', 'ARROW_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '4'
    replay_argv = [plan_hash if value == '{PLAN_SHA256}' else value for value in plan['replay_argv']]
    argv = ['systemd-run', '--user', '--scope', '--quiet', '--unit', unit,
            '-p', 'CPUQuota=400%', '-p', 'MemoryMax=10737418240', '-p', 'MemorySwapMax=0',
            '--', 'taskset', '-c', ','.join(map(str, resource['cpu_ids'])), 'nice', '-n', '10',
            '/usr/bin/timeout', '--signal=TERM', '--kill-after=10s', '900s',
            '/usr/bin/time', '-v', *replay_argv]
    receipt = dict(status='RUNNING', started_utc=datetime.now(timezone.utc).isoformat(),
                   plan_sha256=plan_hash, unit=unit, argv=argv, gpu_before=before,
                   peak_owned_gpu_MiB=0, minimum_free_gpu_MiB=before['free_MiB'], samples=0)
    (output / 'started.json').write_text(json.dumps(receipt, indent=2) + '\n')
    started = time.monotonic()
    process = None
    failure = None
    try:
        with (output / 'replay.log').open('x') as log:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            receipt['launcher_pid'] = process.pid
            while process.poll() is None:
                state, pids = scope(unit)
                current = gpu(resource['gpu_id'])
                app_rows = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                    '--format=csv,noheader,nounits'], check=True).stdout.splitlines()
                owned = 0
                for row in app_rows:
                    uid, pid, used = (value.strip() for value in row.split(','))
                    if uid == current['uuid'] and int(pid) in pids:
                        owned += int(used)
                receipt['peak_owned_gpu_MiB'] = max(receipt['peak_owned_gpu_MiB'], owned)
                receipt['minimum_free_gpu_MiB'] = min(receipt['minimum_free_gpu_MiB'], current['free_MiB'])
                receipt['samples'] += 1
                assert owned <= resource['owned_GPU_MiB_max'], ('owned GPU cap', owned)
                assert current['free_MiB'] >= resource['minimum_running_free_MiB'], current
                assert time.monotonic() - started < 920, 'outer deadline'
                time.sleep(1)
            assert process.returncode == 0, ('replay exit', process.returncode)
        assert sha(args.plan) == plan_hash
        for name, expected in plan['source_sha256'].items():
            assert sha(ROOT / name) == expected, name
        receipt['status'] = 'COMPLETE_REPLAY'
    except BaseException as exc:
        failure = repr(exc)
        receipt['status'] = 'FAILED_PRESERVED'
        raise
    finally:
        if process is not None and process.poll() is None:
            # Only this fresh unit is stopped. Foreign PIDs are never signalled.
            command(['systemctl', '--user', 'stop', unit + '.scope'])
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                command(['systemctl', '--user', 'kill', '--signal=KILL', unit + '.scope'])
                process.wait(timeout=15)
        state, remaining_pids = scope(unit)
        receipt.update(seconds=time.monotonic() - started, failure=failure,
                       exit_code=process.returncode if process is not None else None,
                       scope_state=state, remaining_scope_pids=sorted(remaining_pids),
                       finished_utc=datetime.now(timezone.utc).isoformat(), gpu_after=gpu(resource['gpu_id']))
        (output / 'result.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
