"""One GPU, sequential jobs; dry-run by default. No trainer import or automatic allocation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
from build_plan import parser_from_source, resolve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc)


def command(argv, **kw):
    return subprocess.run(argv, text=True, capture_output=True, timeout=10, **kw)


def sources(plan):
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT / name) == expected, ('source changed', name)


def gpu_info(gpu):
    r = command(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.total,memory.used,memory.free',
                 '--format=csv,noheader,nounits'], check=True)
    total, used, free = map(int, r.stdout.strip().split(','))
    return {'total_MiB': total, 'used_MiB': used, 'free_MiB': free}


def required_files(plan):
    paths = {'datasets/train.parquet', 'datasets/valid.parquet', 'cache/chain_folds.npz',
             'cache/train_targets.npz', 'cache/valid_targets.npz'}
    for job in plan['jobs']:
        a = job['resolved_args']
        for teacher in a['distill'].split(',') if a['distill'] else []:
            for split in ('train', 'valid'):
                paths.add(f'cache/pred_{teacher}_{split}.npy')
        if a['propensity']:
            paths.add(f"artifacts/runs/{a['propensity']}/model.pt")
            for split in ('train', 'valid'):
                for suffix in ('npy', 'json'):
                    paths.add(f"cache/propcache_{a['propensity']}_{split}.{suffix}")
    return sorted(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--allocation', type=Path)
    ap.add_argument('--execute', action='store_true')
    arg = ap.parse_args()
    path = HERE / 'commands.json'
    plan = json.loads(path.read_text()); sources(plan)
    parser = parser_from_source(); steps = []
    for job in plan['jobs']:
        assert re.fullmatch(r'cont_sh256_s3(?:h2)?_f[01]', job['name'])
        assert job['argv'][:2] == ['.venv/bin/python', 'research/iterations/2026-10-01-overnight/training/train_aux.py']
        assert resolve(parser, job['argv'][2:]) == job['resolved_args']
        steps.append((job, job['stage'], job['argv']))
        if 'OOF_argv' in job:
            expected = ['.venv/bin/python', str((HERE / 'predict_fixed.py').relative_to(ROOT)), job['name'],
                        '--fold', str(job['fold']), '--epoch', '1', '--batch', '16', '--output-dir',
                        'research/iterations/2026-10-01-continuation/cache']
            assert job['OOF_argv'] == expected
            steps.append((job, 'OOF', expected))
    assert len(steps) == 6
    if not arg.execute:
        print(json.dumps({'dry_run': True, 'commands_sha256': sha(path), 'runner_sha256': sha(Path(__file__)),
                          'steps': [{'name': j['name'], 'stage': s, 'argv': v} for j, s, v in steps],
                          'note': 'No allocation, data/dependency, GPU or Linux-scope checks performed; no run/log/output mutation.'}, indent=2))
        return
    assert arg.allocation is not None, '--execute requires a root allocation'
    allocation_sha = sha(arg.allocation)
    a = json.loads(arg.allocation.read_text())
    assert a['execution_authorized'] is True
    assert a['commands_sha256'] == sha(path) and a['runner_sha256'] == sha(Path(__file__))
    deadline = datetime.fromisoformat(a['deadline_utc'])
    assert deadline.utcoffset() is not None and deadline > utc()
    assert type(a['GPU_id']) is int and a['GPU_id'] >= 0
    assert 0 < a['allocator_fraction'] <= .9 and 0 < a['allocator_max_MiB'] < 5632
    assert 0 < a['host_memory_bytes'] <= 20 * 2**30 and a['swap_bytes'] == 0
    assert 0 < a['cpu_quota_percent'] <= 600 and 0 < a['stage_seconds'] <= 6000
    assert a['CPU_ids'] and all(type(x) is int and x >= 0 for x in a['CPU_ids'])
    assert 0 < a['monitor_seconds'] <= 1 and a['minimum_free_MiB'] > 0
    assert 0 <= a['initial_max_used_MiB'] < a['device_total_envelope_MiB'] <= a['GPU_total_MiB']
    for kind, upper in [('parent', 3600), ('H2', 1200), ('OOF', 1200)]:
        assert 0 < a['step_seconds'][kind] <= upper
    inventory = ROOT / a['inventory_receipt']
    assert sha(inventory) == a['inventory_receipt_sha256'] and a['inventory_accepted_by_root'] is True
    for name in required_files(plan):
        assert (ROOT / name).is_file(), ('missing dependency', name)
    for job in plan['jobs']:
        run = ROOT / 'artifacts/runs' / job['name']
        assert not run.exists() and not run.is_symlink(), ('run collision', str(run))
        if 'OOF_argv' in job:
            out = HERE.parent / 'cache' / f"pred_{job['name']}_valid.npy"
            for x in (out, out.with_suffix('.npy.partial'), out.with_suffix('.inputs.json')):
                assert not x.exists() and not x.is_symlink(), ('OOF collision', str(x))
    execution = (ROOT / a['execution_directory']).resolve()
    assert execution.is_relative_to(HERE.resolve()) and execution != HERE.resolve()
    assert not execution.exists(), ('execution directory exists', str(execution))
    execution.mkdir(parents=False)
    (execution / 'allocation.json').write_text(json.dumps(a, indent=2) + '\n')
    env = os.environ.copy(); env.update(plan['environment'])
    env.update(CUDA_VISIBLE_DEVICES=str(a['GPU_id']), WNN_GPU_FRACTION=str(a['allocator_fraction']), PYTHONPATH=str(ROOT / 'src'))
    begin = time.monotonic(); end = begin + min(a['stage_seconds'], (deadline - utc()).total_seconds())
    records = []
    def emit(value):
        value['utc'] = utc().isoformat()
        with (execution / 'events.jsonl').open('a') as f:
            f.write(json.dumps(value) + '\n'); f.flush()
    try:
        for job, stage, argv in steps:
            sources(plan)
            assert sha(path) == a['commands_sha256'] and sha(arg.allocation) == allocation_sha
            assert time.monotonic() < end and utc() < deadline
            info = gpu_info(a['GPU_id'])
            assert info['total_MiB'] == a['GPU_total_MiB']
            assert a['allocator_fraction'] * info['total_MiB'] <= a['allocator_max_MiB']
            assert info['free_MiB'] >= a['minimum_free_MiB'] and info['used_MiB'] <= a['initial_max_used_MiB'], info
            if stage == 'H2':
                parent = job['resolved_args']['init'].split('@ep')[0]
                assert (ROOT / f'artifacts/runs/{parent}/model_ep3.pt').is_file()
            if stage == 'OOF':
                assert (ROOT / job['endpoint']).is_file()
            else:
                run = ROOT / 'artifacts/runs' / job['name']
                assert not run.exists() and not run.is_symlink(), ('run collision', str(run))
            unit = 'student-' + uuid.uuid4().hex
            prior = command(['systemctl', '--user', 'show', unit + '.scope', '-p', 'LoadState', '--value'])
            assert prior.stdout.strip() in ('', 'not-found')
            # Independent child deadline survives supervisor/SSH loss; KILL grace is explicit.
            child_seconds = min(a['step_seconds'][stage], end - time.monotonic(), (deadline - utc()).total_seconds()) - 1
            assert child_seconds > 0
            cmd = ['systemd-run', '--user', '--scope', '--quiet', '--unit', unit,
                   '-p', f"MemoryMax={a['host_memory_bytes']}", '-p', 'MemorySwapMax=0',
                   '-p', f"CPUQuota={a['cpu_quota_percent']}%", '--', 'taskset', '-c',
                   ','.join(map(str, a['CPU_ids'])), 'nice', '-n', '10', '/usr/bin/timeout',
                   '--signal=TERM', '--kill-after=15s', f'{child_seconds:.6f}s', '/usr/bin/time', '-v', *argv]
            log = execution / f"{job['name']}.{stage}.log"; t0 = time.monotonic()
            stop_at = min(end, t0 + a['step_seconds'][stage])
            record = {'name': job['name'], 'stage': stage, 'unit': unit, 'argv': cmd, 'GPU_before': info}
            with log.open('x') as stream:
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                record['launcher_PID'] = process.pid; emit({'event': 'started', **record})
                failure = None
                try:
                    while process.poll() is None:
                        if time.monotonic() >= stop_at or utc() >= deadline:
                            raise RuntimeError('allocated time limit')
                        info = gpu_info(a['GPU_id']); emit({'event': 'GPU_sample', 'unit': unit, **info})
                        if info['used_MiB'] > a['device_total_envelope_MiB']:
                            raise RuntimeError('device-total GPU envelope exceeded; resource contention possible, attribution unknown')
                        time.sleep(min(a['monitor_seconds'], max(0, stop_at - time.monotonic())))
                    assert process.returncode == 0, ('child failed', process.returncode)
                except BaseException as exc:
                    failure = repr(exc)
                    raise
                finally:
                    cleanup_error = None
                    # Only this newly created UUID scope is stopped/killed; never other PIDs/units.
                    try:
                        if failure is not None or process.poll() is None:
                            try:
                                stopped = command(['systemctl', '--user', 'stop', unit + '.scope'])
                                emit({'event': 'owned_scope_stop', 'unit': unit, 'code': stopped.returncode, 'stderr': stopped.stderr})
                                kill_needed = stopped.returncode != 0
                            except subprocess.TimeoutExpired:
                                kill_needed = True
                            if kill_needed:
                                killed = command(['systemctl', '--user', 'kill', '--signal=KILL', unit + '.scope'])
                                emit({'event': 'owned_scope_kill', 'unit': unit, 'code': killed.returncode, 'stderr': killed.stderr})
                            try:
                                process.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                command(['systemctl', '--user', 'kill', '--signal=KILL', unit + '.scope'])
                                process.wait(timeout=15)
                    except Exception as exc:
                        cleanup_error = repr(exc)
                    record.update(seconds=time.monotonic() - t0, exit_code=process.poll(), failure=failure)
                    record['cleanup_error'] = cleanup_error
                    try:
                        state = command(['systemctl', '--user', 'show', unit + '.scope', '-p', 'ActiveState', '--value'])
                        record['scope_state'] = state.stdout.strip()
                    except Exception as exc:
                        record['scope_state'] = 'UNKNOWN'; record['release_check_error'] = repr(exc)
                    records.append(record)
                    emit({'event': 'finished', **record})
            assert record['cleanup_error'] is None and record['scope_state'] in ('', 'inactive', 'failed'), record
        status = 'COMPLETED_4_FITS_2_OOF_NO_METRICS'
    except BaseException:
        status = 'FAILED_STOP_NO_RETRY'
        raise
    finally:
        (execution / 'result.json').write_text(json.dumps({'status': status, 'commands_sha256': sha(path),
            'allocation_sha256': allocation_sha, 'runner_sha256': sha(Path(__file__)),
            'seconds': time.monotonic() - begin, 'jobs': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()
