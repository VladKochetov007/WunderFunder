"""Execute four ROOT-admitted fixed-E6 PR82 follow-up fits under CLOCK_R1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

R = Path('/data2/lc-epanov_s/tabdata-all-followups-20261112-r1')
S = R / 'source-r2'
D = Path('/data/lc-epanov_s/tabdata-stem-successors-20261111-r1/project')
CLOCK_PATH = Path('/data/lc-epanov_s/tabdata-all-20261112-r1/control/CLOCK.json')
CONTROL_SOURCE = Path('/data2/lc-epanov_s/tabdata/runtime/architecture-execution-20261107-r1/source-gru-monitor-resume-r1/src')
P = '/tmp/tabdata-lc-epanov_s-architecture-20261107-env/bin/python'
OBSERVED = '/data2/lc-epanov_s/tabdata/runtime/generalization-fix-20261108-r1/control/l4-wave-r1/observed_worker.py'
CELLS = (
    'muon_rnn_k16_s203301',
    'muon_rnn_k16_s203302',
    'since32_k16_s203301',
    'since32_k16_s203302',
)
SOURCE_COMMIT = '390c846cbdacd8e0e06e6079dd3b04312fa6bcc5'
SOURCE_SHA = '29fb45164667e69a71d4be2c696731d9824884d2d51674f37d2a633bd17d9511'
SOURCE_MANIFEST_SHA = '00782a64e1557a8af759e6f987038cabb6ec09f098c43ab61d6cfa6bb07e9e5f'
CLOCK_SHA = '0181e9889a45d0636e3b9981ddb15e17fb5cd26429068bb0ee37f78d09d71270'
SPLIT_SHA = '0c4de3a253d23b3a5d5ca73b7aeb7fcae2b974673e4f9ec51e94e593c59d114c'
DATA_SHA = '4987ed5f263cf3c655714a7af3e792605716f6cb256c1c96474eb46ae9869323'
GPU_BY_CELL = {cell: index for index, cell in enumerate(CELLS)}
RAW_CONFIG_SHA = {
    'muon_rnn_k16_s203301': '7b5e46e48d69694c653f2e3f4ae9f87550a758e2471e71e5c60033845ed9e6b3',
    'muon_rnn_k16_s203302': 'a58bd410e0b52b13073fb581d3f10fba83789b4d176ef6f73abc1010af762df4',
    'since32_k16_s203301': 'e5c820ce524aa305cd77cfa784492cb1aec2ac95eeb919a815894f63d10e7553',
    'since32_k16_s203302': '42972bf72573bf84fd55a5086b2bece2412799a74f9070eaf09ca7b688aa0fdf',
}
CONFIG_SHA = {
    'muon_rnn_k16_s203301': 'e30b9d4d64d04af646d7d428aac2b610a53abb10ac85e35a5d693324da6af9b9',
    'muon_rnn_k16_s203302': 'd6b3db0286edb2c1f03c5409d3a2d2a51b709368d1f30e1b0f83e7badcee6a24',
    'since32_k16_s203301': '5f5f104665a052d4de12d6dcabf84c7f5e20b39d08acbfd4bb88092359f35912',
    'since32_k16_s203302': 'fe7233ecdb56f5e933d1a48f2e6e5cef9b6aa08ae3b9730d5125865a0076de60',
}
CONTROL_WP = {'muon_rnn_k16_s203301': 0.683019498, 'since32_k16_s203301': 0.683019498,
              'muon_rnn_k16_s203302': 0.682629372, 'since32_k16_s203302': 0.682629372}
TOTAL_UPDATES = 38760
WARMUP_UPDATES = 1938
MAX_OWN_RSS = 64 * 1024**3
MAX_OUTPUT_BYTES = 1_000_000_000
MIN_FREE_BYTES = 2_000_000_000


def create_root_approved_profile_copy(
    raw_path: Path,
    approved_path: Path,
    expected_raw_sha: str,
    fit_admission_sha: str,
    write_json: Callable[..., None],
) -> tuple[str, str]:
    """Preserve a ROOT-admitted raw profile and write a separate approved copy."""
    if digest(raw_path) != expected_raw_sha:
        raise ValueError('raw profile receipt differs from ROOT fit admission')
    raw_receipt = json.loads(raw_path.read_text())
    if not isinstance(raw_receipt, dict) or raw_receipt.get('approved') is not False:
        raise ValueError('ROOT approval copy requires the original unapproved receipt')
    approved_receipt = dict(raw_receipt)
    approved_receipt['approved'] = True
    approved_receipt['root_fit_approval'] = {
        'schema': 'wnn.root-approved-sequence-profile.v1',
        'approved_by': 'ROOT',
        'fit_admission_sha256': fit_admission_sha,
        'raw_profile_receipt_sha256': expected_raw_sha,
        'approved_at_epoch': time.time(),
    }
    write_json(approved_path, approved_receipt, replace=False)
    reread = json.loads(approved_path.read_text())
    if (
        reread.get('approved') is not True
        or reread.get('identity') != raw_receipt.get('identity')
        or reread.get('root_fit_approval', {}).get('fit_admission_sha256') != fit_admission_sha
        or reread.get('root_fit_approval', {}).get('raw_profile_receipt_sha256') != expected_raw_sha
    ):
        raise ValueError('ROOT-approved profile copy failed provenance verification')
    return expected_raw_sha, digest(approved_path)


def build_train_command(config: Path, output: Path, approved_profile_receipt: Path) -> list[str]:
    """Build the trainer argv using only the ROOT-approved profile copy."""
    return [
        P, str(S / 'scripts/train_sequence.py'), '--config', str(config),
        '--data-root', str(D), '--split-manifest', str(D / 'chronological_primary.json'),
        '--output', str(output), '--profile-receipt', str(approved_profile_receipt),
    ]


def digest(path: Path) -> str:
    """Hash a file in bounded blocks."""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def tree_size(root: Path) -> int:
    """Count regular output bytes without following symlinks."""
    total = 0
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if not (Path(folder) / name).is_symlink()]
        for name in files:
            path = Path(folder) / name
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
    return total


def validate_admission(admission: dict[str, object], profiles_root: Path) -> dict[str, dict[str, object]]:
    """Bind every new fit to one complete R2 profile, exact config/data and device UUID."""
    if (
        admission.get('schema') != 'wnn.l4-followups-root-fit-admission.r2'
        or admission.get('approved') is not True
        or admission.get('fit_launch_approved') is not True
        or admission.get('approved_by') != 'ROOT'
        or admission.get('max_fresh_fits') != 4
        or admission.get('source_commit') != SOURCE_COMMIT
        or admission.get('source_sha256') != SOURCE_SHA
        or admission.get('source_manifest_sha256') != SOURCE_MANIFEST_SHA
        or admission.get('split_sha256') != SPLIT_SHA
        or admission.get('data_sha256') != DATA_SHA
        or admission.get('clock_sha256') != CLOCK_SHA
        or set(admission.get('profiles', {})) != set(CELLS)
    ):
        raise ValueError('ROOT fit admission differs from this frozen four-fit R2 contract')
    validated: dict[str, dict[str, object]] = {}
    for cell in CELLS:
        spec = admission['profiles'][cell]
        if not isinstance(spec, dict):
            raise TypeError(f'fit profile admission is malformed: {cell}')
        profile_path = profiles_root / cell / 'profile' / 'profile_receipt.json'
        if not profile_path.is_file() or digest(profile_path) != spec.get('profile_receipt_sha256'):
            raise ValueError(f'profile receipt hash differs from ROOT fit admission: {cell}')
        receipt = json.loads(profile_path.read_text())
        identity = receipt['identity']
        gpu = int(spec['gpu'])
        gpu_uuid = identity['runtime']['devices'][0]['uuid']
        if (
            receipt.get('approved') is not False
            or receipt.get('status') != 'profile_complete'
            or receipt.get('updates') != 30
            or receipt.get('errors') != []
            or identity.get('source_sha256') != SOURCE_SHA
            or identity.get('config_sha256') != CONFIG_SHA[cell]
            or identity.get('data_sha256') != DATA_SHA
            or identity['data'].get('split_sha256') != SPLIT_SHA
            or identity.get('world_size') != 1
            or identity['runtime']['world_size'] != 1
            or gpu != GPU_BY_CELL[cell]
            or spec.get('gpu_uuid') != gpu_uuid
            or spec.get('config_sha256') != CONFIG_SHA[cell]
            or spec.get('raw_config_sha256') != RAW_CONFIG_SHA[cell]
            or spec.get('data_sha256') != DATA_SHA
            or spec.get('source_sha256') != SOURCE_SHA
            or spec.get('total_updates') != TOTAL_UPDATES
            or spec.get('warmup_updates') != WARMUP_UPDATES
            or receipt.get('peak_reserved_bytes', MAX_OWN_RSS + 1) > 3 * 1024**3
        ):
            raise ValueError(f'profile does not satisfy ROOT exact-config fit admission: {cell}')
        validated[cell] = spec
    return validated


def main() -> None:
    """Run only after ROOT records exact profile receipts and fit-slot headroom."""
    sys.path.insert(0, str(CONTROL_SOURCE))
    from wnn.multistage_search import atomic_write_json, file_lock
    from wnn.multistage_search_owned_command import run_owned_command
    from wnn.multistage_search_runtime import (
        current_process_owner,
        is_same_process,
        launch_detached_worker,
        terminate_owned_process,
    )
    from wnn.training.sequence_observer_resources import (
        process_table,
        tracked_processes,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--detach', action='store_true')
    parser.add_argument('--admission-sha', required=True)
    args = parser.parse_args()
    control = R / 'control/root-fits-r4'
    control.mkdir(parents=True, exist_ok=True)
    (R / 'scratch').mkdir(exist_ok=True)
    (R / 'receipts').mkdir(exist_ok=True)
    if args.detach:
        print(json.dumps(launch_detached_worker(
            [P, str(Path(__file__).resolve()), '--admission-sha', args.admission_sha],
            cwd=R,
            env=dict(os.environ),
            receipt_path=control / 'owner.json',
            metadata={'new_fits': 4, 'epochs_each': 6, 'source_sha256': SOURCE_SHA},
        )))
        return
    with file_lock(control / 'fits.lock', blocking=False):
        admission_path = R / 'ROOT-fit-admission-r2.json'
        if digest(admission_path) != args.admission_sha:
            raise ValueError('ROOT fit-admission SHA does not match --admission-sha')
        admission = json.loads(admission_path.read_text())
        contention_path = R / 'receipts/fit-headroom-r2.json'
        contention = json.loads(contention_path.read_text())
        if (
            contention.get('status') != 'passed'
            or contention.get('schema') != 'wnn.l4-followups-contention-clearance.r2'
            or contention.get('candidate_fit_slots') != 4
            or contention.get('first_cohort_progress_verified') is not True
            or admission.get('contention_receipt_sha256') != digest(contention_path)
            or admission.get('contention_receipt_path') != 'receipts/fit-headroom-r2.json'
        ):
            raise ValueError('ROOT fit admission is missing the measured contention clearance')
        clock_sha = digest(CLOCK_PATH)
        if clock_sha != CLOCK_SHA:
            raise ValueError('the supplied CLOCK_R1 bytes changed')
        clock = json.loads(CLOCK_PATH.read_text())
        if admission.get('fit_deadline_epoch') != clock['fit_deadline_epoch']:
            raise ValueError('fit admission deadline differs from CLOCK_R1')
        if admission.get('wave_deadline_epoch') != clock['wave_deadline_epoch']:
            raise ValueError('fit admission collection deadline differs from CLOCK_R1')
        if time.time() >= clock['fit_deadline_epoch']:
            raise TimeoutError('fixed fit deadline reached; no candidate fit launched')
        verify = runpy.run_path(str(S / 'src/wnn/training/sequence_source_capsule.py'))[
            'verify_source_capsule'
        ]
        verify(S, S / 'SOURCE_MANIFEST.json', SOURCE_MANIFEST_SHA, SOURCE_COMMIT)
        profiles_root = R / 'runs'
        admitted = validate_admission(admission, profiles_root)
        if (
            contention.get('profile_receipt_sha256')
            != {cell: admitted[cell]['profile_receipt_sha256'] for cell in CELLS}
            or contention.get('gpu_uuid_by_cell')
            != {cell: admitted[cell]['gpu_uuid'] for cell in CELLS}
        ):
            raise ValueError('contention clearance is not bound to these four profile receipts/devices')
        approved_profiles_root = R / 'receipts/approved-profiles-r4'
        approved_profiles_root.mkdir(parents=True, exist_ok=False)
        for cell in CELLS:
            raw_path = profiles_root / cell / 'profile' / 'profile_receipt.json'
            approved_path = approved_profiles_root / f'{cell}.json'
            raw_sha, approved_sha = create_root_approved_profile_copy(
                raw_path,
                approved_path,
                str(admitted[cell]['profile_receipt_sha256']),
                args.admission_sha,
                atomic_write_json,
            )
            admitted[cell]['raw_profile_receipt_sha256'] = raw_sha
            admitted[cell]['approved_profile_receipt_path'] = str(approved_path)
            admitted[cell]['approved_profile_receipt_sha256'] = approved_sha
        owner = current_process_owner()
        atomic_write_json(R / 'receipts/first-fit-r4.json', {
            'clock_sha256': CLOCK_SHA,
            'fit_dispatch_epoch': time.time(),
            'source_commit': SOURCE_COMMIT,
            'source_sha256': SOURCE_SHA,
            'fit_admission_sha256': args.admission_sha,
            'raw_profile_receipt_sha256': {
                cell: admitted[cell]['raw_profile_receipt_sha256'] for cell in CELLS
            },
            'approved_profile_receipt_sha256': {
                cell: admitted[cell]['approved_profile_receipt_sha256'] for cell in CELLS
            },
        }, replace=False)
        lock = threading.Lock()
        resource_errors: list[str] = []
        last_guard = [0.0]

        def guard() -> None:
            """Latch deadline/RSS/disk/output caps; identify only owned children."""
            with lock:
                if resource_errors:
                    raise RuntimeError(resource_errors[0])
                if time.time() >= clock['fit_deadline_epoch']:
                    resource_errors.append('fixed fit deadline reached')
                if time.monotonic() - last_guard[0] >= 10:
                    last_guard[0] = time.monotonic()
                    identities = {}
                    for receipt_path in (control / 'owned').glob('*.json'):
                        child = json.loads(receipt_path.read_text()).get('child')
                        if isinstance(child, dict) and is_same_process(child):
                            identities[int(child['pid'])] = int(child['birth'])
                    processes = tracked_processes(process_table(), identities)
                    rss = sum(row.resident_bytes for row in processes.values())
                    output_bytes = tree_size(R)
                    free = shutil.disk_usage(R).free
                    if rss > MAX_OWN_RSS or output_bytes > MAX_OUTPUT_BYTES or free < MIN_FREE_BYTES:
                        resource_errors.append(
                            f'resource cap: own_rss={rss} output_bytes={output_bytes} free_bytes={free}'
                        )
                    atomic_write_json(control / 'resource_status.json', {
                        'at_epoch': time.time(), 'own_rss_bytes': rss,
                        'output_bytes': output_bytes, 'free_bytes': free,
                        'errors': resource_errors,
                    })
                if resource_errors:
                    raise RuntimeError(resource_errors[0])

        def fit(cell: str) -> dict[str, object]:
            """Launch one newly admitted fixed-E6 fit and keep its terminal receipt."""
            spec = admitted[cell]
            gpu = int(spec['gpu'])
            config = R / 'configs' / f'{cell}.json'
            if digest(config) != RAW_CONFIG_SHA[cell]:
                raise ValueError(f'raw config hash changed for {cell}')
            output = R / 'runs' / cell / 'fit'
            if output.exists():
                raise FileExistsError(f'fit output already exists; preserving prior attempt: {cell}')
            atomic_write_json(control / f'claim-{cell}.json', {
                'cell': cell, 'profile': spec, 'admission_sha256': args.admission_sha,
                'raw_profile_receipt_sha256': spec['raw_profile_receipt_sha256'],
                'approved_profile_receipt_sha256': spec['approved_profile_receipt_sha256'],
                'owner': owner,
            }, replace=False)
            env = dict(
                os.environ,
                PYTHONPATH=str(S / 'src'),
                CUDA_VISIBLE_DEVICES=f"GPU-{str(spec['gpu_uuid']).removeprefix('GPU-')}",
                OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
                NVIDIA_TF32_OVERRIDE='0', TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='0',
                PYTHONDONTWRITEBYTECODE='1', WNN_MONITOR_HEARTBEAT_SECONDS='10',
                TMPDIR=str(R / 'scratch'),
            )
            for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'CUBLAS_WORKSPACE_CONFIG', 'CUBLAS_EMULATE_SINGLE_PRECISION'):
                env.pop(key, None)
            train = build_train_command(
                config, output, Path(str(spec['approved_profile_receipt_path']))
            )
            bootstrap = (
                'import sys,runpy;sys.path.insert(0,' + repr(str(CONTROL_SOURCE))
                + ');runpy.run_path(' + repr(OBSERVED) + ',run_name="__main__")'
            )
            seconds = int(clock['fit_deadline_epoch'] - time.time())
            if seconds < 1:
                raise TimeoutError('fixed fit deadline reached before candidate launch')
            command = [
                P, '-c', bootstrap, '--', P, str(S / 'scripts/observe_sequence.py'),
                '--output', str(output), '--interval', '10', '--max-seconds', str(seconds),
                '--', *train,
            ]
            receipt_path = control / 'owned' / f'{cell}.json'
            failures: list[str] = []

            def progress() -> None:
                try:
                    guard()
                except (RuntimeError, TimeoutError, OSError, ValueError) as error:
                    if failures:
                        return
                    failures.append(str(error))
                    if receipt_path.exists():
                        record = json.loads(receipt_path.read_text())
                        child = record.get('child')
                        if record.get('command') == command and record.get('owner') == {
                            'controller': owner, 'cell_id': cell,
                        } and isinstance(child, dict) and is_same_process(child):
                            terminate_owned_process(child, grace_seconds=0.0)

            code, timed_out, _ = run_owned_command(
                command, cwd=S, env=env, log_path=control / f'{cell}.log',
                receipt_path=receipt_path, owner={'controller': owner, 'cell_id': cell},
                stage=1, trial_number=gpu, mode='fit', gpu_key=str(gpu),
                output_dir=output, timeout_seconds=seconds, progress=progress,
            )
            result: dict[str, object] = {
                'cell': cell, 'exit_code': code, 'timed_out': timed_out,
                'resource_errors': failures, 'fit_admission_sha256': args.admission_sha,
                'raw_profile_receipt_sha256': spec['raw_profile_receipt_sha256'],
                'approved_profile_receipt_sha256': spec['approved_profile_receipt_sha256'],
                'at_epoch': time.time(),
            }
            if code == 0 and not timed_out and not failures:
                terminal = json.loads((output / 'terminal_status.json').read_text())
                events = [
                    json.loads(line)
                    for line in (output / 'monitor_evaluations.jsonl').read_bytes().splitlines()
                ]
                assert terminal['status'] == 'completed'
                assert terminal['completed_epochs'] == 6
                assert terminal['global_updates'] == TOTAL_UPDATES
                assert terminal['model_sha256'] == digest(output / 'model.pt')
                assert len(events) == 6
                epoch_readout = []
                for epoch, event in enumerate(events, start=1):
                    metrics = event['metrics']
                    assert event['epoch'] == epoch
                    assert metrics['valid/sequence_count'] == 1873
                    assert metrics['valid/selected_rows'] == 3_528_273
                    assert metrics['train_eval/sequence_count'] == 32
                    assert event['context']['train_eval_mask'] == 'all_after_warmup'
                    assert event['context']['evaluation_mask'] == 'is_scored after warmup'
                    assert event['context']['evaluation_member_count'] == 1873
                    assert event['context']['checkpoint_selection'] == 'fixed final epoch 6; monitoring is report only'
                    assert all(key in metrics for key in ('valid/wp', 'valid/wp/t0', 'valid/wp/t1'))
                    row: dict[str, object] = {
                        'epoch': epoch,
                        'step': event['step'],
                        'valid_wp': metrics['valid/wp'],
                        'valid_wp_t0': metrics['valid/wp/t0'],
                        'valid_wp_t1': metrics['valid/wp/t1'],
                        'valid_selected_rows': metrics['valid/selected_rows'],
                    }
                    if cell.startswith('since32_'):
                        for age in ('lt32', 'gte32'):
                            prefix = f'valid/wp/since_{age}'
                            available = metrics[f'{prefix}/available']
                            row[f'{age}_available'] = available
                            row[f'{age}_selected_rows'] = metrics[f'{prefix}/selected_rows']
                            row[f'{age}_wp'] = metrics.get(prefix) if available else None
                            row[f'{age}_wp_t0'] = metrics.get(f'{prefix}/t0') if available else None
                            row[f'{age}_wp_t1'] = metrics.get(f'{prefix}/t1') if available else None
                            if not available:
                                assert not any(key in metrics for key in (prefix, f'{prefix}/t0', f'{prefix}/t1'))
                    epoch_readout.append(row)
                index = json.loads((output / 'epoch_index.json').read_text())
                assert index['completed_epochs'] == 6 and len(index['entries']) == 6
                for entry in index['entries']:
                    checkpoint = output / entry['checkpoint_path']
                    assert checkpoint.is_file() and digest(checkpoint) == entry['checkpoint_sha256']
                final = epoch_readout[-1]
                result.update(
                    status='fit_complete',
                    model_sha256=terminal['model_sha256'],
                    total_updates=terminal['global_updates'],
                    final_wp=final['valid_wp'],
                    final_wp_t0=final['valid_wp_t0'],
                    final_wp_t1=final['valid_wp_t1'],
                    paired_delta_vs_same_seed_k16=float(final['valid_wp']) - CONTROL_WP[cell],
                    epoch_readout=epoch_readout,
                    final_metrics=events[-1]['metrics'],
                    epoch_index_sha256=digest(output / 'epoch_index.json'),
                )
            else:
                result['status'] = 'failed_or_partial'
            atomic_write_json(control / f'{cell}-result.json', result, replace=False)
            print(json.dumps(result), flush=True)
            return result

        (control / 'owned').mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(fit, CELLS))
        completed = sum(item.get('status') == 'fit_complete' for item in results)
        atomic_write_json(control / 'complete.json', {
            'fit_count': 4,
            'fit_complete': completed,
            'all_complete': completed == 4,
            'at_epoch': time.time(),
        }, replace=False)
        if completed != 4:
            raise RuntimeError('one or more candidate fits failed or were partial; preserve receipts')


if __name__ == '__main__':
    main()
