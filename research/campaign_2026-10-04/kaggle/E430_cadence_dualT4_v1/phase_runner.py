from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path('/kaggle/working/experiment')
OUTPUT = Path('/kaggle/working/results')
CAMPAIGN = 'research/campaign_2026-10-04/kaggle/E430_cadence_dualT4_v1'
ARMS = ('chunk1000', 'chunk2000_connected', 'accum2000_trunc1000')
FIT_NAMES = {
    arm: f'c20261004_e430_{arm}_s3_f0_t4v1_fit' for arm in ARMS
}
PROBE_NAMES = {
    arm: f'c20261004_e430_{arm}_s3_f0_t4v1_probe' for arm in ARMS
}
OPERATING_GPU_BYTES = 12 * 1024**3
STOP_GPU_BYTES = 13 * 1024**3
WORKER_RAM_BYTES = 10 * 1024**3
EXPECTED_BATCHES_PER_EPOCH = 373
EPOCHS = 4


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def require_two_t4s() -> list[dict]:
    sys.path.insert(0, str(ROOT / 'scripts'))
    from kaggle_gpu_runner import gpu_inventory
    devices = gpu_inventory()
    if len(devices) != 2 or any('T4' not in item['name'] for item in devices):
        raise RuntimeError('E430 requires exactly two visible Tesla T4 devices')
    if devices[0]['uuid'] == devices[1]['uuid']:
        raise RuntimeError('The two visible T4 UUIDs are not distinct')
    return devices


def launch_runner(names: list[str], output: Path) -> dict:
    command = [sys.executable, str(ROOT / 'scripts/kaggle_gpu_runner.py'),
               '--root', str(ROOT), '--output', str(output)]
    for name in names:
        command.extend(['--spec', str(ROOT / f'{CAMPAIGN}/kaggle_specs/{name}.json')])
    completed = subprocess.run(command, cwd=ROOT, check=False)
    summary_path = output / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if completed.returncode or summary.get('incident') or summary.get('not_started'):
        raise RuntimeError(f'GPU runner failed at {output}')
    expected_concurrency = min(len(names), 2)
    if summary.get('concurrency_admitted') != expected_concurrency or len(summary.get('finished', [])) != len(names):
        raise RuntimeError(f'Incomplete worker set at {output}')
    return summary


def validate_rows(summary: dict, expected: set[str]) -> dict[str, dict]:
    rows = {item['name']: item for item in summary['finished']}
    if set(rows) != expected:
        raise RuntimeError(f'Worker identity mismatch: {sorted(rows)} != {sorted(expected)}')
    for name, row in rows.items():
        if row.get('status') != 'complete' or row.get('returncode') != 0:
            raise RuntimeError(f'Worker did not return 0 with all artifacts: {name}')
        if not 0 < row.get('peak_gpu_bytes', 0) <= OPERATING_GPU_BYTES:
            raise RuntimeError(f'Observed GPU use is missing or exceeds operating limit: {name}')
        if row.get('peak_gpu_bytes', STOP_GPU_BYTES + 1) >= STOP_GPU_BYTES:
            raise RuntimeError(f'GPU stop threshold reached: {name}')
        if row.get('peak_ram_bytes', WORKER_RAM_BYTES + 1) > WORKER_RAM_BYTES:
            raise RuntimeError(f'Worker RSS exceeded 10 GiB: {name}')
        if 'T4' not in row.get('device', {}).get('name', '') or not row.get('device', {}).get('uuid'):
            raise RuntimeError(f'Worker device identity missing or not T4: {name}')
        if row.get('artifacts', {}).get('missing'):
            raise RuntimeError(f'Declared worker artifact missing: {name}')
    return rows


def _history_counters(run_name: str, *, expected_batches: int, arm: str) -> dict:
    run_dir = ROOT / 'artifacts/runs' / run_name
    history = json.loads((run_dir / 'history.json').read_text())
    if not isinstance(history, list) or len(history) != EPOCHS:
        raise RuntimeError(f'History must contain all four epoch records: {run_name}')
    attempts_per_batch = 20 if arm == 'chunk1000' else 10
    epoch_summaries = []
    sample_order_hashes = []
    for expected_epoch, record in enumerate(history):
        if record.get('epoch') != expected_epoch:
            raise RuntimeError(f'Nonconsecutive epoch records: {run_name}')
        diag = record.get('update_diagnostics', {})
        counters = diag.get('epoch', {})
        expected_attempts = attempts_per_batch * expected_batches
        if counters.get('completed_batches') != expected_batches:
            raise RuntimeError(f'Batch cardinality mismatch in epoch {expected_epoch}: {run_name}')
        if counters.get('optimizer_attempts') != expected_attempts:
            raise RuntimeError(f'Optimizer-attempt cardinality mismatch in epoch {expected_epoch}: {run_name}')
        completed = counters.get('optimizer_completed_updates')
        skipped = counters.get('amp_skipped_updates')
        if not isinstance(completed, int) or not isinstance(skipped, int) or completed + skipped != expected_attempts:
            raise RuntimeError(f'Attempt/success/skip counters do not reconcile: {run_name}')
        if counters.get('ema_updates') != expected_attempts:
            raise RuntimeError(f'EMA update counter does not match attempted updates: {run_name}')
        order = diag.get('sample_order_identity')
        if not isinstance(order, dict) or not order.get('loader_rng', {}).get('bytes_sha256'):
            raise RuntimeError(f'Missing sampled-row order evidence: {run_name}')
        sample_order_hashes.append(order['loader_rng']['bytes_sha256'])
        epoch_summaries.append({'epoch': expected_epoch, 'completed_batches': expected_batches,
                                'optimizer_attempts': expected_attempts,
                                'optimizer_completed_updates': completed,
                                'amp_skipped_updates': skipped, 'ema_updates': expected_attempts})
    provenance = json.loads((run_dir / 'provenance.json').read_text())
    if provenance.get('selected_epoch') != 3:
        raise RuntimeError(f'Fixed epoch-3 checkpoint was not selected: {run_name}')
    args = provenance.get('args', {})
    if args and (args.get('checkpoint_policy') != 'fixed' or args.get('checkpoint_epoch') != 3):
        raise RuntimeError(f'Checkpoint policy mismatch in provenance: {run_name}')
    required_files = ('model_ep3.pt', 'model.pt', 'provenance.json', 'history.json', 'training_state.pt')
    if any(not (run_dir / name).is_file() for name in required_files):
        raise RuntimeError(f'Full fit is missing a required artifact: {run_name}')
    return {'epochs': epoch_summaries, 'selected_epoch': 3,
            'sample_order_sha256_by_epoch': sample_order_hashes}


def validate_probe_report(name: str, arm: str) -> dict:
    path = ROOT / f'{CAMPAIGN}/reports/{name}_steps.json'
    report = json.loads(path.read_text())
    if report.get('status') != 'complete' or len(report.get('steps', [])) != 1:
        raise RuntimeError(f'Probe sequence report is incomplete: {name}')
    if report['steps'][0].get('returncode') != 0:
        raise RuntimeError(f'Probe step failed: {name}')
    return _history_counters(PROBE_NAMES[arm], expected_batches=1, arm=arm)


def validate_fit_report(name: str, arm: str) -> dict:
    path = ROOT / f'{CAMPAIGN}/reports/{name}_steps.json'
    report = json.loads(path.read_text())
    if report.get('status') != 'complete' or len(report.get('steps', [])) != 1 or report['steps'][0].get('returncode') != 0:
        raise RuntimeError(f'Fit sequence report is incomplete: {name}')
    counters = _history_counters(FIT_NAMES[arm], expected_batches=EXPECTED_BATCHES_PER_EPOCH, arm=arm)
    counters['sequence_step'] = report['steps'][0]
    return counters


def main() -> None:
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    evidence: dict[str, object] = {
        'protocol': 'all three production-shape probes finish before any fit; baseline and connected fit concurrently; accumulation fit only after both terminate',
        'arms': list(ARMS), 'probe_order': ['chunk1000+chunk2000_connected', 'accum2000_trunc1000'],
        'fit_order': ['chunk1000+chunk2000_connected', 'accum2000_trunc1000'],
        'expected_batches_per_epoch': EXPECTED_BATCHES_PER_EPOCH,
        'epochs': EPOCHS,
        'expected_batches_per_full_fit': EXPECTED_BATCHES_PER_EPOCH * EPOCHS,
        'allocator_bytes_per_worker': 11 * 1024**3,
        'observed_gpu_operating_limit_bytes': OPERATING_GPU_BYTES,
        'sampled_gpu_stop_bytes': STOP_GPU_BYTES,
        'worker_ram_bytes': WORKER_RAM_BYTES,
        'full_fit_timeout_seconds': 12600,
        'promotion_rule': 'only full four-epoch exit-zero fits with complete declared artifacts and full expected cardinalities enter later scoring; partial probes never promote',
    }
    try:
        devices = require_two_t4s()
        evidence['visible_devices'] = devices
        probe_pair_names = [PROBE_NAMES['chunk1000'], PROBE_NAMES['chunk2000_connected']]
        pair_summary = launch_runner(probe_pair_names, OUTPUT / 'probe_pair')
        pair_rows = validate_rows(pair_summary, set(probe_pair_names))
        if len({pair_rows[name]['device']['uuid'] for name in probe_pair_names}) != 2:
            raise RuntimeError('Baseline and connected probes did not use distinct T4 devices')
        probe_uuid = {
            'chunk1000': pair_rows[PROBE_NAMES['chunk1000']]['device']['uuid'],
            'chunk2000_connected': pair_rows[PROBE_NAMES['chunk2000_connected']]['device']['uuid'],
        }
        probe_single_name = PROBE_NAMES['accum2000_trunc1000']
        single_summary = launch_runner([probe_single_name], OUTPUT / 'probe_accum')
        single_rows = validate_rows(single_summary, {probe_single_name})
        probe_uuid['accum2000_trunc1000'] = single_rows[probe_single_name]['device']['uuid']
        evidence['probe_workers'] = {
            name: {'device': row['device'], 'peak_gpu_bytes': row['peak_gpu_bytes'],
                   'peak_ram_bytes': row['peak_ram_bytes'], 'status': row['status'],
                   'returncode': row['returncode'], 'artifacts': row['artifacts']}
            for summary in (pair_summary, single_summary) for row in summary['finished'] for name in (row['name'],)
        }
        evidence['probe_reports'] = {
            arm: validate_probe_report(PROBE_NAMES[arm], arm) for arm in ARMS
        }
        evidence['probe_uuid_by_arm'] = probe_uuid
        # Mirror probe ordering to preserve the probe-to-fit physical T4 identity binding.
        fit_pair_names = [FIT_NAMES['chunk1000'], FIT_NAMES['chunk2000_connected']]
        fit_pair_summary = launch_runner(fit_pair_names, OUTPUT / 'fit_pair')
        fit_pair_rows = validate_rows(fit_pair_summary, set(fit_pair_names))
        fit_pair_uuid = {
            'chunk1000': fit_pair_rows[FIT_NAMES['chunk1000']]['device']['uuid'],
            'chunk2000_connected': fit_pair_rows[FIT_NAMES['chunk2000_connected']]['device']['uuid'],
        }
        if any(fit_pair_uuid[arm] != probe_uuid[arm] for arm in ('chunk1000', 'chunk2000_connected')):
            raise RuntimeError('Fit UUIDs differ from corresponding completed probe UUIDs')
        evidence['fit_pair_workers'] = {
            row['name']: {'device': row['device'], 'peak_gpu_bytes': row['peak_gpu_bytes'],
                          'peak_ram_bytes': row['peak_ram_bytes'], 'wall_seconds': row['wall_seconds'],
                          'status': row['status'], 'returncode': row['returncode'],
                          'artifacts': row['artifacts']}
            for row in fit_pair_summary['finished']
        }
        evidence['fit_pair_reports'] = {
            arm: validate_fit_report(FIT_NAMES[arm], arm)
            for arm in ('chunk1000', 'chunk2000_connected')
        }
        # By now both GPUs are idle; runner assigns the solo arm to first UUID, as in its solo probe.
        accum_fit_name = FIT_NAMES['accum2000_trunc1000']
        accum_summary = launch_runner([accum_fit_name], OUTPUT / 'fit_accum')
        accum_rows = validate_rows(accum_summary, {accum_fit_name})
        accum_row = accum_rows[accum_fit_name]
        if accum_row['device']['uuid'] != probe_uuid['accum2000_trunc1000']:
            raise RuntimeError('Accumulation fit UUID differs from its completed solo probe UUID')
        evidence['fit_accum_worker'] = {
            'device': accum_row['device'], 'peak_gpu_bytes': accum_row['peak_gpu_bytes'],
            'peak_ram_bytes': accum_row['peak_ram_bytes'], 'wall_seconds': accum_row['wall_seconds'],
            'status': accum_row['status'], 'returncode': accum_row['returncode'],
            'artifacts': accum_row['artifacts'],
        }
        evidence['fit_accum_report'] = validate_fit_report(accum_fit_name, 'accum2000_trunc1000')
        evidence['status'] = 'fits_complete'
    except BaseException as exc:
        evidence['status'] = 'failed_closed'
        evidence['error_type'] = type(exc).__name__
        evidence['error'] = str(exc)
        evidence['traceback'] = traceback.format_exc()
        raise
    finally:
        evidence['elapsed_seconds'] = time.monotonic() - started
        write_json(OUTPUT / 'phase_report.json', evidence)


if __name__ == '__main__':
    main()
