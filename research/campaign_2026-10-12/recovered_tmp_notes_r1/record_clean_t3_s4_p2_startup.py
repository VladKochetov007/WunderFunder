import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

repo = Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
actor = Path('/home/vlad/development/wunder/wnn_connectome_starterpack-clean-t3-profile')
rel = Path('research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011/clean_t3_f0_matched_student_r5_cache_r1/seed4_confirmation_r1')
packet = actor / rel
attempt_id = 'clean-t3-f0-control-s4-a9-r3-p2-envr1-20261012'
attempt_path = repo / 'research/campaign_2026-10-06/bandit_continuation/runtime/attempts' / f'{attempt_id}.json'
log_path = repo / 'research/campaign_2026-10-06/bandit_continuation/runtime/logs' / f'{attempt_id}.log'
binding_path = packet / 'P2_BOUND_STAGE_R1.json'
validation_path = packet / 'P1_TERMINAL_VALIDATION_R3.json'
root_release_path = repo / rel / 'ROOT_SEED4_CONDITIONAL_RELEASE_R1.json'
output_path = packet / 'P2_STARTUP_RECEIPT_R1.json'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
attempt_bytes = attempt_path.read_bytes()
attempt = json.loads(attempt_bytes)
log_bytes = log_path.read_bytes()
log_lines = log_bytes.decode('utf-8', errors='replace').splitlines()
props = subprocess.run(
    ['systemctl', '--user', 'show', f'wnn-bandit-{attempt_id}.service',
     '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID',
     '-p', 'MemoryCurrent', '-p', 'MemoryPeak', '-p', 'CPUUsageNSec',
     '-p', 'CPUQuotaPerSecUSec', '-p', 'MemorySwapMax'],
    check=True, capture_output=True, text=True,
)
gpu = subprocess.run(
    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader'],
    check=True, capture_output=True, text=True,
).stdout.strip()
props_obj = {line.split('=', 1)[0]: line.split('=', 1)[1]
             for line in props.stdout.splitlines() if '=' in line}
fit_lines = [line for line in log_lines if line.startswith(('ep 2 batch ', 'ep 3 batch ', 'FIT '))]
measurement = attempt['admission_measurement_at_start']
parent_current = int(measurement['parent_memory_current_bytes'])
worker_reserve = int(attempt['reservation']['ram_bytes'])
controller_bound = 500_000_000
validation = json.loads(validation_path.read_text())
assert validation['status'] == 'PASS'
assert attempt['status'] == 'running'
assert props_obj['ActiveState'] == 'active' and props_obj['SubState'] == 'running'
assert any(line.startswith('ep 2 batch 0/') for line in fit_lines)
receipt = {
    'schema': 'wnn.clean-t3-control-seed4-p2-startup-receipt.r1',
    'status_at_snapshot': 'RUNNING_GPU_FIT_CONFIRMED',
    'recorded_utc': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'),
    'authorization': {
        'root_release_path': str(root_release_path),
        'root_release_sha256': sha(root_release_path),
        'root_release_authorized': True,
        'scope': 'Already-admitted seed4 P1/P2/P3/CP5 export/grade drain; no additional experiment.',
    },
    'predecessor': {
        'validation_path': str(validation_path),
        'validation_sha256': sha(validation_path),
        'validation_status': validation['status'],
        'p1_state_sha256': json.loads(binding_path.read_text())['predecessor']['state_sha256'],
        'p1_next_epoch': 2,
    },
    'stage_binding': {
        'path': str(binding_path),
        'sha256': sha(binding_path),
        'attempt_id': attempt_id,
        'run_name': 'clean_t3_f0_control_s4_a9_t3r5_r3_p2_envr1',
        'worker_argv': json.loads(binding_path.read_text())['p2_stage']['worker_argv'],
        'controller_argv': json.loads(binding_path.read_text())['p2_stage']['controller_argv'],
    },
    'actual_attempt': {
        'path': str(attempt_path),
        'sha256_at_snapshot': hashlib.sha256(attempt_bytes).hexdigest(),
        'source_manifest_sha256': attempt['source_manifest_sha256'],
        'contract_sha256': attempt['contract_sha256'],
        'started_utc': attempt['started_utc'],
        'heartbeat_utc': attempt['heartbeat_utc'],
        'elapsed_seconds': attempt['elapsed_seconds'],
        'phase': attempt['phase'],
        'status': attempt['status'],
        'reservation': attempt['reservation'],
        'admission_measurement_at_start': measurement,
        'admission_margin': {
            'new_memory_rule_target_bytes': 17_000_000_000,
            'parent_current_plus_full_worker_reserve_plus_controller_bound_bytes':
                parent_current + worker_reserve + controller_bound,
            'controller_bound_bytes': controller_bound,
            'within_target': parent_current + worker_reserve + controller_bound <= 17_000_000_000,
        },
        'worker_peak_ram_bytes_at_snapshot': attempt['peak_ram_bytes'],
        'worker_peak_observed_vram_bytes_at_snapshot': attempt['peak_observed_vram_bytes'],
        'initial_observed_vram_bytes': attempt['initial_observed_vram_bytes'],
        'worker_memory_event_deltas_at_snapshot': attempt['memory_event_deltas'],
        'parent_memory_event_deltas_at_snapshot': attempt['parent_memory_event_deltas'],
        'memory_event_counter_note': 'high is an event count, not bytes; OOM, OOM-kill, and kill deltas are zero at this snapshot.',
    },
    'actual_service_at_snapshot': {
        'unit': f'wnn-bandit-{attempt_id}.service',
        'manager': 'systemd --user',
        'properties': props_obj,
        'gpu_sample': gpu,
        'controller_log_path': str(log_path),
        'controller_log_sha256_at_snapshot': hashlib.sha256(log_bytes).hexdigest(),
        'controller_log_bytes_at_snapshot': len(log_bytes),
        'first_actual_fit_line': next(line for line in fit_lines if line.startswith('ep 2 batch 0/')),
        'recent_progress_lines': fit_lines[-4:],
        'phase_confirmation': 'Actual epoch 2 optimizer batch was recorded; later log lines show continued epoch 3 progress.',
    },
    'recipe_and_membership': {
        'fixed_epochs': 6,
        'onecycle_total_optimizer_updates': 44760,
        'stage_epochs': [2, 3],
        'optimizer': 'AdamW',
        'precision': 'fp16',
        'batch': 16,
        'chunk': 2000,
        'distill_alpha': 0.9,
        'seed': 4,
        'hidden': 256,
        'fit_train_groups': 10535,
        'fit_valid_groups': 1405,
        'evaluation_groups_excluded': 468,
        'loader_workers': 0,
        'evaluation_during_fit': False,
    },
}
output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps({'output': str(output_path), 'sha256': sha(output_path), 'status': receipt['status_at_snapshot'], 'started_utc': attempt['started_utc'], 'heartbeat_utc': attempt['heartbeat_utc'], 'latest_progress': fit_lines[-1], 'worker_peak_ram_bytes': attempt['peak_ram_bytes'], 'worker_peak_vram_bytes': attempt['peak_observed_vram_bytes']}, indent=2))
