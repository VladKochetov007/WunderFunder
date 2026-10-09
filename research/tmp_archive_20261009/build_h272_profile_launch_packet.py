from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

root = Path('/home/vlad/development/wunder/wnn_connectome_starterpack').resolve()
worktree = Path('/home/vlad/development/wunder/worktrees/h272-parent-v2-profile-launch-20261008').resolve()
base = root / 'research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529'
package = base / 'recipes/H272_PARENT_MULTIPLICATIVE_READOUT_V2_RUNTIME'
auth_path = base / 'H272_PARENT_V2_ROOT_CONDITIONAL_PROFILE_AUTHORIZATION_20261008_R1.json'
review_path = base / 'reviews/H272_PARENT_V2_ROOT_SOURCE_REVIEW_20261008_R1.json'
validation_path = base / 'recipes/H272_PARENT_READOUT_V2_SOURCE_VALIDATION_20261008_R1.json'
source_lock_path = package / 'H272_PARENT_READOUT_V2_SOURCE_LOCK_V1_20261008.json'
recipe_path = package / 'recipes/H272_PARENT_READOUT_V2_CLEANFOLD_20261008.json'
worker_path = package / 'scripts/train_parent_readout.py'
output_path = base / 'profiles/H272_PARENT_V2_CLEANFOLD_PROFILE.json'
state_path = root / 'research/campaign_2026-10-06/bandit_continuation/runtime'
attempt_id = 'h272-parent-v2-cleanfold-profile-r1-20261008'
packet_path = worktree / 'research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/recipes/H272_PARENT_MULTIPLICATIVE_READOUT_V2_RUNTIME/H272_PARENT_V2_CLEANFOLD_PROFILE_LAUNCH_PACKET_20261008_R1.json'

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
auth = json.loads(auth_path.read_text())
review = json.loads(review_path.read_text())
validation = json.loads(validation_path.read_text())
lock_bytes = source_lock_path.read_bytes()
lock = json.loads(lock_bytes)
recipe = json.loads(recipe_path.read_text())

assert auth.get('status') == 'approved'
assert auth.get('command') == 'profile'
assert auth.get('fit_authorized') is False
assert auth.get('recipe_sha256') == sha(recipe_path)
assert auth.get('recipe_sha256') == auth.get('source_lock_source_sha256', {}).get('recipes/H272_PARENT_READOUT_V2_CLEANFOLD_20261008.json')
assert auth.get('source_lock_sha256') == sha(source_lock_path)
assert auth.get('source_lock_source_sha256') == lock.get('source_sha256')
assert review.get('status') == 'source_review_pass_profile_only_conditional_next_free_gpu'
assert auth.get('review_sha256') == sha(review_path)
assert len(lock['source_sha256']) == 65
assert recipe['recipe_id'] == auth['recipe_id']
assert recipe['mode'] == 'cleanfold'
assert recipe['paths']['profile_output'] == str(output_path.relative_to(root))
assert not output_path.exists(), f'profile output already exists: {output_path}'
assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', attempt_id)
assert not (state_path / 'attempts' / f'{attempt_id}.json').exists()

for relative, expected in lock['source_sha256'].items():
    actual = sha(package / relative)
    assert actual == expected, f'package source mismatch: {relative}: {actual} != {expected}'
for relative, expected in lock['native_source_sha256'].items():
    actual = sha(root / relative)
    assert actual == expected, f'native source mismatch: {relative}: {actual} != {expected}'
for relative, expected in lock['baseline_h272_source_sha256'].items():
    actual = sha(root / relative)
    assert actual == expected, f'baseline source mismatch: {relative}: {actual} != {expected}'
for mode, relative in [('cleanfold', 'recipes/H272_PARENT_READOUT_V2_CLEANFOLD_20261008.json'), ('full_data', 'recipes/H272_PARENT_READOUT_V2_FULL_DATA_20261008.json')]:
    expected = lock['recipe_sha256'][mode]
    actual = sha(package / relative)
    assert actual == expected, f'recipe source mismatch: {relative}: {actual} != {expected}'

command = [
    '/usr/bin/timeout', '--signal=TERM', '--kill-after=10s', '900s',
    str(root / '.venv/bin/python'),
    str(worker_path.relative_to(root)), 'profile',
    '--recipe', str(recipe_path.relative_to(root)),
    '--data-root', '.',
    '--authorization', str(auth_path.relative_to(root)),
    '--device', 'cuda', '--warmup-chunks', '2', '--steps', '8', '--cpu-threads', '1',
    '--output', str(output_path.relative_to(root)),
]
sys_path = str(root)
# Importing this stdlib-only controller is read-only; its function is also the exact
# manifest generator used by the future launch.
import sys
sys.path.insert(0, sys_path)
from scripts import bandit_resources
manifest = bandit_resources.source_manifest(root, command, auth_path)
manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
manifest_receipt_path = Path('/tmp/h272_profile_launch_manifest.json')
manifest_receipt = json.loads(manifest_receipt_path.read_text())
assert manifest == manifest_receipt['source_manifest']
assert manifest_sha == manifest_receipt['source_manifest_sha256']

controller_command = [
    str(root / 'scripts/limited'),
    str(root / '.venv/bin/python'),
    'scripts/bandit_resources.py', '--root', str(root), '--state', str(state_path),
    'launch', '--attempt', attempt_id, '--kind', 'gpu', '--cpus', '1',
    '--reserve-bytes', str(auth['resource_limits']['ram_bytes']),
    '--profile', 'H272 V2 cleanfold profile: 2 warmup + 8 timed chunks, one GPU, one CPU',
    '--contract', str(auth_path), '--python', str(root / '.venv/bin/python'),
    '--admission-timeout', '900', '--', *command,
]

packet = {
    'schema': 'h272-parent-readout-v2-profile-launch-packet-v1',
    'status': 'prepared_not_launched',
    'created_utc': datetime.now(UTC).isoformat(),
    'owner': 'root_controller',
    'attempt_id': attempt_id,
    'launch_authority': {
        'authorization_path': str(auth_path.relative_to(root)),
        'authorization_sha256': sha(auth_path),
        'authorization_review_sha256': auth['review_sha256'],
        'source_review_path': str(review_path.relative_to(root)),
        'source_review_sha256': sha(review_path),
        'source_review_status': review['status'],
        'source_validation_path': str(validation_path.relative_to(root)),
        'source_validation_sha256': sha(validation_path),
        'source_validation_status': validation.get('status'),
        'fit_authorized': False,
    },
    'recipe': {
        'path': str(recipe_path.relative_to(root)),
        'recipe_id': recipe['recipe_id'],
        'mode': recipe['mode'],
        'sha256': sha(recipe_path),
        'profile_output_path': str(output_path.relative_to(root)),
        'output_absent_at_packet_creation': True,
    },
    'locked_sources': {
        'source_lock_path': str(source_lock_path.relative_to(root)),
        'source_lock_sha256': sha(source_lock_path),
        'package_source_pin_count': len(lock['source_sha256']),
        'package_source_sha256': lock['source_sha256'],
        'native_source_pin_count': len(lock['native_source_sha256']),
        'baseline_h272_source_pin_count': len(lock['baseline_h272_source_sha256']),
        'unchanged_from_root_authorization': True,
    },
    'profile_contract': {
        'profile_only': True,
        'warmup_chunks': 2,
        'timed_chunks': 8,
        'chunk_rows': 2000,
        'batch_size': 8,
        'cpu_threads': 1,
        'device': 'cuda',
        'cpu_reservation_cores': 1,
        'ram_reservation_bytes': auth['resource_limits']['ram_bytes'],
        'wall_seconds': auth['resource_limits']['wall_seconds'],
        'observed_vram_stop_bytes': auth['resource_limits']['observed_vram_stop_bytes'],
        'allocator_memory_bytes': 4300000000,
        'fit_authorized': False,
        'export_authorized': False,
        'grading_authorized': False,
    },
    'admission_conditions': auth['admission_conditions'],
    'required_fresh_checks_before_launch': [
        'Confirm the healthy seed99 fit and its priority export are terminally complete.',
        'Confirm no GPU worker is running and the local GPU lock is free.',
        'Run bandit_resources durable admission from the main repository worktree; require the current 5-CPU/18-GB-high/20-GB-max/no-swap parent and fit the 1-CPU/6.5-GB reservation under fresh aggregate CPU/RAM headroom.',
        'Require fresh GPU observation below the 5,000,000,000-byte observed stop and allow the controller to take the exclusive GPU lock.',
        'Require admission open, attempt ID unused, frozen source lock and source-manifest hashes unchanged, and the exact profile output path absent.',
        'Recompute the bandit_resources source manifest immediately before launch; it must equal the recorded manifest or this packet must be revised before launch.',
        'Launch only the profile command below; do not chain fit, export, grading, or prediction work.',
    ],
    'controller': {
        'repository_root': str(root),
        'repository_head_at_packet_creation': manifest_receipt['repository_head'],
        'state_path': str(state_path),
        'invocation_environment': {
            'WNN_SLICE': 'wnn-campaign.slice',
            'WNN_CPU_QUOTA': '100%',
            'WNN_THREADS': '1',
            'WNN_MEM_MAX': '1500000000',
            'WNN_RUNTIME_MAX': '120s',
        },
        'exact_argv': controller_command,
        'source_manifest_sha256': manifest_sha,
        'source_manifest_file_count': len(manifest),
        'source_manifest': manifest,
        'attempt_record_created': False,
        'quota_snapshot_captured': False,
    },
    'worker': {
        'exact_argv': command,
        'environment_set_by_bandit_resources': {
            'WNN_GPU_MEMORY_BYTES': '4300000000',
            'WNN_VRAM_GB': '4.3',
            'WNN_THREADS': '1',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'NUMBA_NUM_THREADS': '1',
            'POLARS_MAX_THREADS': '1',
            'ARROW_NUM_THREADS': '1',
            'CUDA_VISIBLE_DEVICES': '0',
            'PYTHONUNBUFFERED': '1',
            'PYTHONHASHSEED': '0',
        },
    },
    'validation': {
        'authorization_recipe_source_lock_and_review_pins_verified': True,
        'all_65_package_source_files_verified': True,
        'all_native_and_baseline_source_pins_verified': True,
        'source_manifest_recomputed_with_bandit_resources_source_manifest': True,
        'source_manifest_file_count': len(manifest),
        'source_manifest_sha256': manifest_sha,
        'profile_output_absent': True,
        'no_attempt_record_written': True,
        'no_profile_or_fit_or_export_or_grade_started': True,
    },
}
packet_path.parent.mkdir(parents=True, exist_ok=True)
packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + '\n')
print(json.dumps({
    'packet_path': str(packet_path),
    'packet_bytes': packet_path.stat().st_size,
    'authorization_sha256': packet['launch_authority']['authorization_sha256'],
    'source_lock_sha256': packet['locked_sources']['source_lock_sha256'],
    'source_manifest_sha256': manifest_sha,
    'source_manifest_file_count': len(manifest),
    'source_pins_verified': 65,
    'attempt_record_created': False,
}, sort_keys=True))
