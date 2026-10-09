from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

root = Path('/home/vlad/development/wunder/wnn_connectome_starterpack').resolve()
packet_path = Path('/home/vlad/development/wunder/worktrees/h272-parent-v2-profile-launch-20261008/research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529/recipes/H272_PARENT_MULTIPLICATIVE_READOUT_V2_RUNTIME/H272_PARENT_V2_CLEANFOLD_PROFILE_LAUNCH_PACKET_20261008_R1.json')
p = json.loads(packet_path.read_text())
base = root / 'research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
assert p['schema'] == 'h272-parent-readout-v2-profile-launch-packet-v1'
assert p['status'] == 'prepared_not_launched'
assert p['profile_contract']['profile_only'] is True
assert p['profile_contract']['fit_authorized'] is False
assert p['profile_contract']['export_authorized'] is False
assert p['profile_contract']['grading_authorized'] is False
assert p['controller']['attempt_record_created'] is False
assert p['controller']['quota_snapshot_captured'] is False
assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', p['attempt_id'])
assert p['controller']['source_manifest_file_count'] == len(p['controller']['source_manifest'])
manifest = p['controller']['source_manifest']
expected_digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
assert expected_digest == p['controller']['source_manifest_sha256']
auth_path = root / p['launch_authority']['authorization_path']
review_path = root / p['launch_authority']['source_review_path']
validation_path = root / p['launch_authority']['source_validation_path']
lock_path = root / p['locked_sources']['source_lock_path']
recipe_path = root / p['recipe']['path']
output_path = root / p['recipe']['profile_output_path']
auth = json.loads(auth_path.read_text())
review = json.loads(review_path.read_text())
lock = json.loads(lock_path.read_text())
assert sha(auth_path) == p['launch_authority']['authorization_sha256']
assert sha(review_path) == p['launch_authority']['source_review_sha256'] == auth['review_sha256']
assert review['status'] == p['launch_authority']['source_review_status']
assert sha(validation_path) == p['launch_authority']['source_validation_sha256']
assert sha(lock_path) == p['locked_sources']['source_lock_sha256'] == auth['source_lock_sha256']
assert len(lock['source_sha256']) == p['locked_sources']['package_source_pin_count'] == 65
assert lock['source_sha256'] == auth['source_lock_source_sha256'] == p['locked_sources']['package_source_sha256']
assert sha(recipe_path) == p['recipe']['sha256'] == auth['recipe_sha256']
assert not output_path.exists()
assert not (root / p['controller']['state_path'].replace(str(root) + '/', '') / 'attempts' / (p['attempt_id'] + '.json')).exists()
sys.path.insert(0, str(root))
from scripts import bandit_resources
actual = bandit_resources.source_manifest(root, p['worker']['exact_argv'], auth_path)
assert actual == manifest, 'prepared source manifest differs from current root'
print(json.dumps({
    'status': 'PASS',
    'packet_sha256': sha(packet_path),
    'manifest_sha256': expected_digest,
    'manifest_files': len(manifest),
    'package_source_pins_verified': len(lock['source_sha256']),
    'root_authorization_sha256': sha(auth_path),
    'source_review_sha256': sha(review_path),
    'source_validation_sha256': sha(validation_path),
    'attempt_id_unused': True,
    'output_absent': True,
    'launch_or_fit_started': False,
}, sort_keys=True))
