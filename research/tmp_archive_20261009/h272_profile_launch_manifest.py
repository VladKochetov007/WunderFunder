from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

root = Path('/home/vlad/development/wunder/wnn_connectome_starterpack').resolve()
base = root / 'research/campaign_2026-10-06/bandit_continuation/continuous_execution_E529'
package = base / 'recipes/H272_PARENT_MULTIPLICATIVE_READOUT_V2_RUNTIME'
authorization = base / 'H272_PARENT_V2_ROOT_CONDITIONAL_PROFILE_AUTHORIZATION_20261008_R1.json'
recipe = package / 'recipes/H272_PARENT_READOUT_V2_CLEANFOLD_20261008.json'
script = package / 'scripts/train_parent_readout.py'
output = base / 'profiles/H272_PARENT_V2_CLEANFOLD_PROFILE.json'
command = [
    '/usr/bin/timeout', '--signal=TERM', '--kill-after=10s', '900s',
    '/home/vlad/development/wunder/wnn_connectome_starterpack/.venv/bin/python',
    str(script.relative_to(root)), 'profile', '--recipe', str(recipe.relative_to(root)),
    '--data-root', '.', '--authorization', str(authorization.relative_to(root)),
    '--device', 'cuda', '--warmup-chunks', '2', '--steps', '8', '--cpu-threads', '1',
    '--output', str(output.relative_to(root)),
]
sys.path.insert(0, str(root))
from scripts import bandit_resources
manifest = bandit_resources.source_manifest(root, command, authorization)
canonical = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
lock_path = package / 'H272_PARENT_READOUT_V2_SOURCE_LOCK_V1_20261008.json'
lock_bytes = lock_path.read_bytes()
lock = json.loads(lock_bytes)
auth = json.loads(authorization.read_text())
review_path = base / 'reviews/H272_PARENT_V2_ROOT_SOURCE_REVIEW_20261008_R1.json'
result = {
    'repository_head': __import__('subprocess').check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
    'source_manifest': manifest,
    'source_manifest_sha256': hashlib.sha256(canonical).hexdigest(),
    'source_manifest_file_count': len(manifest),
    'authorization_sha256': hashlib.sha256(authorization.read_bytes()).hexdigest(),
    'authorization_review_sha256': auth['review_sha256'],
    'review_file_sha256': hashlib.sha256(review_path.read_bytes()).hexdigest(),
    'source_lock_sha256': hashlib.sha256(lock_bytes).hexdigest(),
    'source_lock_package_file_count': len(lock['source_sha256']),
    'source_lock_sha256_map': lock['source_sha256'],
    'recipe_sha256': hashlib.sha256(recipe.read_bytes()).hexdigest(),
    'exact_worker_argv': command,
}
Path('/tmp/h272_profile_launch_manifest.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in {'source_manifest', 'source_lock_sha256_map', 'exact_worker_argv'}}, sort_keys=True))
