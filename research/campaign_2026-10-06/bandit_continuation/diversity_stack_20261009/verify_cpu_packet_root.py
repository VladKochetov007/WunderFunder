import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

packet_root = Path('/home/vlad/development/wunder/worktrees/luna-runner-20261008/research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/cloud_delivery/cpu_source_stage_candidate_r1_final')
packet_path = packet_root / 'cpu_source_stage.packet.json'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(packet_path) == '0e04b36e04134b085f8f272e172a636d28c376064943c04c23ac27ba9554abbf'
p = json.loads(packet_path.read_text())
for field in ['metadata', 'script']:
    item = p['kernel'][field]
    assert digest(packet_root / item['path']) == item['sha256']
metadata = json.loads((packet_root / p['kernel']['metadata']['path']).read_text())
assert metadata['is_private'] and not metadata['enable_gpu'] and not metadata['enable_tpu']
assert metadata['id'].split('/')[1] == metadata['title'].lower().replace(' ', '-')
assert len((packet_root / p['kernel']['script']['path']).read_bytes()) < 1000000
compile((packet_root / p['kernel']['script']['path']).read_bytes(), '<sealed_kernel>', 'exec')
for entry in p['source_asset']['upload_file_inventory']:
    path = packet_root / 'source_asset' / entry['filename']
    assert path.stat().st_size == entry['size'] and digest(path) == entry['sha256']
asset = p['source_asset']
archive = packet_root / 'source_asset' / (asset['archive_sha256'] + '.wnn')
entries = {entry['path']: entry for entry in asset['archive_entry_sha256s']}
with tempfile.TemporaryDirectory(prefix='wnn-root-cpu-packet-') as tmp:
    folder = Path(tmp)
    with tarfile.open(archive) as tar:
        assert set(tar.getnames()) == set(entries)
        for member in tar:
            assert member.isfile() and '..' not in Path(member.name).parts and not Path(member.name).is_absolute()
            raw = tar.extractfile(member).read()
            assert len(raw) == entries[member.name]['size']
            assert hashlib.sha256(raw).hexdigest() == entries[member.name]['sha256']
            target = folder / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    env = dict(os.environ, PYTHONPATH=str(folder / 'src'))
    for name in ['parsed_sequence_cache.py', 'profile_parsed_sequence_cache.py']:
        run = subprocess.run([sys.executable, str(folder / 'scripts' / name), '--help'], env=env, capture_output=True, text=True, timeout=30)
        assert run.returncode == 0, run.stderr
        assert 'usage:' in run.stdout
print(json.dumps({'status': 'pass', 'packet_sha256': digest(packet_path), 'private_cpu_only': True, 'all_payload_hashes': True, 'exact_extracted_entrypoint_cli_imports': 2, 'source_archive_files': len(entries)}))
