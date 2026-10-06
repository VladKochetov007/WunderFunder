"""Record authorized read-only identities; no array decoding or model execution."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata
import json
from common import ROOT, HERE, sha, stat_identity

def main():
    assert not (HERE / 'bindings.json').exists()
    test = json.loads((HERE / 'selftest-result.json').read_text())
    assert test['status'] == 'PASS' and test['full_generated_orchestration'] == 'PASS'
    for name, digest in test['sources'].items():
        if name == 'extract_selected.py':
            amendment = json.loads((HERE / 'resource-amendment.json').read_text())
            original = HERE / 'history/extract_selected.GPU3-only.py'
            assert sha(original) == digest == amendment['old_extract_sha256']
            replacement = amendment['only_source_replacement']
            assert original.read_text().replace(replacement['old'], replacement['new']) == (HERE / name).read_text()
            assert sha(HERE / name) == amendment['new_extract_sha256']
        else:
            assert sha(HERE / name) == digest
    inventory_path = ROOT / 'research/iterations/2026-10-01-continuation/student/inventory/receipt.json'
    inventory = json.loads(inventory_path.read_text())
    valid = next(r for r in inventory['datasets'] if r['path'].endswith('/valid.parquet'))
    path = ROOT / 'datasets/valid.parquet'
    stat = stat_identity(path)
    assert stat['st_size'] == valid['size'] and stat['st_ino'] == valid['inode'] and stat['st_mtime_ns'] == valid['mtime_ns']
    with path.open('rb') as stream:
        stream.seek(-65536, 2)
        import hashlib
        tail = hashlib.sha256(stream.read()).hexdigest()
    assert tail == valid['last65536bytes_sha256']
    inputs = {'datasets/valid.parquet': {'sha256': '2861ee30914065702626a3ac033d80d0563cdacaea50c667fc828c2eeb418782',
        'bytes': stat['st_size'], 'verification': 'stat_guard', 'stat': stat,
        'full_sha256_previously_verified': True, 'fresh_tail65536_sha256': tail,
        'limitation': 'Historical full hash plus current inventory inode/size/mtime and fresh tail; no fresh full-file hash.'}}
    for name, digest in {
        'cache/chain_folds.npz': '315b84f385f5368615c76087288b623f99c33bd680e9ebf7c3465bf5ab8afd35',
        'research/artifacts/market-classes-v1.npz': 'a805a6aab37f74120265a8abb6cad407b532e8ebae38290bdd8e7c80f6580fc2',
    }.items():
        path = ROOT / name
        assert sha(path) == digest
        inputs[name] = {'sha256': digest, 'bytes': path.stat().st_size}
    for fold in (0, 1):
        item = next(r for r in inventory['comparison_H2_inputs'] if r['name'] == f'on_ctl128s3_f{fold}')
        name = f'artifacts/runs/on_ctl128s3_f{fold}/model_ep1.pt'
        assert sha(ROOT / name) == item['checkpoint_sha256']
        inputs[name] = {'sha256': item['checkpoint_sha256'], 'bytes': (ROOT / name).stat().st_size}
    paths = [*HERE.glob('*.py'), HERE / 'spec.md', ROOT / 'scripts/train_rnn.py', *sorted((ROOT / 'src/wnn').glob('*.py'))]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    result = {'status': 'FROZEN_SELECTION_RISK_BINDINGS', 'utc': datetime.now(timezone.utc).isoformat(),
        'inputs': inputs, 'sources': sources, 'versions': {name: importlib.metadata.version(name) for name in
        ('numpy', 'scipy', 'scikit-learn', 'torch', 'pyarrow', 'threadpoolctl')},
        'generated_receipt_sha256': sha(HERE / 'selftest-result.json'),
        'student_inventory_sha256': sha(inventory_path),
        'historical_full_hash_receipt': 'research/iterations/2026-10-01-overnight/v1-runtime-audit/execution/worker-result.json',
        'historical_full_hash_receipt_sha256': sha(ROOT / 'research/iterations/2026-10-01-overnight/v1-runtime-audit/execution/worker-result.json'),
        'identity_only_reads': True, 'data_arrays_decoded': 0, 'model_objects_loaded': 0}
    (HERE / 'bindings.json').write_text(json.dumps(result, indent=2) + '\n')
    allocation = {'status': 'ROOT_ALLOCATED_SELECTION_RISK', 'utc': datetime.now(timezone.utc).isoformat(),
        'authority': 'Parent /root explicitly authorized one extraction600s + CPUanalysis300s after generated PASS, GPU7 after explicit student-completion notice and minGRU peer release, no retry/no WP/no forecast fit.',
        'bindings': str((HERE / 'bindings.json').relative_to(ROOT)), 'bindings_sha256': sha(HERE / 'bindings.json'),
        'extract_cpus': [34, 36], 'analysis_cpus': [34, 36], 'gpu_index': 7,
        'output': str((HERE / 'execution').relative_to(ROOT)),
        'limits': {'extract_seconds': 600, 'analysis_seconds': 300, 'kill_grace_seconds': 5, 'CPUQuota': '200%', 'MemoryMax': '4G', 'MemorySwapMax': 0, 'GPU_MiB': 5632, 'allocator_fraction': .30},
        'fresh_precheck': {'GPU7_used_MiB_at_most': 256, 'GPU7_compute_processes': 0},
        'retry': False, 'forecast_fits': 0, 'WP_queries': 0}
    (HERE / 'allocation.json').write_text(json.dumps(allocation, indent=2) + '\n')
    print(json.dumps({'bindings_sha256': sha(HERE / 'bindings.json'), 'allocation_sha256': sha(HERE / 'allocation.json')}))

if __name__ == '__main__': main()
