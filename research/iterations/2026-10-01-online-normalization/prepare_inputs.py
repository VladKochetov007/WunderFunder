"""Fresh metadata bindings to the already audited control inputs; no model execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    inventory = ROOT / 'research/iterations/2026-10-01-continuation/student/inventory/receipt.json'
    expected = '8862f45c9f953dc04a577edc0579d7de7e4a6b2806e7785e627599bea421c20a'
    assert sha(inventory) == expected
    old = json.loads(inventory.read_text())
    records = {}
    for key in ('teacher_caches', 'teacher_checkpoint_metadata', 'row_maps', 'chain_folds',
                'propensity_models', 'propensity_caches', 'datasets'):
        rows = old[key] if isinstance(old[key], list) else [old[key]]
        for v in rows:
            p = Path(v['path']); s = p.stat()
            identity = dict(resolved=str(p.resolve()), size=s.st_size,
                            mtime_ns=s.st_mtime_ns, inode=s.st_ino)
            assert all(v[k] == x for k, x in identity.items()), (str(p), identity, v)
            record = dict(path=str(p), device=s.st_dev, **identity,
                          prior_inventory_section=key, prior_sha256=v.get('sha256', v.get('whole_file_sha256')))
            if s.st_size < 20 * 2**20:
                record['sha256'] = sha(p)
                if v.get('sha256'):
                    assert record['sha256'] == v['sha256']
            records[str(p)] = record
            if key == 'propensity_caches':
                side = p.with_suffix('.json'); t = side.stat()
                assert sha(side) == v['sidecar_sha256']
                records[str(side)] = dict(path=str(side), resolved=str(side.resolve()),
                    size=t.st_size, mtime_ns=t.st_mtime_ns, inode=t.st_ino, device=t.st_dev,
                    sha256=v['sidecar_sha256'])
    p = ROOT / 'research/artifacts/market-classes-v1.npz'; s = p.stat()
    records[str(p)] = dict(path=str(p), resolved=str(p.resolve()), size=s.st_size,
                          mtime_ns=s.st_mtime_ns, inode=s.st_ino, device=s.st_dev, sha256=sha(p))
    result = dict(status='PASS_PRIOR_IDENTITIES_AND_FRESH_SMALL_HASHES',
        created_utc=datetime.now(timezone.utc).isoformat(), inventory=str(inventory),
        inventory_sha256=expected, records=list(records.values()),
        limitation='Large arrays/parquet reused via resolved path + inode/device/size/mtime identity; not freshly whole-file hashed. Historical teacher producer epochs remain unbound. No cache regeneration.')
    with (HERE / 'input-bindings.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(dict(status=result['status'], records=len(records))))


if __name__ == '__main__':
    main()
