"""Independent opaque-byte audit of the completed research backup.

No extraction, model import, tensor decoding, forecasting or metric calculation.
The output must be outside the archived overnight tree.
"""
from pathlib import Path, PurePosixPath
import argparse
import datetime
import hashlib
import json
import subprocess
import tarfile

P = Path(__file__).resolve().parent.parent
ROOT = P.parents[2]
ZIP_HASHES = {
    'candV1_ft2.zip': ('ea494921497d9f3cf53b19b1b68285133983560a249b38ef40738b3d846d319d', 7382512),
    'baseline3_fixed.zip': ('9471821e81e7c3f80a088147512183b72e0ab52bc593a1136d2becc229d0d098', 3796019),
    'experimental/quant3_fixed.zip': ('48cf940bd1c7e812fe8005a90e3f11fb3199a8940c55e0adbfef00ad1de1f10f', 6501798),
}


def stream_sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''):
        h.update(block)
    return h.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return stream_sha(stream)


def safe_name(name):
    parts = PurePosixPath(name).parts
    assert name and not name.startswith('/') and '..' not in parts
    assert str(PurePosixPath(name)) == name and '\\' not in name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    delivery = args.delivery.resolve()
    output = args.output.resolve()
    assert not output.exists() and not output.is_relative_to(P)
    archive_path = delivery / 'research-overnight.tar.gz'
    manifest_path = delivery / 'research-archive.manifest.json'
    receipt_path = delivery / 'research-archive.receipt.json'
    receipt = json.loads(receipt_path.read_text())
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    assert receipt['status'] == 'PASS_ARCHIVE_BYTES_VERIFIED'
    assert receipt['competition_model_execution'] is False and receipt['WP_queries'] == 0
    assert receipt['standalone_bundle_empty_repository_verify_and_fetch'] is True
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert head == args.expected_commit == receipt['source_commit'] == manifest['source_commit']
    assert receipt['archive_file'] == archive_path.name
    assert archive_path.stat().st_size == receipt['archive_bytes']
    archive_sha = sha(archive_path)
    assert archive_sha == receipt['archive_sha256']
    assert hashlib.sha256(manifest_bytes).hexdigest() == receipt['manifest_sha256']
    records = manifest['files']
    assert len(records) == receipt['file_count_excluding_manifest']
    assert sum(r['bytes'] for r in records) == receipt['uncompressed_file_bytes']
    expected = {}
    for row in records:
        name = row['path']
        safe_name(name)
        assert name not in expected and row['bytes'] >= 0 and len(row['sha256']) == 64
        expected[name] = row
    seen = set()
    with tarfile.open(archive_path, 'r|gz') as archive:
        for member in archive:
            safe_name(member.name)
            assert member.isfile() and member.name not in seen
            seen.add(member.name)
            stream = archive.extractfile(member)
            assert stream is not None
            if member.name == 'archive-manifest.json':
                assert member.size == len(manifest_bytes) and stream.read() == manifest_bytes
            else:
                row = expected[member.name]
                assert member.size == row['bytes'] and stream_sha(stream) == row['sha256']
    assert seen == set(expected) | {'archive-manifest.json'}

    # Recheck the present sources against the packed snapshot using bytes only.
    for name, row in expected.items():
        if name.startswith('WunderFunder-creative/'):
            source = ROOT / name.removeprefix('WunderFunder-creative/')
        elif name.startswith('delivery-2026-10-01/'):
            source = delivery / name.removeprefix('delivery-2026-10-01/')
        else:
            assert name in ['RESTORE.md', 'research-source.bundle']
            source = delivery / name
        assert source.is_file() and not source.is_symlink(), source
        assert source.stat().st_size == row['bytes'] and sha(source) == row['sha256'], source

    # Independently require exact coverage of the closed overnight directory.
    prefix = 'WunderFunder-creative/'
    required_tree = set()
    for source in P.rglob('*'):
        if '__pycache__' in source.parts or source.name == '.DS_Store' or source.suffix == '.pyc':
            continue
        assert not source.is_symlink(), source
        if source.is_file():
            required_tree.add(prefix + source.relative_to(ROOT).as_posix())
    p_prefix = prefix + P.relative_to(ROOT).as_posix() + '/'
    actual_tree = {name for name in expected if name.startswith(p_prefix)}
    assert actual_tree == required_tree

    fits = json.loads((P / 'completed-fit-archive-52.json').read_text())
    assert fits['run_count'] == receipt['forecast_runs'] == manifest['forecast_runs'] == 52
    assert len(fits['files']) == receipt['run_files'] == manifest['run_files'] == 224
    fit_names = set()
    run_directories = set()
    for row in fits['files']:
        name = prefix + row['path']
        assert name not in fit_names
        fit_names.add(name)
        run_directories.add(PurePosixPath(row['path']).parts[2])
        assert expected[name] == {'path': name, 'bytes': row['bytes'], 'sha256': row['sha256']}
    assert len(run_directories) == 52 and run_directories == set(fits['runs'])
    assert {n for n in expected if n.startswith(prefix + 'artifacts/runs/')} == fit_names
    for relative, (digest, size) in ZIP_HASHES.items():
        row = expected['delivery-2026-10-01/' + relative]
        assert row['sha256'] == digest and row['bytes'] == size
    required_delivery = {'research-source.bundle', 'RESTORE.md',
                         'delivery-2026-10-01/README.md', 'delivery-2026-10-01/manifest.json'}
    required_delivery |= {'delivery-2026-10-01/' + name for name in ZIP_HASHES}
    assert set(expected) == required_tree | fit_names | required_delivery

    result = {
        'status': 'PASS_INDEPENDENT_RESEARCH_ARCHIVE_BYTES',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'checker_sha256': sha(__file__), 'source_commit': head,
        'archive_sha256': archive_sha, 'archive_bytes': archive_path.stat().st_size,
        'manifest_sha256': sha(manifest_path), 'packager_receipt_sha256': sha(receipt_path),
        'archived_regular_files_excluding_manifest': len(records),
        'overnight_tree_files': len(required_tree), 'forecast_runs': 52, 'run_files': 224,
        'archived_run_bytes': sum(r['bytes'] for r in fits['files']),
        'three_zip_identities_match': True, 'internal_external_manifest_identical': True,
        'all_archive_members_match_current_source_bytes': True,
        'archive_path_safety_and_exact_inventory': True,
        'model_or_data_decoding': False, 'new_inference': 0, 'new_WP_queries': 0,
        'limits': ['Local byte audit only. Remote copied-archive identity needs a separate hash receipt.',
                   'Packager tested standalone Git restoration; this audit binds its receipt and bundle bytes without repeating restore.',
                   'Organizer dataset bodies, large remote prediction caches and automatic runtime-environment restoration are outside this backup.'],
    }
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
