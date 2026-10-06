"""Independent byte-only supplement audit. No array decoding or extraction."""
from pathlib import Path, PurePosixPath
import argparse
import collections
import datetime
import hashlib
import json
import tarfile

MAIN_SHA = 'ea67f1acd0d2cad0bb25463d7cc4bd668575de032588c0cbb1b082dfb731a6e9'
MAIN_MANIFEST_SHA = 'c1a5004df1726338d87090d3554dadadb95f9f5a37c5a923f1dbf691a533773c'
PREFIX = 'WunderFunder-creative/research/iterations/2026-10-01-overnight/'


def digest_stream(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1 << 20), b''):
        h.update(chunk)
    return h.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return digest_stream(stream)


def safe(name):
    p = PurePosixPath(name)
    assert name and not p.is_absolute() and '..' not in p.parts
    assert str(p) == name and '\\' not in name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delivery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    d = args.delivery.resolve()
    assert args.output.resolve().parent == d and not args.output.exists()
    inv_path = d / 'remote-supplement.inventory.json'
    man_path = d / 'remote-supplement.manifest.json'
    rec_path = d / 'remote-supplement.receipt.json'
    arc_path = d / 'remote-supplement.tar.gz'
    source = d / 'build-remote-supplement.py'
    inv = json.loads(inv_path.read_text())
    man = json.loads(man_path.read_text())
    rec = json.loads(rec_path.read_text())
    main_manifest = json.loads((d / 'research-archive.manifest.json').read_text())
    assert sha(d / 'research-archive.manifest.json') == MAIN_MANIFEST_SHA
    assert sha(d / 'research-overnight.tar.gz') == MAIN_SHA
    assert inv['main_manifest_sha256'] == MAIN_MANIFEST_SHA
    assert inv['main_archive_sha256'] == man['main_archive_sha256'] == rec['main_archive_sha256'] == MAIN_SHA
    assert rec['status'] == 'PASS_REMOTE_SUPPLEMENT_BYTES_VERIFIED'
    assert rec['main_archive_unchanged'] is True
    assert rec['new_inference'] == rec['new_WP_queries'] == 0
    assert man['not_a_competition_submission'] is True and man['no_new_forecast_or_metric'] is True
    assert sha(inv_path) == rec['inventory_sha256'] == man['inventory_sha256']
    assert sha(man_path) == rec['manifest_sha256']
    assert sha(source) == rec['source_sha256']
    assert arc_path.stat().st_size == rec['archive_bytes'] and sha(arc_path) == rec['archive_sha256']
    assert inv['remote_root'] == man['remote_root']
    known = {r['path'][len(PREFIX):]: r for r in main_manifest['files'] if r['path'].startswith(PREFIX)}
    observed = {}
    counts = collections.Counter()
    for row in inv['files']:
        name = row['path']
        safe(name)
        assert name not in observed and row['bytes'] >= 0 and len(row['sha256']) == 64
        old = known.get(name)
        classification = ('remote_only' if old is None else
                          'shared_equal' if old['bytes'] == row['bytes'] and old['sha256'] == row['sha256'] else
                          'shared_divergent')
        assert classification == row['classification'], name
        observed[name] = row
        counts[classification] += 1
    assert len(observed) == rec['remote_regular_files'] == 2159
    assert {c: counts[c] for c in ['shared_equal', 'shared_divergent', 'remote_only']} == rec['classification_counts']
    main_only = sorted(set(known) - set(observed))
    assert main_only == inv['main_only_paths'] == man['main_only_paths'] == rec['main_only_paths']
    limit = inv['limit_bytes_per_added_file']
    assert limit == 64 << 20
    selected = [r for r in inv['files'] if r['classification'] != 'shared_equal' and r['bytes'] <= limit]
    excluded = [r for r in inv['files'] if r['classification'] != 'shared_equal' and r['bytes'] > limit]
    assert excluded == man['excluded_large_files']
    assert len(excluded) == rec['excluded_large_files'] == 56
    assert sum(r['bytes'] for r in excluded) == rec['excluded_large_bytes']
    assert all(r['bytes'] == 299680128 and r['path'].endswith('.npy') and
               r['path'].startswith(('cache/', 'ensemble/cache/')) for r in excluded)
    excluded_types = collections.Counter('training_partner_cache' if r['path'].startswith('ensemble/cache/')
                                         else 'heldfold_cache' for r in excluded)
    assert excluded_types == {'heldfold_cache': 48, 'training_partner_cache': 8}
    expected = {}
    for row in selected:
        name = 'beleriand-snapshot/' + row['path']
        expected[name] = {'path': name, 'remote_relative_path': row['path'], 'bytes': row['bytes'],
                          'sha256': row['sha256'], 'classification': row['classification']}
    assert len(expected) == len(man['files']) == rec['added_files'] == 430
    assert len({r['path'] for r in man['files']}) == len(man['files'])
    assert {r['path']: r for r in man['files']} == expected
    assert sum(r['bytes'] for r in selected) == rec['added_file_bytes']
    links = inv['symlinks_not_followed']
    assert links == man['symlinks_not_followed'] and len(links) == rec['symlinks_not_followed'] == 1
    link_coverage = []
    for link in links:
        safe(link['path'])
        assert link['path'] not in observed
        target = PurePosixPath(link['target']).relative_to(PurePosixPath(inv['remote_root'])).as_posix()
        assert target in observed
        if target in known and observed[target]['classification'] == 'shared_equal':
            preserved_in = 'main_archive'
        else:
            assert 'beleriand-snapshot/' + target in expected
            preserved_in = 'remote_supplement'
        link_coverage.append({'path': link['path'], 'target_relative_path': target,
                              'target_preserved_in': preserved_in,
                              'target_sha256': observed[target]['sha256']})
    metadata = {'supplement-manifest.json': man_path.read_bytes(),
                'remote-supplement.inventory.json': inv_path.read_bytes(),
                'build-remote-supplement.py': source.read_bytes()}
    assert rec['embedded_metadata_files'] == len(metadata) == 3
    seen = set()
    with tarfile.open(arc_path, 'r|gz') as archive:
        for member in archive:
            safe(member.name)
            assert member.isfile() and member.name not in seen
            seen.add(member.name)
            stream = archive.extractfile(member)
            assert stream is not None
            if member.name in metadata:
                assert member.size == len(metadata[member.name]) and stream.read() == metadata[member.name]
            else:
                row = expected[member.name]
                assert member.size == row['bytes'] and digest_stream(stream) == row['sha256']
    assert seen == set(expected) | set(metadata)
    result = {'status': 'PASS_INDEPENDENT_REMOTE_SUPPLEMENT_BYTES',
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'checker_sha256': sha(__file__), 'main_archive_sha256': MAIN_SHA,
              'main_manifest_sha256': MAIN_MANIFEST_SHA,
              'supplement_archive_sha256': sha(arc_path), 'supplement_archive_bytes': arc_path.stat().st_size,
              'inventory_sha256': sha(inv_path), 'manifest_sha256': sha(man_path),
              'builder_sha256': sha(source), 'builder_receipt_sha256': sha(rec_path),
              'remote_regular_files': len(observed),
              'classification_counts': {c: counts[c] for c in ['shared_equal', 'shared_divergent', 'remote_only']},
              'added_regular_files': len(expected), 'added_file_bytes': sum(r['bytes'] for r in selected),
              'embedded_metadata_files': len(metadata), 'main_only_paths': main_only,
              'excluded_large_files': len(excluded), 'excluded_large_bytes': sum(r['bytes'] for r in excluded),
              'excluded_cache_types_by_path': dict(excluded_types), 'symlinks_recorded_not_followed': links,
              'symlink_target_preservation': link_coverage,
              'all_member_bytes_and_exact_inventory_verified': True, 'no_main_snapshot_overwrite': True,
              'internal_external_metadata_identical': True, 'main_archive_unchanged': True,
              'new_forecast_fits': 0, 'new_inference': 0, 'new_WP_queries': 0, 'array_or_model_decoding': False,
              'limits': ['Remote source hashes come from the bound builder inventory; this audit independently verifies classification and all transferred archive bytes.',
                         'The56 large prediction caches remain remote; their bytes were hashed by the builder and their arrays were not decoded or copied.',
                         'Symlink is metadata only; its target is preserved in the separately identified main or supplemental archive.',
                         'Additional logs, scripts and fixtures are preservation evidence, not additional experiments.']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
