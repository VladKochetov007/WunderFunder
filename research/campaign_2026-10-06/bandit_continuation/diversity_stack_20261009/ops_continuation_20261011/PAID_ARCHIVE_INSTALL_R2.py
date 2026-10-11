#!/usr/bin/env python3
"""Validate and no-overwrite install a frozen manifest-bound Vast artifact archive."""
from __future__ import annotations
import hashlib
import json
import os
import stat
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

MAIN = Path('/home/vlad/development/wunder/wnn_connectome_starterpack').resolve()
ARCHIVE = MAIN / 'research/campaign_2026-10-06/bandit_continuation/storage_handoff_20261011/paid_archive_resume_r2/paid_artifact_archive_b4e49625_r1.tar'
COLLECTION = MAIN / 'research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1/collected_b/collection_manifest.json'
EXPECTED_ARCHIVE_BYTES = 735_406_080
EXPECTED_ARCHIVE_SHA = 'd7cdea95581c938cc7382c3561f6c14e842e970e0cfafc50b9cbb7b31586df4c'
EXPECTED_SELECTION_SHA = 'bf24193b6a0dfcc61f9d54292576b976a50abe15901be680bba7f9739e720c7c'
EXPECTED_COLLECTION_SHA = 'b4e496252c4b28884f37854e4705b2ff4751bafd9cf965317c543a2957cf94c6'
EXPECTED_INVENTORY_SHA = 'ca709be18ce7d37b5a4a5370b06fd3d73a92853ad99531aaba17c8cf867dfeb8'
EXPECTED_PAYLOAD_BYTES = 732_881_211
EXPECTED_ATTEMPT = 'vast-teachers-r1-lane-b-20261010'
EXPECTED_INSTANCE = 55080658
MANIFEST_NAME = 'PAID_ARTIFACT_ARCHIVE_MANIFEST.json'
VARIANT_ROOT = MAIN / 'artifacts/source-qualified/vast-lane-b-55080658-paid-archive-b4e49625-r1'
RECEIPT = Path(__file__).parent / 'monitor_20261011_1730_r2/PAID_ARTIFACT_ARCHIVE_MEMBER_INSTALL_R2.json'
CHUNK = 4 * 1024 * 1024


def hash_stream(src, dst=None, fadvise_fd=None, fadvise_offset=0, fadvise_len=0):
    h = hashlib.sha256()
    size = 0
    while True:
        block = src.read(CHUNK)
        if not block:
            break
        h.update(block)
        size += len(block)
        if dst is not None:
            dst.write(block)
    if fadvise_fd is not None and hasattr(os, 'posix_fadvise'):
        try:
            os.posix_fadvise(fadvise_fd, fadvise_offset, fadvise_len or size, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
    return size, h.hexdigest()


def hash_path(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    size = 0
    with path.open('rb', buffering=0) as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
            size += len(block)
        if hasattr(os, 'posix_fadvise'):
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    return size, h.hexdigest()


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def safe_main_relative(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        raise RuntimeError(f'non-absolute local destination: {path_text}')
    # Reject lexical traversal and paths outside the project even if a symlink might resolve inward.
    try:
        rel = path.relative_to(MAIN)
    except ValueError as exc:
        raise RuntimeError(f'local destination outside MAIN: {path_text}') from exc
    if any(part in ('', '.', '..') for part in rel.parts):
        raise RuntimeError(f'unsafe local destination: {path_text}')
    if path == MAIN:
        raise RuntimeError('destination resolves to project root')
    return rel


def resolved_parent_is_safe(path: Path) -> bool:
    current = path.parent
    while current != MAIN and current != current.parent:
        if current.is_symlink():
            return False
        current = current.parent
    try:
        path.parent.resolve(strict=False).relative_to(MAIN)
        return True
    except ValueError:
        return False


def select_target(canonical: Path, rel: Path, wanted_size: int, wanted_sha: str) -> tuple[Path, str]:
    # Existing canonical content is only accepted after byte count and SHA verification.
    if not resolved_parent_is_safe(canonical):
        canonical_state = 'canonical_parent_symlink_or_escape'
    elif canonical.is_symlink():
        canonical_state = 'canonical_symlink_preserved'
    elif canonical.exists():
        if not canonical.is_file():
            canonical_state = 'canonical_nonfile_preserved'
        else:
            n = canonical.stat().st_size
            if n == wanted_size:
                found_n, found_sha = hash_path(canonical)
                if found_n == wanted_size and found_sha == wanted_sha:
                    return canonical, 'canonical_preexisting_identical'
                canonical_state = 'canonical_different_sha_preserved'
            else:
                canonical_state = 'canonical_different_size_preserved'
    else:
        canonical_state = 'canonical_missing'
    if canonical_state == 'canonical_missing' and not resolved_parent_is_safe(canonical):
        canonical_state = 'canonical_parent_symlink_or_escape'
    if canonical_state == 'canonical_missing':
        return canonical, 'canonical_new'
    variant = VARIANT_ROOT / rel
    if variant.is_symlink():
        raise RuntimeError(f'source-qualified destination is a symlink: {variant}')
    if variant.exists():
        if not variant.is_file():
            raise RuntimeError(f'source-qualified destination is not regular: {variant}')
        n = variant.stat().st_size
        if n == wanted_size:
            found_n, found_sha = hash_path(variant)
            if found_n == wanted_size and found_sha == wanted_sha:
                return variant, 'source_variant_preexisting_identical'
        raise RuntimeError(f'conflicting source-qualified destination exists: {variant}')
    if not resolved_parent_is_safe(variant):
        raise RuntimeError(f'source-qualified path escapes MAIN: {variant}')
    return variant, canonical_state + '_source_variant_new'


def write_no_replace(member: tarfile.TarInfo, src, target: Path, expected_size: int, expected_sha: str, archive_fd: int) -> tuple[str, int, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f'.{target.name}.rescue-', suffix='.part', dir=target.parent)
    temp = Path(temp_name)
    h = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, 'wb', buffering=0) as out:
            while True:
                block = src.read(CHUNK)
                if not block:
                    break
                out.write(block)
                h.update(block)
                size += len(block)
            out.flush()
            os.fsync(out.fileno())
            if hasattr(os, 'posix_fadvise'):
                try:
                    os.posix_fadvise(out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
        got_sha = h.hexdigest()
        if size != expected_size or got_sha != expected_sha:
            raise RuntimeError(f'archive member mismatch: {member.name} size={size} sha={got_sha}')
        if hasattr(os, 'posix_fadvise'):
            try:
                os.posix_fadvise(archive_fd, member.offset_data, member.size, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        try:
            os.link(temp, target)
            fsync_dir(target.parent)
            status = 'installed_no_replace'
        except FileExistsError:
            # A concurrent writer won; only accept an independently identical result.
            if target.is_file() and not target.is_symlink():
                n, sha = hash_path(target)
                if n == expected_size and sha == expected_sha:
                    status = 'concurrent_identical_preserved'
                else:
                    raise RuntimeError(f'target appeared with different bytes: {target}')
            else:
                raise RuntimeError(f'target appeared non-regular: {target}')
        return status, size, got_sha
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    started = time.monotonic()
    archive_size, archive_sha = hash_path(ARCHIVE)
    if archive_size != EXPECTED_ARCHIVE_BYTES or archive_sha != EXPECTED_ARCHIVE_SHA:
        raise RuntimeError(f'archive identity mismatch: {archive_size} {archive_sha}')
    collection_sha = hashlib.sha256(COLLECTION.read_bytes()).hexdigest()
    if collection_sha != EXPECTED_COLLECTION_SHA:
        raise RuntimeError(f'local restored source collection manifest SHA mismatch: {collection_sha}')

    archive_fd = os.open(ARCHIVE, os.O_RDONLY)
    if hasattr(os, 'posix_fadvise'):
        try:
            os.posix_fadvise(archive_fd, 0, archive_size, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
    with tarfile.open(ARCHIVE, mode='r:') as tf:
        members = tf.getmembers()
        if len(members) != 1180:
            raise RuntimeError(f'expected 1180 tar members including embedded manifest, got {len(members)}')
        manifest_matches = [m for m in members if m.name == MANIFEST_NAME]
        if len(manifest_matches) != 1 or not manifest_matches[0].isfile():
            raise RuntimeError('expected one regular embedded selection manifest')
        manifest_member = manifest_matches[0]
        manifest_bytes = tf.extractfile(manifest_member).read()
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        if manifest_sha != EXPECTED_SELECTION_SHA:
            raise RuntimeError(f'embedded selection manifest SHA mismatch: {manifest_sha}')
        plan = json.loads(manifest_bytes)
        if plan.get('schema') != 'paid-artifact-rescue-selection-v1':
            raise RuntimeError('unexpected archive selection schema')
        if plan.get('attempt_id') != EXPECTED_ATTEMPT or plan.get('instance_id') != EXPECTED_INSTANCE:
            raise RuntimeError('embedded lane-B source identity mismatch')
        if plan.get('source_collection_manifest_sha256') != EXPECTED_COLLECTION_SHA:
            raise RuntimeError('embedded source collection manifest SHA mismatch')
        if plan.get('inventory_sha256') != EXPECTED_INVENTORY_SHA:
            raise RuntimeError('embedded source inventory SHA mismatch')
        source_manifest = json.loads(COLLECTION.read_text())
        if source_manifest.get('schema') != 'vast-teacher-remote-manifest-v2' or source_manifest.get('attempt_id') != EXPECTED_ATTEMPT or source_manifest.get('instance_id') != EXPECTED_INSTANCE:
            raise RuntimeError('restored collection manifest identity/schema mismatch')
        if source_manifest.get('inventory_sha256') != EXPECTED_INVENTORY_SHA or source_manifest.get('inventory_complete') is not True or source_manifest.get('quiescent') is not True:
            raise RuntimeError('restored collection manifest is incomplete/non-quiescent or inventory differs')
        rows = plan.get('files')
        if not isinstance(rows, list) or len(rows) != 1179:
            raise RuntimeError('expected 1179 manifest-bound payload files')
        if sum(row.get('size_bytes', -1) for row in rows if isinstance(row, dict)) != EXPECTED_PAYLOAD_BYTES:
            raise RuntimeError('selected payload byte total differs from pinned archive plan')
        source_rows = source_manifest.get('files')
        if not isinstance(source_rows, list) or len(source_rows) != 2736:
            raise RuntimeError('restored source inventory row count differs from pinned complete manifest')
        source_by_remote = {}
        for source_row in source_rows:
            if not isinstance(source_row, dict) or not isinstance(source_row.get('remote_path'), str):
                raise RuntimeError('malformed restored source inventory row')
            source_path = PurePosixPath(source_row['remote_path'])
            if source_path.is_absolute() or any(p in ('', '.', '..') for p in source_path.parts):
                raise RuntimeError(f'unsafe restored source inventory path: {source_row["remote_path"]}')
            if source_row['remote_path'] in source_by_remote:
                raise RuntimeError(f'duplicate restored source inventory path: {source_row["remote_path"]}')
            source_by_remote[source_row['remote_path']] = source_row
        file_rows = {}
        canonical_paths = set()
        for row in rows:
            if not isinstance(row, dict):
                raise RuntimeError('manifest file row is not an object')
            transfer = row.get('transfer_path')
            remote = row.get('remote_path')
            sha = row.get('sha256')
            size = row.get('size_bytes')
            dest = row.get('local_destination')
            if not isinstance(transfer, str) or not isinstance(remote, str):
                raise RuntimeError('missing transfer/remote path in selected artifact row')
            pure = PurePosixPath(transfer)
            source_pure = PurePosixPath(remote)
            if pure.is_absolute() or any(p in ('', '.', '..') for p in pure.parts) or source_pure.is_absolute() or any(p in ('', '.', '..') for p in source_pure.parts):
                raise RuntimeError(f'unsafe archive/source member path: {transfer} / {remote}')
            if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                raise RuntimeError(f'invalid SHA-256 in row: {transfer}')
            if type(size) is not int or size < 0 or not isinstance(dest, str):
                raise RuntimeError(f'invalid size/destination in row: {transfer}')
            source_row = source_by_remote.get(remote)
            if source_row is None or source_row.get('transfer_path') != transfer or source_row.get('sha256') != sha or source_row.get('size_bytes') != size:
                raise RuntimeError(f'selected row does not match restored source inventory: {remote}')
            rel = safe_main_relative(dest)
            if rel in canonical_paths:
                raise RuntimeError(f'duplicate canonical destination: {dest}')
            canonical_paths.add(rel)
            if transfer in file_rows:
                raise RuntimeError(f'duplicate transfer path: {transfer}')
            file_rows[transfer] = (row, rel)
        payload_members = [m for m in members if m.name != MANIFEST_NAME]
        member_names = {m.name for m in payload_members}
        expected_archive_names = {row['remote_path'] for row in rows}
        if len(expected_archive_names) != 1179:
            raise RuntimeError('duplicate remote archive path in selected manifest')
        if len(payload_members) != 1179 or member_names != expected_archive_names:
            missing = sorted(expected_archive_names - member_names)[:5]
            extra = sorted(member_names - expected_archive_names)[:5]
            raise RuntimeError(f'archive/selection member set mismatch: missing={missing} extra={extra}')
        by_name = {m.name:m for m in payload_members}
        for name, m in by_name.items():
            if not m.isfile():
                raise RuntimeError(f'non-regular tar member rejected: {name}')
            row = source_by_remote[name]
            if m.size != row['size_bytes']:
                raise RuntimeError(f'tar header size mismatch: {name}')

        # Read the embedded manifest again via the normal item loop only once; its bytes are separately SHA-pinned.
        manifest_member_offset = manifest_member.offset_data
        if hasattr(os, 'posix_fadvise'):
            try:
                os.posix_fadvise(archive_fd, manifest_member_offset, manifest_member.size, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        results = []
        counts = {}
        for index, row in enumerate(rows, start=1):
            name = row['remote_path']
            transfer_path = row['transfer_path']
            _, rel = file_rows[transfer_path]
            member = by_name[name]
            canonical = MAIN / rel
            target, route = select_target(canonical, rel, row['size_bytes'], row['sha256'])
            src = tf.extractfile(member)
            if src is None:
                raise RuntimeError(f'cannot read archive member: {name}')
            if route.endswith('_preexisting_identical') or route == 'canonical_preexisting_identical':
                got_n, got_sha = hash_stream(src, fadvise_fd=archive_fd, fadvise_offset=member.offset_data, fadvise_len=member.size)
                if got_n != row['size_bytes'] or got_sha != row['sha256']:
                    raise RuntimeError(f'archive member digest mismatch: {name}')
                status = route
            else:
                status, got_n, got_sha = write_no_replace(member, src, target, row['size_bytes'], row['sha256'], archive_fd)
            counts[status] = counts.get(status, 0) + 1
            results.append({
                'remote_path': name,
                'transfer_path': transfer_path,
                'archive_member_name': member.name,
                'category': row.get('category'),
                'size_bytes': got_n,
                'sha256': got_sha,
                'canonical_destination': str(canonical),
                'installed_or_verified_destination': str(target),
                'status': status,
            })
            if index % 50 == 0 or index == len(file_rows):
                print(json.dumps({'phase':'member_install','verified':index,'total':len(file_rows),'counts':counts}), flush=True)
        try:
            os.posix_fadvise(archive_fd, 0, archive_size, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    os.close(archive_fd)

    payload_total = sum(x['size_bytes'] for x in results)
    receipt = {
        'schema':'wnn.vast-paid-artifact-archive-member-install.v2',
        'recorded_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'ownership':{'attempt_id':EXPECTED_ATTEMPT,'instance_id':EXPECTED_INSTANCE,'lane':'b'},
        'archive':{'path':str(ARCHIVE),'size_bytes':archive_size,'sha256':archive_sha,'selection_manifest_sha256':manifest_sha,'payload_members':len(results),'payload_bytes':payload_total},
        'sources':{'collection_manifest_path':str(COLLECTION),'collection_manifest_sha256':collection_sha,'inventory_sha256':EXPECTED_INVENTORY_SHA},
        'install_policy':'Per-member archive bytes/SHA verified; existing canonical files preserved; mismatches routed to source-qualified namespace; atomic hard-link no-replace install; finished files receive bounded DONTNEED advice.',
        'source_variant_root':str(VARIANT_ROOT),
        'counts':counts,
        'elapsed_seconds':round(time.monotonic()-started,3),
        'files':results,
        'complete':len(results)==1179 and payload_total==732881211,
        'scientific_promotion':False,
        'public_or_validation_score_claim':False,
    }
    if not receipt['complete']:
        raise RuntimeError('not all 1179 manifest members accounted for')
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    if RECEIPT.exists():
        raise RuntimeError(f'receipt path already exists: {RECEIPT}')
    fd, temp_name = tempfile.mkstemp(prefix=f'.{RECEIPT.name}.', suffix='.part', dir=RECEIPT.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(receipt, out, sort_keys=True, separators=(',',':'))
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.link(temp, RECEIPT)
        fsync_dir(RECEIPT.parent)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    print(json.dumps({'phase':'complete','receipt':str(RECEIPT),'counts':counts,'payload_bytes':payload_total,'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'phase':'failed','error':f'{type(exc).__name__}: {exc}'}),flush=True)
        raise
