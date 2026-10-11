#!/usr/bin/env python3
"""Collect exactly the manifest-bound missing training-state artifacts from lane B."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

MAIN = Path('/home/vlad/development/wunder/wnn_connectome_starterpack').resolve()
BASE = MAIN / 'research/campaign_2026-10-06/bandit_continuation/diversity_stack_20261009/ops_continuation_20261011'
CONTROLLER = MAIN / 'research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1/controller.py'
SOURCE_MANIFEST = MAIN / 'research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1/collected_b/collection_manifest.json'
SOURCE_SHA = 'b4e496252c4b28884f37854e4705b2ff4751bafd9cf965317c543a2957cf94c6'
INSTANCE_ID = 55080658
OUTDIR = MAIN / 'research/campaign_2026-10-06/bandit_continuation/storage_handoff_20261011/paid_training_state_r1'
SELECTION = BASE / 'PAID_STATE_ARCHIVE_SELECTION_R1.json'
RECEIPT = BASE / 'monitor_20261011_1730_r2/PAID_TRAINING_STATE_ARCHIVE_R1.json'
VARIANT_ROOT = MAIN / 'artifacts/source-qualified/vast-lane-b-55080658-paid-training-state-r1'
REMOTE_ARCHIVE_PREFIX = '/root/run/paid_training_states_22_'
GROSS_PRIOR_BYTES = 915_033_984  # 179,627,904 + 735,406,080 already received.
GROSS_LIMIT_BYTES = 1_500_000_000
EXPECTED_FILES = 22
EXPECTED_PAYLOAD_BYTES = 276_788_229
CHUNK = 2 * 1024 * 1024


def sha256_path(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open('rb', buffering=0) as source:
        while True:
            chunk = source.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        result = digest.hexdigest()
        if hasattr(os, 'posix_fadvise'):
            try:
                os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
        return size, result


def sha256_stream(source, destination=None) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = source.read(CHUNK)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
        if destination is not None:
            destination.write(chunk)
    return size, digest.hexdigest()


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as target:
            json.dump(value, target, sort_keys=True, indent=2)
            target.write('\n')
            target.flush()
            os.fsync(target.fileno())
        os.link(tmp, path)
        fsync_directory(path.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def load_lane_b_controller():
    # The controller captures VT_LANE at import time; never reuse a prior import.
    os.environ['VT_LANE'] = 'b'
    spec = importlib.util.spec_from_file_location('wnn_paid_state_lane_b', CONTROLLER)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load MAIN lane-B controller')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module.LANE != 'b' or module.RECEIPT.name != 'OWNERSHIP_RECEIPT_b.json':
        raise RuntimeError('controller did not capture lane B')
    ownership = module.receipt()
    if ownership.get('instance_id') != INSTANCE_ID:
        raise RuntimeError('lane-B ownership receipt instance mismatch')
    return module


def canonical_for_row(row: dict) -> Path:
    transfer_path = row['transfer_path']
    if not transfer_path.startswith('wnn/'):
        raise RuntimeError('unexpected collection transfer path')
    rel = PurePosixPath(transfer_path[4:])
    if rel.is_absolute() or any(part in ('', '.', '..') for part in rel.parts):
        raise RuntimeError('unsafe transfer path in collection manifest')
    if not str(rel).startswith('artifacts/runs/') or rel.name != 'training_state.pt':
        raise RuntimeError('selected non-training-state artifact')
    return MAIN.joinpath(*rel.parts)


def selection_rows(source: dict) -> list[dict]:
    rows = []
    for row in source['files']:
        if not row['remote_path'].endswith('/training_state.pt'):
            continue
        canonical = canonical_for_row(row)
        rel = PurePosixPath(row['transfer_path'][4:])
        variant = VARIANT_ROOT.joinpath(*rel.parts)
        if not canonical.exists() and not variant.exists():
            rows.append({
                'remote_path': row['remote_path'],
                'archive_member': str(rel),
                'canonical_destination': str(canonical),
                'source_variant_destination': str(variant),
                'size_bytes': row['size_bytes'],
                'sha256': row['sha256'],
            })
    rows.sort(key=lambda row: row['remote_path'])
    return rows


def install_member(member: tarfile.TarInfo, source, row: dict) -> dict:
    expected_size = row['size_bytes']
    expected_sha = row['sha256']
    canonical = Path(row['canonical_destination'])
    variant = Path(row['source_variant_destination'])

    def existing_status(path: Path, label: str):
        if path.is_symlink():
            raise RuntimeError(f'{label} destination is a symlink: {path}')
        if not path.exists():
            return None
        if not path.is_file():
            raise RuntimeError(f'{label} destination is not a regular file: {path}')
        size, digest = sha256_path(path)
        if size == expected_size and digest == expected_sha:
            return label + '_preexisting_identical'
        return label + '_preexisting_different'

    prior = existing_status(canonical, 'canonical')
    prior_variant = existing_status(variant, 'source_variant')
    if prior == 'canonical_preexisting_identical':
        return {'destination': str(canonical), 'status': prior, 'size_bytes': expected_size, 'sha256': expected_sha}
    if prior_variant == 'source_variant_preexisting_identical':
        return {'destination': str(variant), 'status': prior_variant, 'size_bytes': expected_size, 'sha256': expected_sha}
    if prior_variant is not None:
        raise RuntimeError(f'conflicting source-qualified state exists: {variant}')
    target = canonical if prior is None else variant
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.resolve().is_relative_to(MAIN) is False:
        raise RuntimeError(f'destination escapes MAIN: {target}')

    fd, temp_name = tempfile.mkstemp(prefix='.' + target.name + '.rescue-', suffix='.part', dir=target.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, 'wb', buffering=0) as output:
            size, digest = sha256_stream(source, output)
            output.flush()
            os.fsync(output.fileno())
            if hasattr(os, 'posix_fadvise'):
                try:
                    os.posix_fadvise(output.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
        if size != expected_size or digest != expected_sha:
            raise RuntimeError(f'member digest mismatch: {member.name}')
        try:
            os.link(temp, target)
            fsync_directory(target.parent)
            status = 'installed_no_replace'
        except FileExistsError:
            # Preserve a concurrently-created canonical file; use the source-qualified path.
            if target == canonical:
                concurrent = existing_status(canonical, 'canonical')
                if concurrent == 'canonical_preexisting_identical':
                    status = 'concurrent_identical_preserved'
                    target = canonical
                else:
                    concurrent_variant = existing_status(variant, 'source_variant')
                    if concurrent_variant == 'source_variant_preexisting_identical':
                        status = 'source_variant_preexisting_identical'
                        target = variant
                    elif concurrent_variant is None:
                        try:
                            os.link(temp, variant)
                            fsync_directory(variant.parent)
                            status = 'installed_source_qualified_no_replace'
                            target = variant
                        except FileExistsError as exc:
                            raise RuntimeError('source-qualified target raced; rerun exact inventory') from exc
                    else:
                        raise RuntimeError('canonical and source-qualified destinations conflict')
            else:
                raise RuntimeError('source-qualified target appeared during install')
        return {'destination': str(target), 'status': status, 'size_bytes': size, 'sha256': digest}
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    started = time.monotonic()
    if not SOURCE_MANIFEST.is_file():
        raise RuntimeError('pinned local source collection manifest is missing')
    source_size, source_sha = sha256_path(SOURCE_MANIFEST)
    if source_sha != SOURCE_SHA:
        raise RuntimeError('source collection manifest SHA mismatch')
    source = json.loads(SOURCE_MANIFEST.read_text())
    if (source.get('schema') != 'vast-teacher-remote-manifest-v2'
            or source.get('attempt_id') != 'vast-teachers-r1-lane-b-20261010'
            or source.get('instance_id') != INSTANCE_ID
            or source.get('inventory_complete') is not True
            or source.get('quiescent') is not True):
        raise RuntimeError('source collection manifest is not the expected complete, quiescent lane-B inventory')
    rows = selection_rows(source)
    payload_bytes = sum(row['size_bytes'] for row in rows)
    if len(rows) != EXPECTED_FILES or payload_bytes != EXPECTED_PAYLOAD_BYTES:
        raise RuntimeError(f'missing-state selection changed: {len(rows)} files / {payload_bytes} bytes')
    if SELECTION.exists() or RECEIPT.exists() or OUTDIR.exists():
        raise RuntimeError('selection, receipt, or staging destination already exists; no overwrite')
    selection = {
        'schema': 'wnn.paid-training-state-selection.v1',
        'source_collection_manifest': str(SOURCE_MANIFEST),
        'source_collection_manifest_sha256': source_sha,
        'source_inventory_sha256': source['inventory_sha256'],
        'attempt_id': source['attempt_id'],
        'instance_id': INSTANCE_ID,
        'selection_rule': 'Only training_state.pt files in the pinned source inventory absent from both canonical and source-qualified local paths at selection time.',
        'file_count': len(rows),
        'payload_bytes': payload_bytes,
        'files': rows,
    }
    write_json_new(SELECTION, selection)
    selection_size, selection_sha = sha256_path(SELECTION)
    remote_archive = REMOTE_ARCHIVE_PREFIX + selection_sha[:16] + '.tar'
    out_free = shutil.disk_usage(OUTDIR.parent).free
    if out_free < 2_000_000_000:
        raise RuntimeError(f'insufficient disk staging reserve: {out_free} bytes')
    if GROSS_PRIOR_BYTES + payload_bytes + 1024 * 1024 > GROSS_LIMIT_BYTES:
        raise RuntimeError('gross transfer cap would be exceeded')
    OUTDIR.mkdir(mode=0o700)
    fsync_directory(OUTDIR.parent)
    part = OUTDIR / (Path(remote_archive).name + '.part')
    final = OUTDIR / Path(remote_archive).name
    if part.exists() or final.exists():
        raise RuntimeError('archive staging file already exists; preserve it and inspect before retry')

    module = load_lane_b_controller()
    remote_paths = [row['archive_member'] for row in rows]
    check_lines = []
    for row in rows:
        remote_rel = row['archive_member']
        remote_abs = '/root/wnn/' + remote_rel
        quoted = shlex.quote(remote_abs)
        check_lines.extend([
            f'test -f {quoted}',
            f'n=$(stat -c %s -- {quoted}); test "$n" = {row["size_bytes"]}',
            f'h=$(sha256sum -- {quoted}); h=${{h%% *}}; test "$h" = {shlex.quote(row["sha256"])}',
            f'printf "STATE_FILE\\t{shlex.quote(remote_rel)}\\t%s\\t%s\\n" "$n" "$h"',
        ])
    tar_args = ' '.join(shlex.quote(path) for path in remote_paths)
    remote_script = '\n'.join([
        'set -euo pipefail',
        'test -d /root/wnn && test -d /root/run',
        f'archive={shlex.quote(remote_archive)}',
        'test ! -e "$archive"',
        'free=$(df -B1 --output=avail /root/run | tail -n 1 | tr -d " ")',
        'test "$free" -ge 2000000000',
        *check_lines,
        f'tar --format=gnu -cf "$archive" -C /root/wnn -- {tar_args}',
        'bytes=$(stat -c %s -- "$archive")',
        'digest=$(sha256sum -- "$archive"); digest=${digest%% *}',
        'printf "STATE_ARCHIVE_META\\t%s\\t%s\\t%s\\n" "$bytes" "$digest" "$free"',
    ])
    result = module.ssh('bash -lc ' + shlex.quote(remote_script), timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f'lane-B exact-state archive preflight/build failed with rc={result.returncode}')
    remote_member_rows = []
    archive_meta = None
    for line in result.stdout.splitlines():
        fields = line.split('\t')
        if fields[0] == 'STATE_FILE' and len(fields) == 4:
            remote_member_rows.append(fields[1:])
        elif fields[0] == 'STATE_ARCHIVE_META' and len(fields) == 4:
            archive_meta = fields[1:]
    if len(remote_member_rows) != len(rows) or archive_meta is None:
        raise RuntimeError('lane-B response omitted exact source or archive digest receipts')
    for expected, observed in zip(rows, remote_member_rows, strict=True):
        if observed != [expected['archive_member'], str(expected['size_bytes']), expected['sha256']]:
            raise RuntimeError('remote state file disagrees with frozen source inventory')
    remote_bytes, remote_sha, remote_free = int(archive_meta[0]), archive_meta[1], int(archive_meta[2])
    if remote_bytes < payload_bytes or remote_bytes > payload_bytes + 1024 * 1024:
        raise RuntimeError('remote tar byte count is outside expected header/padding overhead')
    if len(remote_sha) != 64:
        raise RuntimeError('invalid remote archive digest')
    if GROSS_PRIOR_BYTES + remote_bytes > GROSS_LIMIT_BYTES:
        raise RuntimeError('actual archive would exceed authorized gross transfer cap')

    transfer_started = time.monotonic()
    module.scp([remote_archive], str(part), upload=False, timeout=600)
    if not part.is_file() or part.is_symlink():
        raise RuntimeError('SCP did not produce a regular disk-backed archive')
    local_bytes, local_sha = sha256_path(part)
    if local_bytes != remote_bytes or local_sha != remote_sha:
        raise RuntimeError('received training-state archive does not match remote size/SHA')
    if final.exists():
        raise RuntimeError('final archive path appeared; partial is preserved')
    os.replace(part, final)
    fsync_directory(OUTDIR)
    transfer_seconds = time.monotonic() - transfer_started

    expected_members = {row['archive_member']: row for row in rows}
    installed = []
    with tarfile.open(final, mode='r:') as archive:
        members = archive.getmembers()
        if len(members) != len(rows) or {member.name for member in members} != set(expected_members):
            raise RuntimeError('tar member names/count differ from exact state selection')
        for member in members:
            row = expected_members[member.name]
            if not member.isfile() or member.size != row['size_bytes']:
                raise RuntimeError(f'unsafe or wrong-size archive member: {member.name}')
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f'cannot read archive member: {member.name}')
            installed.append({
                'archive_member': member.name,
                'remote_path': row['remote_path'],
                'expected_sha256': row['sha256'],
                'expected_size_bytes': row['size_bytes'],
                **install_member(member, stream, row),
            })
    installed_bytes = sum(item['size_bytes'] for item in installed)
    if installed_bytes != payload_bytes:
        raise RuntimeError('installed training-state byte total changed')
    statuses: dict[str, int] = {}
    for item in installed:
        statuses[item['status']] = statuses.get(item['status'], 0) + 1
    receipt = {
        'schema': 'wnn.paid-training-state-archive-install.v1',
        'recorded_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'complete': True,
        'ownership': {'lane': 'b', 'instance_id': INSTANCE_ID, 'attempt_id': source['attempt_id']},
        'source_collection_manifest': str(SOURCE_MANIFEST),
        'source_collection_manifest_sha256': source_sha,
        'source_inventory_sha256': source['inventory_sha256'],
        'selection_manifest': str(SELECTION),
        'selection_manifest_sha256': selection_sha,
        'selection_manifest_bytes': selection_size,
        'remote_archive_path': remote_archive,
        'local_archive_path': str(final),
        'archive_bytes': local_bytes,
        'archive_sha256': local_sha,
        'payload_files': len(rows),
        'payload_bytes': payload_bytes,
        'gross_transfer_bytes_including_prior_receipts': GROSS_PRIOR_BYTES + local_bytes,
        'authorized_gross_transfer_cap_bytes': GROSS_LIMIT_BYTES,
        'remote_free_bytes_at_archive_creation': remote_free,
        'transfer_seconds': round(transfer_seconds, 3),
        'total_elapsed_seconds': round(time.monotonic() - started, 3),
        'collector_script': str(Path(__file__).resolve()),
        'collector_script_sha256': sha256_path(Path(__file__))[1],
        'files': installed,
        'counts': statuses,
        'non_overwrite_policy': 'Atomic hard-link no-replace installation. Existing canonical bytes are preserved; conflicting canonical variants use the lane-B source-qualified destination. No remote/provider lifecycle or billing state was changed.',
        'scientific_score_claim': False,
    }
    write_json_new(RECEIPT, receipt)
    print(json.dumps({'result': 'PASS', 'receipt': str(RECEIPT), 'archive': str(final), 'archive_bytes': local_bytes, 'archive_sha256': local_sha, 'payload_files': len(rows), 'payload_bytes': payload_bytes, 'gross_bytes': GROSS_PRIOR_BYTES + local_bytes, 'statuses': statuses, 'elapsed_seconds': round(time.monotonic() - started, 3)}, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Do not print controller outputs, connection details, or credentials.
        print(json.dumps({'result': 'FAIL', 'error_type': type(exc).__name__, 'error': str(exc)[:240]}, sort_keys=True), flush=True)
        raise
