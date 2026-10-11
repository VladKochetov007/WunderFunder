#!/usr/bin/env python3
"""Resume the exact pinned lane-B archive after a cgroup-limited transfer interruption."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

MAIN = Path('/home/vlad/development/wunder/wnn_connectome_starterpack')
CTRL_PATH = MAIN / 'research/campaign_2026-10-06/bandit_continuation/vast_teachers_r1/controller.py'
SRC = Path('/tmp/paid_artifact_archive_b4e49625_r1.tar.partial')
REMOTE = '/root/run/paid_artifact_archive_b4e49625_r1.tar'
EXPECTED_SIZE = 735_406_080
EXPECTED_SHA = 'd7cdea95581c938cc7382c3561f6c14e842e970e0cfafc50b9cbb7b31586df4c'
OFFSET = 487_511_040
EXPECTED_REMOTE_PREFIX_SHA = None  # captured from preflight, never assumed
OUTDIR = MAIN / 'research/campaign_2026-10-06/bandit_continuation/storage_handoff_20261011/paid_archive_resume_r2'
PART = OUTDIR / 'paid_artifact_archive_b4e49625_r1.tar.part'
FINAL = OUTDIR / 'paid_artifact_archive_b4e49625_r1.tar'
UNIT = 'wnn-paid-archive-resume-r2-20261011.scope'


def digest(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    n = 0
    with path.open('rb', buffering=0) as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
            n += len(block)
    return n, h.hexdigest()


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    started = time.monotonic()
    if not SRC.is_file() or SRC.stat().st_size != OFFSET:
        raise RuntimeError(f'partial-prefix size mismatch: got {SRC.stat().st_size if SRC.exists() else "missing"}, expected {OFFSET}')
    if FINAL.exists() or PART.exists() or OUTDIR.exists():
        raise RuntimeError('destination exists; fail closed without overwrite')
    usage = shutil.disk_usage(OUTDIR.parent)
    if usage.free < 2_000_000_000:
        raise RuntimeError(f'insufficient disk reserve: {usage.free} bytes')
    if os.statvfs(SRC).f_fsid == os.statvfs(OUTDIR.parent).f_fsid:
        raise RuntimeError('source and staging are on same filesystem; expected tmpfs-to-disk handoff')

    # Lane must be selected before module import: controller captures LANE at import time.
    os.environ['VT_LANE'] = 'b'
    spec = importlib.util.spec_from_file_location('vast_controller_lane_b_archive_resume', CTRL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load main lane-B controller')
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    if m.LANE != 'b' or m.RECEIPT.name != 'OWNERSHIP_RECEIPT_b.json' or m.receipt().get('instance_id') != 55080658:
        raise RuntimeError('lane-B ownership assertion failed')

    # Reassert remote immutability and the already-received byte prefix before any append.
    remote_cmd = (
        'set -eu; f=' + REMOTE +
        '; test -f "$f"; bytes=$(stat -c %s "$f"); test "$bytes" = ' + str(EXPECTED_SIZE) +
        '; sha256sum "$f"; head -c ' + str(OFFSET) + ' "$f" | sha256sum'
    )
    r = m.ssh(remote_cmd, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f'remote immutable-archive preflight failed (rc {r.returncode})')
    lines = [line.split() for line in r.stdout.splitlines() if line.strip()]
    if len(lines) != 2 or lines[0][0] != EXPECTED_SHA or len(lines[1][0]) != 64:
        raise RuntimeError('remote archive SHA/size or prefix SHA assertion failed')
    remote_prefix_sha = lines[1][0]

    OUTDIR.mkdir(mode=0o700)
    fsync_dir(OUTDIR.parent)
    # Copy the intact tmpfs prefix, fsync it, and prove both bytes and digest match.
    source_bytes, source_sha = digest(SRC)
    if source_bytes != OFFSET or source_sha != remote_prefix_sha:
        raise RuntimeError('local tmpfs prefix differs from remote archive prefix')
    with SRC.open('rb', buffering=0) as src, PART.open('xb', buffering=0) as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    staged_prefix_bytes, staged_prefix_sha = digest(PART)
    if staged_prefix_bytes != OFFSET or staged_prefix_sha != source_sha:
        raise RuntimeError('disk-backed prefix copy differs from tmpfs source')
    print(json.dumps({'phase':'prefix_verified','bytes':OFFSET,'sha256':staged_prefix_sha,'remote_archive_sha256':EXPECTED_SHA,'free_bytes_before':usage.free}), flush=True)

    host, port = m.ssh_target(55080658)
    ssh_argv = [
        'ssh', '-i', str(m.KEY), '-p', str(port), '-o', 'StrictHostKeyChecking=no',
        '-o', 'UserKnownHostsFile=/dev/null', '-o', 'ConnectTimeout=20', '-o', 'LogLevel=ERROR',
        f'root@{host}', f'exec tail -c +{OFFSET + 1} -- {REMOTE}'
    ]
    with PART.open('ab', buffering=0) as dst:
        proc = subprocess.Popen(ssh_argv, stdin=subprocess.DEVNULL, stdout=dst, stderr=subprocess.PIPE, text=False)
        try:
            _, stderr = proc.communicate(timeout=1100)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                _, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                _, stderr = proc.communicate()
            dst.flush()
            os.fsync(dst.fileno())
            raise RuntimeError('range-resume SSH reached 1100s; intact partial retained')
        dst.flush()
        os.fsync(dst.fileno())
    if proc.returncode != 0:
        err = (stderr or b'').decode('utf-8', errors='replace')[-400:]
        raise RuntimeError(f'remote byte-range stream failed rc={proc.returncode}: {err}')
    got_bytes, got_sha = digest(PART)
    if got_bytes != EXPECTED_SIZE:
        raise RuntimeError(f'resumed archive size mismatch: {got_bytes} != {EXPECTED_SIZE}; partial retained')
    if got_sha != EXPECTED_SHA:
        raise RuntimeError(f'resumed archive SHA mismatch: {got_sha}; partial retained')
    if FINAL.exists():
        raise RuntimeError('final archive path appeared during transfer; partial retained')
    os.replace(PART, FINAL)
    fsync_dir(OUTDIR)
    elapsed = time.monotonic() - started
    print(json.dumps({'phase':'archive_sha_verified_and_installed','bytes':got_bytes,'sha256':got_sha,'remote_prefix_sha256':remote_prefix_sha,'local_prefix_sha256':source_sha,'resumed_bytes':EXPECTED_SIZE-OFFSET,'elapsed_seconds':round(elapsed,3),'path':str(FINAL)}), flush=True)
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'phase':'failed','error':f'{type(exc).__name__}: {exc}'}), flush=True)
        raise
