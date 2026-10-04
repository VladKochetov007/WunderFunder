from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import signal
import time
import select
import tarfile
from pathlib import Path, PurePosixPath

BUNDLE_SLUG = 'wnn-w384-fold1-dualt4-20261004-v4'
EXPECTED_ARCHIVE_SHA256 = 'fe13b05d3ec6383ec9f08ab41d7273155345103ec640352cfe76e25406bb3b19'
EXPECTED_EXTERNAL_MOUNTS = {'artifacts/runs/c20260929_prop_f1/model.pt': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '1fbf412df8823c703fdf79263830f341269a68e769bc4a7258d93cdf8e40dc5c.wnn'}, 'artifacts/runs/c20260929_since_s3_f1/model.pt': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': 'd550dbc2b005c36477abd6dfcbab8abca2c79c6599638e6c4b3db920bdd08b12.wnn'}, 'artifacts/runs/c20260929_since_s3_f1/provenance.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': 'fecd314bd2973f33979222e5c05df9c499e4b6a7ecb7f6e0619f05fa0f07fbd4.wnn'}, 'artifacts/runs/c20260929_since_s4_f1/model.pt': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '95682ebffcc01970ffd72ea0a369852def3cdd8d1e1e7f046363a12365274faf.wnn'}, 'artifacts/runs/c20260929_since_s4_f1/provenance.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '293f01c644c4854d301c436aa9b04de869df150874db5cb4129e7de741d1d392.wnn'}, 'cache/chain_folds.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': 'a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f.wnn'}, 'cache/chains.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766.wnn'}, 'cache/pred_c20260929_since_s3_f1_train.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': 'b90c0ac48c47b13a16561d2a67219647fd73e6c85200cdeda3c50b67d2655374.wnn'}, 'cache/pred_c20260929_since_s3_f1_train.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '44526f6e37b07a6fdeba8270ddc86bd5d8e18ea35bd066de7b053fdb37d81102.wnn'}, 'cache/pred_c20260929_since_s3_f1_valid.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '99503b4cfb0cca9f7af2e4c658d6f389d1e81735594bc97e97561c9a800d8a24.wnn'}, 'cache/pred_c20260929_since_s3_f1_valid.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '6f38cdbb19023ac20ddda76e093849c45012422f8765a59bbcf2c3cf5539175f.wnn'}, 'cache/pred_c20260929_since_s4_f1_train.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': 'de0ec63c0a9e0a8b1d1ef97ece9edcb8c7868af3d6e569cc07e1ad77733df8f2.wnn'}, 'cache/pred_c20260929_since_s4_f1_train.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '63e5fac9d276d3ec4cc438240535c89c66620a264f1c5454e34cafac40f1f4d8.wnn'}, 'cache/pred_c20260929_since_s4_f1_valid.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '8aa448f27f87e17de245240eb75a19a6207da32b3e28dc2ccc9544d663c54c2c.wnn'}, 'cache/pred_c20260929_since_s4_f1_valid.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': 'e7461af12689e776510843fdfaf30ac8493ab362d7e1fe5f2cd2f60e3f626948.wnn'}, 'cache/propcache_c20260929_prop_f1_train.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '8f21d3d64c5c495f86cb7942dc237c03d772e8990198f866a29c6f4ffa8435e3.wnn'}, 'cache/propcache_c20260929_prop_f1_train.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '1c01d75a517b95764e04591bede5fd08ed63a27ab753f5097c6c1eb1969d9247.wnn'}, 'cache/propcache_c20260929_prop_f1_valid.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': 'f5fabd1c645a7577972f7fc34c059a5bce7e269ab0cfca35b6e14e63db1f2612.wnn'}, 'cache/propcache_c20260929_prop_f1_valid.npy': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '9ee80755e0d50b7e3869462926b70260304913a3f5dbc1d61dac82d0c242c349.wnn'}, 'cache/train_targets.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': 'ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9.wnn'}, 'cache/valid_targets.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': 'dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955.wnn'}, 'datasets/train.parquet': 'train.parquet', 'datasets/valid.parquet': 'valid.parquet', 'datasets/valid_mask.parquet': 'valid_mask.parquet', 'research/campaign_2026-09-29/artifacts/splits/fold1.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '31b9dc040761dcc81e99bfe69e0bc446f1522b21ab629b2a9c5a03d94bb006ae.wnn'}, 'research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4/portable_inputs_f1.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold1-assets', 'filename': '551905f3363c65608e2d5a0f49e8eeca362f574ab6d98d443e31cee0790fb5fd.wnn'}}


def release_mounts(root: Path, mounts: dict) -> None:
    """Unlink only external input symlinks before Kaggle captures working outputs."""
    for logical in mounts:
        target = root / logical
        if target.is_symlink():
            target.unlink()


def _kernel_work() -> None:
    """Extract a verified private bundle and mount hashed raw data without copies."""
    matches = list(Path("/kaggle/input").glob(f"**/{BUNDLE_SLUG}/bundle-manifest.json"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one named private bundle")
    manifest = json.loads(matches[0].read_text())
    if manifest.get("private") is not True:
        raise ValueError("Bundle is not marked private")
    if manifest.get("external_mounts") != EXPECTED_EXTERNAL_MOUNTS:
        raise ValueError("External mappings differ from the controller-bound kernel")
    archive = matches[0].parent / "bundle.wnn"
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    if (
        digest.hexdigest() != manifest["archive_sha256"]
        or digest.hexdigest() != EXPECTED_ARCHIVE_SHA256
    ):
        raise ValueError("Private bundle archive identity mismatch")
    root = Path("/kaggle/working/experiment")
    if shutil.disk_usage(root.parent).free < manifest["expanded_bytes"] + 2 * 1024**3:
        raise ValueError("Insufficient working disk to extract bundle with reserve")
    root.mkdir(exist_ok=False)
    allowed = {item["path"] for item in manifest["files"]} | set(manifest["spec_paths"])
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise ValueError("Archive path escapes experiment directory")
            if not member.isfile() or member.name not in allowed or member.name in seen:
                raise ValueError("Unsupported, undeclared or duplicate archive member")
            seen.add(member.name)
            target = root / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise ValueError("Archive member has no file content")
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, 8 * 1024**2)
    for logical, mapping in manifest["external_mounts"].items():
        path = PurePosixPath(logical)
        if path.is_absolute() or ".." in path.parts or logical not in allowed:
            raise ValueError("Unsafe external input mapping")
        dataset = (
            mapping["dataset"] if isinstance(mapping, dict) else "vladkochetov07/wunder-funder"
        )
        filename = mapping["filename"] if isinstance(mapping, dict) else mapping
        slug = dataset.split("/")[1]
        candidates = list(Path("/kaggle/input").glob(f"**/{slug}/{filename}"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one authorized input: {dataset}/{filename}")
        target = root / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(candidates[0])
    command = [sys.executable, str(root / "research/campaign_2026-10-04/kaggle/W384_fold1_dualT4_v4/phase_runner.py")]
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        subprocess.run(command, check=True)
    finally:
        release_mounts(root, manifest["external_mounts"])



def _proc_identity(pid: int) -> tuple[str, int] | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    if len(fields) <= 19:
        return None
    return fields[0], int(fields[19])


def _children(pid: int) -> list[int]:
    try:
        raw = Path(f"/proc/{pid}/task/{pid}/children").read_text().strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return []
    return [int(value) for value in raw.split()] if raw else []


def _descendants(root_pid: int, monitor_pid: int) -> list[int]:
    ordered: list[int] = []
    seen = {root_pid, monitor_pid}

    def visit(pid: int) -> None:
        for child in _children(pid):
            if child in seen:
                continue
            seen.add(child)
            visit(child)
            ordered.append(child)

    visit(root_pid)
    return ordered


def _write_timeout(path: Path, record: dict[str, object]) -> None:
    payload = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(payload)
    os.replace(temp, path)


def _watch(
    root_pid: int,
    root_start_ticks: int,
    timeout_seconds: int,
    report: str,
    ready_fd: int,
) -> None:
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    os.write(ready_fd, b"R")
    os.close(ready_fd)
    monitor_pid = os.getpid()
    started = time.monotonic()
    deadline = started + timeout_seconds
    while not stopping:
        identity = _proc_identity(root_pid)
        if identity is None or identity[1] != root_start_ticks or identity[0] == "Z":
            return
        now = time.monotonic()
        if now >= deadline:
            break
        time.sleep(min(1.0, deadline - now))
    if stopping:
        return

    victims = _descendants(root_pid, monitor_pid)
    captured = []
    for pid in victims:
        identity = _proc_identity(pid)
        if identity is not None:
            captured.append({"pid": pid, "start_ticks": identity[1]})
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    grace_deadline = time.monotonic() + 10.0
    while time.monotonic() < grace_deadline:
        if not any(
            identity is not None and identity[0] != "Z"
            for item in captured
            if (identity := _proc_identity(int(item["pid"]))) is not None
        ):
            break
        time.sleep(0.2)

    killed = []
    for item in captured:
        pid = int(item["pid"])
        identity = _proc_identity(pid)
        if (
            identity is not None
            and identity[0] != "Z"
            and identity[1] == item["start_ticks"]
        ):
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except ProcessLookupError:
                pass

    record = {
        "status": "whole_job_deadline_exceeded",
        "root_pid": root_pid,
        "root_start_ticks": root_start_ticks,
        "timeout_seconds": timeout_seconds,
        "watchdog_elapsed_seconds": time.monotonic() - started,
        "term_sent_pids": [item["pid"] for item in captured],
        "kill_sent_pids": killed,
        "kill_order": "descendants first; main kernel process after timeout record is flushed",
    }
    try:
        _write_timeout(Path(report), record)
    finally:
        identity = _proc_identity(root_pid)
        if identity is not None and identity[1] == root_start_ticks:
            try:
                os.kill(root_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def start_watchdog(
    timeout_seconds: int = 32_400,
    report_path: str = "/kaggle/working/whole_job_watchdog.json",
) -> int:
    """Fork a monitor before any worker setup and return its PID to the parent."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    root_pid = os.getpid()
    identity = _proc_identity(root_pid)
    if identity is None:
        raise RuntimeError("Cannot bind watchdog to the current process identity")
    ready_read_fd, ready_write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(ready_read_fd)
        try:
            _watch(root_pid, identity[1], timeout_seconds, report_path, ready_write_fd)
        except BaseException as exc:
            record = {
                "status": "watchdog_internal_error_fail_closed",
                "root_pid": root_pid,
                "root_start_ticks": identity[1],
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
            try:
                _write_timeout(Path(report_path), record)
            except BaseException:
                pass
            current = _proc_identity(root_pid)
            if current is not None and current[1] == identity[1]:
                try:
                    os.kill(root_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            os._exit(0)
    os.close(ready_write_fd)
    ready, _, _ = select.select([ready_read_fd], [], [], 5.0)
    if not ready or os.read(ready_read_fd, 1) != b"R":
        os.close(ready_read_fd)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)
        raise RuntimeError("Whole-job watchdog failed to start; refusing kernel work")
    os.close(ready_read_fd)
    return pid


def stop_watchdog(pid: int) -> None:
    """Stop and reap the monitor after the kernel's ordinary work finishes."""
    try:
        waited, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return
    if waited == pid:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    while True:
        try:
            os.waitpid(pid, 0)
            return
        except InterruptedError:
            continue



def main() -> None:
    watchdog_pid = start_watchdog(
        timeout_seconds=25_180,
        report_path="/kaggle/working/whole_job_watchdog.json",
    )
    try:
        _kernel_work()
    finally:
        stop_watchdog(watchdog_pid)


if __name__ == "__main__":
    main()
