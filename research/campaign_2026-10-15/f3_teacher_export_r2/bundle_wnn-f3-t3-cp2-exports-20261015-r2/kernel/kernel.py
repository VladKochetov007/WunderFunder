from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

BUNDLE_SLUG = 'wnn-f3-t3-cp2-exports-20261015-r2'
EXPECTED_ARCHIVE_SHA256 = '07a0299427b33260f250a6a5dc2cc958f36a084c30d5685126e442ca57593902'
EXPECTED_EXTERNAL_MOUNTS = {'artifacts/runs/c20260929_prop_f3/model.pt': {'dataset': 'vladkochetov07/wnn-strategy-clean-t3-f3-assets-20261013-r1', 'filename': '8898ef3d31533058c354a16d00632b7c7adca461eaa813f0bed1f59a1fef59b8.wnn'}, 'artifacts/runs/clean_t3_f3_muonall_h384_e3_s3_r5/model_ep2.pt': {'dataset': 'vladkochetov07/wnn-f3-t3-cp2-assets-20261015-r1', 'filename': 'd67bed97eaa16f73338ea275c316bf76c785b37f8451aa5cee8fe5daadf2f1b5.wnn'}, 'artifacts/runs/clean_t3_f3_muonall_h384_e3_s3_r5/provenance.json': {'dataset': 'vladkochetov07/wnn-f3-t3-cp2-assets-20261015-r1', 'filename': 'c60719d1db7d44d17e04e0379f576616a43c6ad2564fd58e192ced5453bc8da3.wnn'}, 'cache/chain_folds.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'a2cfe843a7ce3085d383524a9d1438154fba4f2e0af6df48ee1e1b015e5c942f.wnn'}, 'cache/chains.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '39aba46fad7b717830e90687e1c4a6991794864cd2cf621b0bbf341f2d9a7766.wnn'}, 'cache/propcache_c20260929_prop_f3_train.json': {'dataset': 'vladkochetov07/wnn-strategy-clean-t3-f3-assets-20261013-r1', 'filename': 'd36e33187ca8352e8a37b79e9e30ad055e33691b5e67df4e16382c9532c4183d.wnn'}, 'cache/propcache_c20260929_prop_f3_train.npy': {'dataset': 'vladkochetov07/wnn-strategy-clean-t3-f3-assets-20261013-r1', 'filename': 'f6f8fd5c8eabd853d5051999d3d2bf7b9c512a60ae7591aae0e1a93b50b0ca08.wnn'}, 'cache/propcache_c20260929_prop_f3_valid.json': {'dataset': 'vladkochetov07/wnn-strategy-clean-t3-f3-assets-20261013-r1', 'filename': '835cf7393b26944f346dab9dc6241be1e398bfd58299234a8d53950cc59951cb.wnn'}, 'cache/propcache_c20260929_prop_f3_valid.npy': {'dataset': 'vladkochetov07/wnn-strategy-clean-t3-f3-assets-20261013-r1', 'filename': '0ed66359db3645679644c71e14777ae02a9124ab7eedf9fa88484cda1d4438b8.wnn'}, 'cache/train_targets.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9.wnn'}, 'cache/valid_targets.npz': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': 'dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955.wnn'}, 'datasets/train.parquet': 'train.parquet', 'datasets/valid.parquet': 'valid.parquet', 'research/campaign_2026-09-29/artifacts/splits/fold3.json': {'dataset': 'vladkochetov07/wnn-strategy-clean-t3-f3-assets-20261013-r1', 'filename': '78157b3347b9e9ba6d2c075d0c53df612e67af9ecd5ee8af7fd7957e406b8343.wnn'}, 'research/campaign_2026-10-04/portable_inputs.json': {'dataset': 'vladkochetov07/wnn-c20261004-fold0-assets', 'filename': '90d26c7063050092eeda774c7c6298897a8fb133e4804ac13d0a281bd328ff8c.wnn'}}
# Kaggle's current maximum kernel session window; each sealed worker spec still
# carries its own timeout and the runner leaves its finalization reserve.
KERNEL_PROCESS_WALL_SECONDS_MAX = 21_600
RUNNER_FINALIZATION_RESERVE_SECONDS = 300
BOOTSTRAP_STARTED_MONOTONIC = time.monotonic()


def remaining_seconds(deadline_monotonic: float, *, now_monotonic: float | None = None) -> float:
    """Return remaining kernel-process budget from the bootstrap-entry deadline."""
    now = time.monotonic() if now_monotonic is None else now_monotonic
    return max(0.0, deadline_monotonic - now)


def require_remaining(deadline_monotonic: float, phase: str) -> None:
    """Fail closed when synchronous bootstrap work has exhausted its deadline."""
    if remaining_seconds(deadline_monotonic) <= 0:
        raise TimeoutError(f"kernel bootstrap exceeded its deadline during {phase}")


def worker_timeout_for_deadline(
    declared_seconds: int,
    deadline_monotonic: float,
    *,
    now_monotonic: float | None = None,
) -> int:
    """Shrink a worker cap to leave time for runner artifact collection and exit."""
    now = time.monotonic() if now_monotonic is None else now_monotonic
    available = int(
        deadline_monotonic - now - RUNNER_FINALIZATION_RESERVE_SECONDS
    )
    return max(0, min(int(declared_seconds), available))


def terminate_process_tree(process: subprocess.Popen, *, grace_seconds: float = 2.0) -> None:
    """Stop only the runner and its descendants, including workers in new sessions."""
    try:
        import psutil
    except ImportError as error:  # the bundled runner requires psutil as well
        raise RuntimeError("cannot safely enforce kernel deadline without psutil") from error

    try:
        root = psutil.Process(process.pid)
        members = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        process.wait()
        return

    # The runner creates a separate session for each admitted worker. Kill the
    # process tree by identity, not the bootstrap's process group alone.
    for child in members[1:]:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        root.terminate()
    except psutil.NoSuchProcess:
        pass
    _, alive = psutil.wait_procs(members, timeout=grace_seconds)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=grace_seconds)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        # The Popen itself is also in our process tree; this is a final bounded
        # fallback if the process changed state during psutil's snapshot.
        try:
            os.kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=grace_seconds)


def write_runtime_spec(source: Path, destination: Path, deadline_monotonic: float) -> int:
    """Copy a sealed recipe spec with a runtime timeout reduced to current headroom."""
    spec = json.loads(source.read_text(encoding="utf-8"))
    limits = spec.get("limits")
    if not isinstance(limits, dict) or not isinstance(limits.get("timeout_seconds"), int):
        raise ValueError("kernel spec has no integer worker timeout")
    worker_seconds = worker_timeout_for_deadline(
        limits["timeout_seconds"], deadline_monotonic
    )
    if worker_seconds < 1:
        raise TimeoutError("kernel deadline leaves no worker time after finalization reserve")
    limits["timeout_seconds"] = worker_seconds
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return worker_seconds


def release_mounts(root: Path, mounts: dict) -> None:
    """Unlink only external input symlinks before Kaggle captures working outputs."""
    for logical in mounts:
        target = root / logical
        if target.is_symlink():
            target.unlink()


def main() -> None:
    """Extract a verified private bundle and mount hashed raw data without copies."""
    kernel_started = BOOTSTRAP_STARTED_MONOTONIC
    kernel_deadline = kernel_started + KERNEL_PROCESS_WALL_SECONDS_MAX
    require_remaining(kernel_deadline, "input discovery")
    matches = list(Path("/kaggle/input").glob(f"**/{BUNDLE_SLUG}/bundle-manifest.json"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one named private bundle")
    manifest = json.loads(matches[0].read_text())
    if manifest.get("private") is not True:
        raise ValueError("Bundle is not marked private")
    if manifest.get("external_mounts") != EXPECTED_EXTERNAL_MOUNTS:
        raise ValueError("External mappings differ from the controller-bound kernel")
    require_remaining(kernel_deadline, "bundle metadata verification")
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
    require_remaining(kernel_deadline, "bundle archive verification")
    root = Path("/kaggle/working/experiment")
    if shutil.disk_usage(root.parent).free < manifest["expanded_bytes"] + 2 * 1024**3:
        raise ValueError("Insufficient working disk to extract bundle with reserve")
    root.mkdir(exist_ok=False)
    allowed = {item["path"] for item in manifest["files"]} | set(manifest["spec_paths"])
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            require_remaining(kernel_deadline, "bundle extraction")
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
    require_remaining(kernel_deadline, "bundle extraction")
    for logical, mapping in manifest["external_mounts"].items():
        require_remaining(kernel_deadline, "external input mounting")
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
    command = [
        sys.executable,
        str(root / "scripts/kaggle_gpu_runner.py"),
        "--root",
        str(root),
        "--output",
        "/kaggle/working/results",
    ]
    for name in manifest["spec_paths"]:
        require_remaining(kernel_deadline, "runtime deadline setup")
        source_spec = root / name
        runtime_spec = Path("/kaggle/working/runtime_specs") / PurePosixPath(name).name
        write_runtime_spec(source_spec, runtime_spec, kernel_deadline)
        command.extend(["--spec", str(runtime_spec)])
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    environment = os.environ.copy()
    environment["WNN_KERNEL_DEADLINE_MONOTONIC"] = repr(kernel_deadline)
    try:
        require_remaining(kernel_deadline, "runner launch")
        remaining = remaining_seconds(kernel_deadline)
        if remaining <= 0:
            raise TimeoutError("kernel bootstrap exhausted its provider session budget")
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            terminate_process_tree(process)
            raise TimeoutError(
                "runner exceeded the provider kernel-process budget; its process tree was stopped"
            ) from error
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        if remaining_seconds(kernel_deadline) <= 0:
            raise TimeoutError("runner completed outside the 6,000-second kernel-process budget")
    finally:
        release_mounts(root, manifest["external_mounts"])


if __name__ == "__main__":
    main()
