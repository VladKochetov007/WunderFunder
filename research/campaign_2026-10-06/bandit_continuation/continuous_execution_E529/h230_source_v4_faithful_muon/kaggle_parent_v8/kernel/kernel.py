from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

BUNDLE_SLUG = "h230-seed9-parent-v8"
EXPECTED_ARCHIVE_SHA256 = "b279d7dcf6b0326ecc43c2c4a50f255d7b85b7725f69b7940b2bf5be835019c6"
DEFERRED_CACHE_ARRAYS = {
    "cache/bd26_h230_fold0_teacher_family_all_rows_v1/main_source_train.npy",
    "cache/bd26_h230_fold0_teacher_family_all_rows_v1/main_source_valid.npy",
}
EXPECTED_EXTERNAL_MOUNTS = {
    ".transport/fold0-assets-manifest.json": {
        "dataset": "vladkochetov07/wnn-c20261004-fold0-assets",
        "filename": "assets-manifest.json",
    },
    ".transport/h230-parent-cache-assets-manifest.json": {
        "dataset": "vladkochetov07/h230-seed9-parent-source-cache-v2",
        "filename": "assets-manifest.json",
    },
    "artifacts/runs/c20260929_prop_f0/model.pt": {
        "dataset": "vladkochetov07/wnn-c20261004-fold0-assets",
        "filename": "b243b9d44a71bcaa02589be0a38b19477f448f9da98a02e16495af6730b517a7.wnn",
    },
    "cache/bd26_h230_fold0_teacher_family_all_rows_v1/main_source_train.npy": {
        "dataset": "vladkochetov07/h230-seed9-parent-source-cache-v2",
        "filename": "d706c979f3fae46258ee9684dc754d02a70c35d142fed5ebaf785ae7e1ded165.wnn",
    },
    "cache/bd26_h230_fold0_teacher_family_all_rows_v1/main_source_valid.npy": {
        "dataset": "vladkochetov07/h230-seed9-parent-source-cache-v2",
        "filename": "e464d29ca5ec0c3c07e70e1f8b08afb180b5d35b70b4fb3130e6a8f56d0e1a8c.wnn",
    },
    "cache/bd26_h230_fold0_teacher_family_all_rows_v1/metadata.json": {
        "dataset": "vladkochetov07/h230-seed9-parent-source-cache-v2",
        "filename": "b05e652f5fdec4e6b1e4e07d1d3a705655d4588ab8ee9d802ccdf4339fd66500.wnn",
    },
    "cache/train_targets.npz": {
        "dataset": "vladkochetov07/wnn-c20261004-fold0-assets",
        "filename": "ead1a05e7aa5ac3d1c10cde65986be9f48fd342cadb95330611e58c3a8f817e9.wnn",
    },
    "cache/valid_targets.npz": {
        "dataset": "vladkochetov07/wnn-c20261004-fold0-assets",
        "filename": "dbb7a1a48dd0af9fa524a80c2694008a56cb91d5dae80e8d0d4984c0c5622955.wnn",
    },
    "datasets/train.parquet": "train.parquet",
    "datasets/valid.parquet": "valid.parquet",
    "research/campaign_2026-09-29/artifacts/splits/fold0.json": {
        "dataset": "vladkochetov07/wnn-c20261004-fold0-assets",
        "filename": "09263ad4fe2b18e2b2f3b79f4562678dce801b067b868134ddc79279f9316a3d.wnn",
    },
}
KERNEL_PROCESS_WALL_SECONDS_MAX = 10_000
RUNNER_FINALIZATION_RESERVE_SECONDS = 300
BOOTSTRAP_STARTED_MONOTONIC = time.monotonic()


def materialize_verified_input(source: Path, target: Path, descriptor: dict) -> None:
    """Copy only containment-sensitive inputs, preserving their exact identity."""
    expected_size = descriptor["size"]
    if source.stat().st_size != expected_size:
        raise ValueError("Contained input transport size mismatch")
    if shutil.disk_usage(target.parent).free < expected_size + 2 * 1024**3:
        raise ValueError("Insufficient disk for contained input with reserve")
    digest = hashlib.sha256()
    copied = 0
    try:
        with source.open("rb") as origin, target.open("xb") as destination:
            while chunk := origin.read(8 * 1024**2):
                copied += len(chunk)
                digest.update(chunk)
                destination.write(chunk)
        if copied != expected_size or digest.hexdigest() != descriptor["sha256"]:
            raise ValueError("Contained input transport hash or size mismatch")
    except BaseException:
        # The destination is newly created; preserve any pre-existing target.
        if copied and target.is_file() and not target.is_symlink():
            target.unlink()
        raise
    if target.is_symlink() or target.resolve().parent != target.parent.resolve():
        raise ValueError("Contained input transport did not materialize a regular file")


def materialize_verified_cache_alias(source: Path, target: Path, descriptor: dict) -> Path:
    """Verify a cache input and expose it through an in-cache symlink.

    The worker defers a second full identity scan for these two arrays. Keeping
    the alias inside the declared cache root satisfies both the runner's
    deferred-mount guard and the cache validator's containment check.
    """
    backing_dir = target.parent / ".transport-materialized"
    backing_dir.mkdir(exist_ok=True)
    backing = backing_dir / target.name
    materialize_verified_input(source, backing, descriptor)
    created_link = False
    try:
        target.symlink_to(Path(backing_dir.name) / backing.name)
        created_link = True
        observed_size = target.stat().st_size
        resolved_target = target.resolve(strict=True)
        if (
            not target.is_symlink()
            or observed_size != descriptor["size"]
            or not resolved_target.is_relative_to(target.parent.resolve())
        ):
            raise ValueError("Verified cache alias failed size or containment checks")
        return backing
    except BaseException:
        if created_link and target.is_symlink():
            target.unlink()
        if backing.is_file() and not backing.is_symlink():
            backing.unlink()
        raise


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
    available = int(deadline_monotonic - now - RUNNER_FINALIZATION_RESERVE_SECONDS)
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
    worker_seconds = worker_timeout_for_deadline(limits["timeout_seconds"], deadline_monotonic)
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
    materialized_backings = []
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
        if logical in DEFERRED_CACHE_ARRAYS:
            descriptor = next(item for item in manifest["files"] if item["path"] == logical)
            backing = materialize_verified_cache_alias(candidates[0], target, descriptor)
            materialized_backings.append(backing)
            print(
                json.dumps(
                    {
                        "event": "verified_internal_cache_alias",
                        "path": logical,
                        "expected_size": descriptor["size"],
                        "observed_size": target.stat().st_size,
                        "symlink": target.is_symlink(),
                        "resolved_inside_cache": target.resolve().is_relative_to(
                            target.parent.resolve()
                        ),
                    }
                ),
                flush=True,
            )
        elif (
            logical == "research/campaign_2026-09-29/artifacts/splits/fold0.json"
            or logical.startswith("cache/bd26_h230_fold0_teacher_family_all_rows_v1/")
        ):
            descriptor = next(item for item in manifest["files"] if item["path"] == logical)
            materialize_verified_input(candidates[0], target, descriptor)
        else:
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
            raise TimeoutError("kernel bootstrap exhausted its 10,000-second process budget")
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
                "runner exceeded the 10,000-second kernel-process budget; its process tree was stopped"
            ) from error
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        if remaining_seconds(kernel_deadline) <= 0:
            raise TimeoutError("runner completed outside the 10,000-second kernel-process budget")
    finally:
        release_mounts(root, manifest["external_mounts"])
        for backing in materialized_backings:
            if (
                backing.is_file()
                and not backing.is_symlink()
                and backing.resolve().is_relative_to(root.resolve())
            ):
                backing.unlink()
        for parent in {path.parent for path in materialized_backings}:
            if parent.is_dir():
                parent.rmdir()


if __name__ == "__main__":
    main()
