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

BUNDLE_SLUG = 'e540-v12-cache-dirfix-20261008'
EXPECTED_ARCHIVE_SHA256 = 'ccddc41a2a0a4a0edaa3f05b52b830a3175d21e361dc787d87260acd0533d321'
EXPECTED_EXTERNAL_MOUNTS = {'datasets/valid.parquet': 'valid.parquet', 'e540_assets/linear_control_manifest.json': {'dataset': 'vladkochetov07/e540-v10-artifact-assets-20261008', 'filename': 'af50c15d3b63f9a6eff9680b0d24be15969360feb95e073cfd24b8c5da799198.wnn'}, 'e540_assets/linear_epoch0.pt': {'dataset': 'vladkochetov07/e540-v10-artifact-assets-20261008', 'filename': '17c91be768a7f41cd4c6ae0b2bbce26bcf7d6fbf5e4f6509e38c4be6bd44403a.wnn'}, 'e540_assets/linear_history.json': {'dataset': 'vladkochetov07/e540-v10-artifact-assets-20261008', 'filename': '679b6c7bf3d8b93027c6076ab8bbe6216a9c76d94885a7a06ffeaa64929aaba0.wnn'}, 'e540_assets/linear_model.pt': {'dataset': 'vladkochetov07/e540-v10-artifact-assets-20261008', 'filename': '73880506cf241502d961ce2cba20819085056fef05f5ea8e56d586535bbfb946.wnn'}, 'e540_assets/linear_provenance.json': {'dataset': 'vladkochetov07/e540-v10-artifact-assets-20261008', 'filename': 'ef342abcc11b94eabed66bd4d1e29009fc645079bb17bb95f5f63d83b7e82794.wnn'}, 'e540_assets/ple_epoch0.pt': {'dataset': 'vladkochetov07/e540-v10-artifact-assets-20261008', 'filename': '397b1666000c82534da8196298233786947c317f71dad8adfa2af360c593aa3d.wnn'}, 'e540_assets/ple_history.json': {'dataset': 'vladkochetov07/e540-v10-artifact-assets-20261008', 'filename': '4f99ee2e91f41abcbd28ab1466b76e14bce318fdbc1aff2bd7d241ada3b7a090.wnn'}, 'e540_assets/ple_model.pt': {'dataset': 'vladkochetov07/e540-v10-artifact-assets-20261008', 'filename': '04feca17c7961d7071c74e5a1028d4e9753cab35599464b61daf1f2966b513da.wnn'}, 'e540_assets/ple_provenance.json': {'dataset': 'vladkochetov07/e540-v10-artifact-assets-20261008', 'filename': '8e68f7c7387440dcdfaf16dca5dd532b1cc3a274c67998f8d4cd6f626e52eac2.wnn'}}
KERNEL_PROCESS_WALL_SECONDS_MAX = 6_000
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
            raise TimeoutError("kernel bootstrap exhausted its 6,000-second process budget")
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
                "runner exceeded the 6,000-second kernel-process budget; its process tree was stopped"
            ) from error
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        if remaining_seconds(kernel_deadline) <= 0:
            raise TimeoutError("runner completed outside the 6,000-second kernel-process budget")
    finally:
        release_mounts(root, manifest["external_mounts"])


if __name__ == "__main__":
    main()
