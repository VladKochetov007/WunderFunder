from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

BUNDLE_SLUG = ""
EXPECTED_ARCHIVE_SHA256 = ""
EXPECTED_EXTERNAL_MOUNTS = None
KERNEL_PROCESS_WALL_SECONDS_MAX = 43_200  # Packet builder binds this to the Kaggle push timeout.
RUNNER_FINALIZATION_RESERVE_SECONDS = 300
BOOTSTRAP_STARTED_MONOTONIC = time.monotonic()


def remaining_seconds(deadline: float) -> float:
    """Return wall-clock budget remaining for preflight, work, and collection."""
    return max(0.0, deadline - time.monotonic())


def require_remaining(deadline: float, phase: str) -> None:
    """Stop bootstrap phases that cannot finish before Kaggle's deadline."""
    if remaining_seconds(deadline) <= 0:
        raise TimeoutError(f"kernel bootstrap exceeded its deadline during {phase}")


def terminate_process_tree(process: subprocess.Popen[bytes], *, grace_seconds: float = 2.0) -> None:
    """Stop the runner and any workers after a kernel deadline expires."""
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("cannot enforce kernel deadline without psutil") from error
    try:
        root = psutil.Process(process.pid)
        members = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        process.wait()
        return
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
        try:
            os.kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=grace_seconds)


def worker_timeout_for_deadline(declared: int, deadline: float) -> int:
    """Leave five minutes for runner artifact export and Kaggle shutdown."""
    available = int(deadline - time.monotonic() - RUNNER_FINALIZATION_RESERVE_SECONDS)
    return max(0, min(int(declared), available))


def write_runtime_spec(source: Path, destination: Path, deadline: float) -> None:
    """Copy one sealed spec with its timeout reduced to current kernel headroom."""
    spec = json.loads(source.read_text(encoding="utf-8"))
    limits = spec.get("limits")
    if not isinstance(limits, dict) or not isinstance(limits.get("timeout_seconds"), int):
        raise ValueError("kernel spec has no integer worker timeout")
    timeout = worker_timeout_for_deadline(limits["timeout_seconds"], deadline)
    if timeout < 1:
        raise TimeoutError("kernel deadline leaves no worker time after finalization reserve")
    limits["timeout_seconds"] = timeout
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def execute_preflight_barrier(
    run_preflight: Callable[[], dict[str, Any]],
    verify_preflight: Callable[[dict[str, Any]], None],
    run_candidates: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run candidates only after both runner terminal evidence and input receipt pass."""
    preflight_result = run_preflight()
    verify_preflight(preflight_result)
    return run_candidates()


def run_runner(root: Path, output: Path, specs: list[Path], deadline: float) -> dict[str, Any]:
    """Run one sealed runner phase and require a complete terminal summary."""
    require_remaining(deadline, "runner phase setup")
    command = [sys.executable, str(root / "scripts/kaggle_gpu_runner.py"), "--root", str(root),
               "--output", str(output)]
    for spec in specs:
        command.extend(["--spec", str(spec)])
    environment = os.environ.copy()
    environment["WNN_KERNEL_DEADLINE_MONOTONIC"] = repr(deadline)
    remaining = remaining_seconds(deadline)
    if remaining <= 0:
        raise TimeoutError("kernel wall-time budget expired before runner phase")
    process = subprocess.Popen(command, cwd=root, env=environment, start_new_session=True)
    try:
        return_code = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        terminate_process_tree(process)
        raise TimeoutError("runner phase exceeded the Kaggle kernel wall-time budget")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"runner phase did not write its terminal summary: {output}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("incident") is not None or summary.get("not_started"):
        raise ValueError(f"runner phase did not complete every job: {output}")
    expected_names = [json.loads(path.read_text(encoding="utf-8"))["name"] for path in specs]
    finished = summary.get("finished", [])
    finished_names = [item.get("name") for item in finished]
    if len(finished_names) != len(expected_names) or set(finished_names) != set(expected_names):
        raise ValueError(f"runner terminal job order differs: {output}")
    if any(item.get("status") != "complete" for item in finished):
        raise ValueError(f"runner phase contains an incomplete job: {output}")
    return summary


def verify_preflight_result(root: Path, output: Path, spec: Path) -> dict[str, Any]:
    """Require the sole preflight worker and its actual-data receipt to pass."""
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise ValueError("preflight runner result is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("incident") is not None or summary.get("not_started"):
        raise ValueError("preflight runner did not terminate successfully")
    spec_data = json.loads(spec.read_text(encoding="utf-8"))
    if spec_data.get("kind") != "actual_data_identity_preflight":
        raise ValueError("first sealed job is not the actual-data preflight")
    if len(summary.get("finished", [])) != 1:
        raise ValueError("preflight must complete in an isolated single-job runner phase")
    record = summary["finished"][0]
    if record.get("name") != spec_data["name"] or record.get("status") != "complete":
        raise ValueError("preflight runner terminal record is not complete")
    if record.get("artifacts", {}).get("missing"):
        raise ValueError("preflight receipt was not collected as a declared artifact")
    result_path = output / spec_data["name"] / "result.json"
    if not result_path.is_file():
        raise ValueError("preflight per-job terminal result is missing")
    job_result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        job_result.get("returncode") != 0
        or job_result.get("status") != "complete"
        or job_result.get("artifacts", {}).get("missing")
    ):
        raise ValueError("preflight per-job terminal result is not complete")
    receipt_path = root / "artifacts/qat_kaggle_f0_s4_r1/input_preflight.json"
    if not receipt_path.is_file():
        raise ValueError("actual mounted-input preflight receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "pass" or receipt.get("bytes_hashed", 0) <= 0:
        raise ValueError("actual mounted-input preflight receipt did not pass")
    return receipt


def _cgroup_cache_snapshot() -> dict[str, int | None]:
    """Read bounded cgroup-v2 counters around targeted cache advice."""
    cgroup = Path("/sys/fs/cgroup")
    values: dict[str, int | None] = {}
    for name in ("memory.current", "memory.max"):
        try:
            raw = (cgroup / name).read_text(encoding="ascii").strip()
            values[name] = None if raw == "max" else int(raw)
        except (OSError, ValueError):
            values[name] = None
    try:
        for line in (cgroup / "memory.stat").read_text(encoding="ascii").splitlines():
            key, separator, raw = line.partition(" ")
            if separator and key in {"file", "inactive_file", "active_file", "shmem"}:
                values[f"memory.stat.{key}"] = int(raw)
    except (OSError, ValueError):
        pass
    return values


def advise_hashed_inputs_dontneed(
    receipt: dict[str, Any],
    output: Path,
    *,
    allowed_root: Path = Path("/kaggle/input/datasets"),
) -> dict[str, Any]:
    """Drop cache for only the immutable files successfully hashed by preflight.

    Paths, observed byte counts, and digests are taken from the actual-input
    preflight receipt. No payload is read or modified in this handoff.
    """
    if receipt.get("status") != "pass" or not isinstance(receipt.get("inputs"), list):
        raise ValueError("cache handoff requires a passing actual-input receipt")
    if not callable(getattr(os, "posix_fadvise", None)) or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable; refusing candidate admission")

    root = allowed_root.resolve(strict=True)
    files: dict[Path, dict[str, Any]] = {}
    for item in receipt["inputs"]:
        if not isinstance(item, dict):
            raise ValueError("preflight input identity is malformed")
        expected_size, observed_size = item.get("expected_size"), item.get("size")
        expected_hash, observed_hash = item.get("expected_sha256"), item.get("sha256")
        if (
            not isinstance(expected_size, int)
            or expected_size <= 0
            or observed_size != expected_size
            or not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or observed_hash != expected_hash
        ):
            raise ValueError("preflight input identity is not hash- and size-verified")
        resolved = Path(item.get("resolved_path", "")).resolve(strict=True)
        if not resolved.is_relative_to(root) or not stat.S_ISREG(resolved.stat().st_mode):
            raise ValueError("preflight input resolves outside the authorized dataset mount")
        if resolved.stat().st_size != expected_size:
            raise ValueError("preflight input length changed before cache handoff")
        previous = files.get(resolved)
        identity = {
            "dataset": item.get("dataset"),
            "path": item.get("path"),
            "size": expected_size,
            "sha256": expected_hash,
        }
        if previous is not None and previous != identity:
            raise ValueError("conflicting preflight identities resolve to one file")
        files[resolved] = identity
    if not files or sum(item["size"] for item in files.values()) != receipt.get("bytes_hashed"):
        raise ValueError("preflight hashed-byte total does not match its unique inputs")

    record: dict[str, Any] = {
        "schema": "qat-kaggle-targeted-cache-handoff-r2",
        "status": "started",
        "advice": "POSIX_FADV_DONTNEED",
        "files_expected": len(files),
        "bytes_expected": sum(item["size"] for item in files.values()),
        "preflight_identity_manifest_sha256": receipt.get("identity_manifest_sha256"),
        "before": _cgroup_cache_snapshot(),
        "files": [],
    }
    try:
        for path, identity in files.items():
            with path.open("rb") as source:
                opened = os.fstat(source.fileno())
                if not stat.S_ISREG(opened.st_mode) or opened.st_size != identity["size"]:
                    raise ValueError("preflight input changed while opening for cache advice")
                os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            record["files"].append({**identity, "advice_returned": "success"})
    except (OSError, ValueError) as error:
        record.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        record["after"] = _cgroup_cache_snapshot()
        write_phase_receipt(output, record)
        raise RuntimeError("targeted cache handoff failed; candidate admission blocked") from error
    record.update({"status": "pass", "after": _cgroup_cache_snapshot()})
    write_phase_receipt(output, record)
    return record


def write_phase_receipt(path: Path, value: dict) -> None:
    """Persist the preflight barrier and candidate phase handoff for collection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def release_mounts(root: Path, mounts: dict) -> None:
    """Remove only external symlinks before Kaggle captures local results."""
    for logical in mounts:
        target = root / logical
        if target.is_symlink():
            target.unlink()


def main() -> None:
    """Preflight actual private mounts alone, then run the candidate tranche."""
    deadline = BOOTSTRAP_STARTED_MONOTONIC + KERNEL_PROCESS_WALL_SECONDS_MAX
    require_remaining(deadline, "input discovery")
    matches = list(Path("/kaggle/input").glob(f"**/{BUNDLE_SLUG}/bundle-manifest.json"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one named private bundle")
    manifest_path = matches[0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("private") is not True or manifest.get("external_mounts") != EXPECTED_EXTERNAL_MOUNTS:
        raise ValueError("Bundle privacy or external mappings differ from the controller-bound kernel")
    archive = manifest_path.parent / "bundle.wnn"
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest.get("archive_sha256") or digest.hexdigest() != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Private bundle archive identity mismatch")
    root = Path("/kaggle/working/experiment")
    if shutil.disk_usage(root.parent).free < manifest["expanded_bytes"] + 2 * 1024**3:
        raise ValueError("Insufficient working disk to extract bundle with reserve")
    root.mkdir(exist_ok=False)
    allowed = {item["path"] for item in manifest["files"]} | set(manifest["spec_paths"])
    seen: set[str] = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            require_remaining(deadline, "bundle extraction")
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
        require_remaining(deadline, "external input mounting")
        path = PurePosixPath(logical)
        if path.is_absolute() or ".." in path.parts or logical not in allowed:
            raise ValueError("Unsafe external input mapping")
        dataset = mapping["dataset"] if isinstance(mapping, dict) else "vladkochetov07/wunder-funder"
        filename = mapping["filename"] if isinstance(mapping, dict) else mapping
        slug = dataset.split("/")[1]
        candidates = list(Path("/kaggle/input").glob(f"**/{slug}/{filename}"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one authorized input: {dataset}/{filename}")
        target = root / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(candidates[0])

    try:
        specs = [root / name for name in manifest["spec_paths"]]
        if len(specs) != 4:
            raise ValueError("QAT packet requires one preflight and exactly three matched candidates")
        spec_data = [json.loads(path.read_text(encoding="utf-8")) for path in specs]
        if spec_data[0].get("kind") != "actual_data_identity_preflight":
            raise ValueError("first sealed job is not the actual-data preflight")
        if any(item.get("kind") == "actual_data_identity_preflight" for item in spec_data[1:]):
            raise ValueError("preflight appears more than once in the sealed stage list")
        if any(item.get("kind") not in {"full_shape_qat_profile", "full_epoch_matched_qat_fit"}
               for item in spec_data[1:]):
            raise ValueError("candidate stage contains an unsupported job kind")
        if len({item.get("kind") for item in spec_data[1:]}) != 1:
            raise ValueError("profile and fit jobs cannot share a Kaggle execution packet")

        runtime_dir = Path("/kaggle/working/runtime_specs")
        runtime_specs = []
        for source in specs:
            require_remaining(deadline, "runtime deadline setup")
            destination = runtime_dir / f"{json.loads(source.read_text(encoding='utf-8'))['name']}.json"
            write_runtime_spec(source, destination, deadline)
            runtime_specs.append(destination)

        def run_preflight() -> dict:
            output = Path("/kaggle/working/results/preflight")
            return run_runner(root, output, [runtime_specs[0]], deadline)

        def verify_preflight(summary: dict) -> dict:
            record = summary["finished"][0]
            if record.get("artifacts", {}).get("missing"):
                raise ValueError("preflight runner did not collect its declared input receipt")
            return verify_preflight_result(root, Path("/kaggle/working/results/preflight"), runtime_specs[0])

        def run_candidates() -> dict:
            output = Path("/kaggle/working/results/candidates")
            return run_runner(root, output, runtime_specs[1:], deadline)

        preflight_receipt: dict[str, Any] = {}
        cache_handoff_receipt: dict[str, Any] = {}

        def capture_preflight(summary: dict[str, Any]) -> None:
            nonlocal preflight_receipt, cache_handoff_receipt
            preflight_receipt = verify_preflight(summary)
            cache_handoff_receipt = advise_hashed_inputs_dontneed(
                preflight_receipt, Path("/kaggle/working/results/preflight/cache_handoff.json")
            )

        # Keep the guard and invocation visibly sequential: phase two is unreachable
        # until the isolated runner's terminal record and receipt both pass.
        def verify(summary: dict[str, Any]) -> None:
            capture_preflight(summary)

        candidates_summary = execute_preflight_barrier(run_preflight, verify, run_candidates)
        write_phase_receipt(Path("/kaggle/working/results/qat_sequence.json"), {
            "schema": "qat-kaggle-two-phase-execution-r1",
            "status": "complete",
            "preflight": {
                "status": preflight_receipt["status"],
                "bytes_hashed": preflight_receipt["bytes_hashed"],
                "identity_manifest_sha256": preflight_receipt["identity_manifest_sha256"],
                "isolated_runner_output": "/kaggle/working/results/preflight",
            },
            "cache_handoff": {
                "status": cache_handoff_receipt["status"],
                "files_advised": cache_handoff_receipt["files_expected"],
                "bytes_advised": cache_handoff_receipt["bytes_expected"],
                "before": cache_handoff_receipt["before"],
                "after": cache_handoff_receipt["after"],
                "receipt": "/kaggle/working/results/preflight/cache_handoff.json",
            },
            "candidate_phase": {
                "status": "complete",
                "runner_output": "/kaggle/working/results/candidates",
                "jobs": [item["name"] for item in candidates_summary["finished"]],
                "concurrency_admitted": candidates_summary["concurrency_admitted"],
            },
            "source_bundle_sha256": EXPECTED_ARCHIVE_SHA256,
        })
    finally:
        release_mounts(root, manifest["external_mounts"])


if __name__ == "__main__":
    main()
